"""
Pose Encoder with Compositional Token Representation
======================================================

Implements the pose encoding pipeline inspired by PreViPS (arXiv:2502.13883).
Converts continuous 2D human pose keypoints into discrete token representations
using a VQ-VAE-style tokenizer, then processes them through a transformer
encoder with spatio-temporal positional embeddings.

Key components:
    - PoseTokenizer: VQ-VAE that converts continuous (x,y) joint coordinates
      into discrete codebook embeddings (Pose as Compositional Tokens).
    - PoseTransformerEncoder: Processes tokenized pose sequences with multi-head
      self-attention, incorporating time, person-ID, and view positional embeddings.

Reference: "Multi-view Video-Pose Pretraining for Operating Room Surgical
Activity Recognition" (Hamoud et al., 2025)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseTokenizer(nn.Module):
    """Vector-Quantized pose tokenizer (Pose as Compositional Tokens).

    Encodes continuous 2D joint coordinates into discrete codebook indices,
    producing compact per-person pose embeddings that handle occlusions by
    modelling inter-joint dependencies.

    Args:
        num_joints: Number of body keypoints (17 for COCO, 10 for CAMMA).
        coord_dim: Coordinate dimension per joint (2 for 2D poses).
        embed_dim: Dimensionality of the codebook embeddings.
        codebook_size: Number of entries in the discrete codebook.
        num_tokens: Number of tokens per pose (factorised representation).
        commitment_cost: Weight for the commitment loss term in VQ training.
    """

    def __init__(
        self,
        num_joints: int = 17,
        coord_dim: int = 2,
        embed_dim: int = 256,
        codebook_size: int = 512,
        num_tokens: int = 4,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.num_tokens = num_tokens
        self.commitment_cost = commitment_cost

        input_dim = num_joints * coord_dim

        # Encoder: continuous pose -> latent vectors
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim * num_tokens),
        )

        # Discrete codebook
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(
            self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size
        )

        # Decoder: codebook embeddings -> reconstructed pose
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * num_tokens, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode poses to latent vectors and quantize.

        Args:
            poses: (B, num_joints, coord_dim) or (B, num_joints * coord_dim)

        Returns:
            quantized: (B, num_tokens, embed_dim) quantized embeddings
            indices: (B, num_tokens) codebook indices
        """
        if poses.dim() == 3:
            poses = poses.reshape(poses.size(0), -1)

        z = self.encoder(poses)  # (B, embed_dim * num_tokens)
        z = z.view(-1, self.num_tokens, self.embed_dim)  # (B, T, D)

        # Quantize each token independently
        distances = torch.cdist(z, self.codebook.weight.unsqueeze(0))  # (B, T, K)
        indices = distances.argmin(dim=-1)  # (B, T)
        quantized = self.codebook(indices)  # (B, T, D)

        return quantized, indices

    def forward(
        self, poses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, quantize, decode.

        Args:
            poses: (B, num_joints, coord_dim) continuous poses.

        Returns:
            quantized: (B, num_tokens, embed_dim) token embeddings.
            reconstructed: (B, num_joints * coord_dim) reconstructed poses.
            vq_loss: Scalar VQ-VAE loss (codebook + commitment).
        """
        if poses.dim() == 3:
            flat = poses.reshape(poses.size(0), -1)
        else:
            flat = poses

        z = self.encoder(flat).view(-1, self.num_tokens, self.embed_dim)

        # Quantize
        distances = torch.cdist(z, self.codebook.weight.unsqueeze(0))
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices)

        # VQ losses
        codebook_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # Decode
        decoded = self.decoder(quantized_st.reshape(-1, self.num_tokens * self.embed_dim))

        return quantized_st, decoded, vq_loss


class PoseTransformerEncoder(nn.Module):
    """Transformer encoder for tokenized pose sequences.

    Processes per-frame, per-person pose token sequences with spatio-temporal
    positional embeddings. Outputs a global [CLS] embedding for each camera
    view that can be used for downstream classification or pretraining.

    Args:
        embed_dim: Token embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_persons: Maximum persons per frame.
        max_frames: Maximum frames per clip.
        max_views: Maximum camera views.
        num_tokens_per_pose: Number of tokens produced by the tokenizer per pose.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_persons: int = 8,
        max_frames: int = 16,
        max_views: int = 6,
        num_tokens_per_pose: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_persons = max_persons
        self.max_frames = max_frames
        self.num_tokens_per_pose = num_tokens_per_pose

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        self.time_embed = self._sinusoidal_embedding(max_frames, embed_dim)
        self.person_embed = self._sinusoidal_embedding(max_persons, embed_dim)
        self.view_embed = nn.Embedding(max_views, embed_dim)

        # Token projection (if input dim differs from embed_dim)
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _sinusoidal_embedding(max_len: int, dim: int) -> nn.Embedding:
        """Create fixed sinusoidal positional embeddings."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        embed = nn.Embedding(max_len, dim)
        embed.weight = nn.Parameter(pe, requires_grad=False)
        return embed

    def forward(
        self,
        pose_tokens: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        person_ids: Optional[torch.Tensor] = None,
        view_id: int = 0,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the pose transformer.

        Args:
            pose_tokens: (B, N_persons * T, num_tokens_per_pose, embed_dim)
                         or already flattened (B, seq_len, embed_dim).
            time_ids: (B, seq_len) frame index for each token.
            person_ids: (B, seq_len) person index for each token.
            view_id: Camera view index (int).
            padding_mask: (B, seq_len) True for padded positions.

        Returns:
            cls_embed: (B, embed_dim) global [CLS] embedding.
            all_embeds: (B, seq_len+1, embed_dim) all token embeddings.
        """
        B = pose_tokens.size(0)

        # Flatten tokens if needed: (B, N*T, K, D) -> (B, N*T*K, D)
        num_sub_tokens = 1
        if pose_tokens.dim() == 4:
            N, K, D = pose_tokens.shape[1], pose_tokens.shape[2], pose_tokens.shape[3]
            num_sub_tokens = K
            pose_tokens = pose_tokens.reshape(B, N * K, D)

        seq_len = pose_tokens.size(1)

        # Project tokens
        tokens = self.token_proj(pose_tokens)

        # Add positional embeddings
        # When tokens were flattened from (B, N, K, D), repeat IDs K times
        if time_ids is not None:
            if num_sub_tokens > 1 and time_ids.size(1) * num_sub_tokens == seq_len:
                time_ids = time_ids.repeat_interleave(num_sub_tokens, dim=1)
            tokens = tokens + self.time_embed(time_ids)
        if person_ids is not None:
            if num_sub_tokens > 1 and person_ids.size(1) * num_sub_tokens == seq_len:
                person_ids = person_ids.repeat_interleave(num_sub_tokens, dim=1)
            tokens = tokens + self.person_embed(person_ids)

        # Add view embedding
        view_emb = self.view_embed(
            torch.tensor(view_id, device=tokens.device)
        ).unsqueeze(0).unsqueeze(0)
        tokens = tokens + view_emb

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Adjust padding mask for [CLS]
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer
        out = self.transformer(tokens, src_key_padding_mask=padding_mask)
        out = self.norm(out)

        cls_embed = out[:, 0]     # (B, embed_dim)
        all_embeds = out          # (B, seq_len+1, embed_dim)

        return cls_embed, all_embeds
