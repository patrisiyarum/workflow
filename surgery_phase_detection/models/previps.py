"""
PreViPS: Multi-view Video-Pose Pretraining for Surgical Activity Recognition
==============================================================================

Implements the PreViPS framework (arXiv:2502.13883): a CLIP-style dual-encoder
that aligns video and tokenized 2D pose embeddings across multiple camera views.

Architecture:
    - Video Encoder: MViT-S / ViT with MaskFeat pretraining (or ResNet fallback)
    - Pose Encoder: PoseTokenizer + PoseTransformerEncoder
    - Pretraining: Cross/in-modality contrastive + geometric consistency +
                   masked pose modeling
    - Finetuning: Average-pooled global tokens from all views/modalities -> MLP

Reference: Hamoud et al., "Multi-view Video-Pose Pretraining for Operating Room
Surgical Activity Recognition", 2025.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .pose_encoder import PoseTokenizer, PoseTransformerEncoder


# ---------------------------------------------------------------------------
# Pretraining Loss Functions
# ---------------------------------------------------------------------------


def info_nce_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss (CLIP-style).

    Brings corresponding (a_i, b_i) pairs together while pushing
    non-corresponding pairs apart.

    Args:
        embeddings_a: (N, D) normalised embeddings from modality/view A.
        embeddings_b: (N, D) normalised embeddings from modality/view B.
        temperature: Softmax temperature.

    Returns:
        Scalar InfoNCE loss.
    """
    logits = embeddings_a @ embeddings_b.T / temperature  # (N, N)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def cross_modality_geometric_loss(
    video_embeds: Dict[int, torch.Tensor],
    pose_embeds: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Cross-modality geometric consistency (Eq. 6 in PreViPS).

    Ensures similarity scores for video-pose pairs are symmetric across views:
        L_C-Geo = sum_{p,q} (sim(I_p, J_q) - sim(J_p, I_q))^2

    Args:
        video_embeds: {view_id: (N, D)} video embeddings per view.
        pose_embeds: {view_id: (N, D)} pose embeddings per view.
    """
    loss = torch.tensor(0.0, device=next(iter(video_embeds.values())).device)
    views = list(video_embeds.keys())
    count = 0
    for i, p in enumerate(views):
        for q in views[i:]:
            sim_ip_jq = (video_embeds[p] * pose_embeds[q]).sum(dim=-1)
            sim_jp_iq = (pose_embeds[p] * video_embeds[q]).sum(dim=-1)
            loss = loss + ((sim_ip_jq - sim_jp_iq) ** 2).mean()
            count += 1
    return loss / max(count, 1)


def in_modality_geometric_loss(
    video_embeds: Dict[int, torch.Tensor],
    pose_embeds: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """In-modality geometric consistency (Eq. 7 in PreViPS).

    Ensures cross-view similarity is consistent between modalities:
        L_I-Geo = sum_{p,q} (sim(I_p, I_q) - sim(J_p, J_q))^2

    Args:
        video_embeds: {view_id: (N, D)} video embeddings per view.
        pose_embeds: {view_id: (N, D)} pose embeddings per view.
    """
    loss = torch.tensor(0.0, device=next(iter(video_embeds.values())).device)
    views = list(video_embeds.keys())
    count = 0
    for i, p in enumerate(views):
        for q in views[i + 1:]:
            sim_ii = (video_embeds[p] * video_embeds[q]).sum(dim=-1)
            sim_jj = (pose_embeds[p] * pose_embeds[q]).sum(dim=-1)
            loss = loss + ((sim_ii - sim_jj) ** 2).mean()
            count += 1
    return loss / max(count, 1)


# ---------------------------------------------------------------------------
# Video Encoder (ResNet-based fallback; MViT-S preferred in full PreViPS)
# ---------------------------------------------------------------------------


class VideoEncoder(nn.Module):
    """Video encoder that extracts per-clip visual features.

    Uses a ResNet backbone with temporal average pooling as a practical
    fallback.  For full PreViPS reproduction, replace with MViT-S + MaskFeat.

    Args:
        backbone: ResNet variant.
        pretrained: Use ImageNet pretrained weights.
        embed_dim: Output embedding dimension (projected from backbone features).
    """

    FEATURE_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 256,
    ):
        super().__init__()

        self.feat_dim = self.FEATURE_DIMS.get(backbone, 2048)

        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            model = models.resnet34(weights=weights)
        else:
            model = models.resnet50(weights=weights)

        self.cnn = nn.Sequential(*list(model.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video clip (B, T, C, H, W) or single image (B, C, H, W).

        Returns:
            Global video embedding (B, embed_dim).
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            feats = self.cnn(x).flatten(1)     # (B*T, feat_dim)
            feats = feats.reshape(B, T, -1).mean(dim=1)  # temporal avg
        else:
            feats = self.cnn(x).flatten(1)

        return self.proj(feats)


# ---------------------------------------------------------------------------
# PreViPS: Full Dual-Encoder Framework
# ---------------------------------------------------------------------------


class PreViPS(nn.Module):
    """PreViPS dual-encoder for video-pose pretraining and finetuning.

    In pretraining mode, returns per-view video/pose embeddings and
    masked-pose predictions for computing the alignment loss.
    In finetuning mode, averages global tokens across views/modalities
    and classifies with an MLP head.

    Args:
        num_classes: Number of activity/phase classes.
        embed_dim: Shared embedding dimension.
        backbone: CNN backbone for the video encoder.
        pretrained: Use ImageNet pretrained weights.
        num_joints: Keypoints per person (17 for COCO).
        codebook_size: VQ codebook size for pose tokenizer.
        num_pose_tokens: Tokens per pose from the tokenizer.
        pose_layers: Transformer layers in the pose encoder.
        pose_heads: Attention heads in the pose encoder.
        max_persons: Max people per frame.
        max_frames: Max frames per clip.
        max_views: Max camera views.
        dropout: Dropout probability.
        temperature: InfoNCE temperature.
        lambda_geo: Weight for geometric consistency losses.
        lambda_mask: Weight for masked pose loss.
        mask_ratio: Fraction of pose tokens to mask during pretraining.
    """

    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 256,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_joints: int = 17,
        codebook_size: int = 512,
        num_pose_tokens: int = 4,
        pose_layers: int = 6,
        pose_heads: int = 8,
        max_persons: int = 8,
        max_frames: int = 16,
        max_views: int = 6,
        dropout: float = 0.1,
        temperature: float = 0.07,
        lambda_geo: float = 0.5,
        lambda_mask: float = 0.5,
        mask_ratio: float = 0.15,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.lambda_geo = lambda_geo
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio

        # ----- Video Encoder -----
        self.video_encoder = VideoEncoder(
            backbone=backbone, pretrained=pretrained, embed_dim=embed_dim,
        )

        # ----- Pose Pipeline -----
        self.pose_tokenizer = PoseTokenizer(
            num_joints=num_joints,
            embed_dim=embed_dim,
            codebook_size=codebook_size,
            num_tokens=num_pose_tokens,
        )

        self.pose_encoder = PoseTransformerEncoder(
            embed_dim=embed_dim,
            num_layers=pose_layers,
            num_heads=pose_heads,
            max_persons=max_persons,
            max_frames=max_frames,
            max_views=max_views,
            num_tokens_per_pose=num_pose_tokens,
            dropout=dropout,
        )

        # ----- Masked Pose Decoder -----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=pose_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.mask_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.mask_pred_head = nn.Linear(embed_dim, num_joints * 2)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ----- Finetuning Classifier -----
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    # ----- Pretraining Forward -----

    def pretrain_forward(
        self,
        video_clips: Dict[int, torch.Tensor],
        pose_sequences: Dict[int, torch.Tensor],
        time_ids: Optional[Dict[int, torch.Tensor]] = None,
        person_ids: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pretraining forward pass across multiple views.

        Args:
            video_clips: {view_id: (B, T, C, H, W)} video clips per view.
            pose_sequences: {view_id: (B, N_persons*T, num_joints, 2)}
                            raw 2D pose coordinates per view.
            time_ids: {view_id: (B, N*T)} frame indices.
            person_ids: {view_id: (B, N*T)} person indices.

        Returns:
            Dictionary of losses: contrastive, geometric, mask, vq, total.
        """
        video_embeds = {}  # {view: (B, D)}
        pose_embeds = {}   # {view: (B, D)}

        total_vq_loss = 0.0
        total_mask_loss = 0.0

        for view_id in video_clips:
            # --- Video ---
            v_emb = self.video_encoder(video_clips[view_id])
            v_emb = F.normalize(v_emb, dim=-1)
            video_embeds[view_id] = v_emb

            # --- Pose ---
            raw_poses = pose_sequences[view_id]  # (B, N*T, J, 2)
            B, S, J, _ = raw_poses.shape

            # Tokenize all poses
            flat_poses = raw_poses.reshape(B * S, J, 2)
            tokens, decoded, vq_loss = self.pose_tokenizer(flat_poses)
            total_vq_loss = total_vq_loss + vq_loss
            # tokens: (B*S, num_tokens, D)
            tokens = tokens.reshape(B, S, -1, self.embed_dim)
            # (B, S, K, D)

            # Masked pose modeling
            mask_loss, tokens_masked = self._masked_pose_forward(
                tokens, flat_poses, B, S
            )
            total_mask_loss = total_mask_loss + mask_loss

            # Pose encoder
            t_ids = time_ids[view_id] if time_ids else None
            p_ids = person_ids[view_id] if person_ids else None

            cls_emb, _ = self.pose_encoder(
                tokens_masked, time_ids=t_ids, person_ids=p_ids, view_id=view_id
            )
            cls_emb = F.normalize(cls_emb, dim=-1)
            pose_embeds[view_id] = cls_emb

        # ----- Contrastive Loss (CLIP*) -----
        views = list(video_embeds.keys())
        con_loss = torch.tensor(0.0, device=v_emb.device)
        count = 0

        for p in views:
            for q in views:
                # Cross-modality: video_p <-> pose_q
                con_loss = con_loss + info_nce_loss(
                    video_embeds[p], pose_embeds[q], self.temperature
                )
                count += 1

        # In-modality: video_p <-> video_q, pose_p <-> pose_q
        for i, p in enumerate(views):
            for q in views[i + 1:]:
                con_loss = con_loss + info_nce_loss(
                    video_embeds[p], video_embeds[q], self.temperature
                )
                con_loss = con_loss + info_nce_loss(
                    pose_embeds[p], pose_embeds[q], self.temperature
                )
                count += 2

        con_loss = con_loss / max(count, 1)

        # ----- Geometric Consistency -----
        geo_cross = cross_modality_geometric_loss(video_embeds, pose_embeds)
        geo_in = in_modality_geometric_loss(video_embeds, pose_embeds)
        geo_loss = geo_cross + geo_in

        # ----- Total Loss -----
        n_views = len(views)
        total = (
            con_loss
            + self.lambda_geo * geo_loss
            + self.lambda_mask * (total_mask_loss / n_views)
            + total_vq_loss / n_views
        )

        return {
            "total": total,
            "contrastive": con_loss.detach(),
            "geometric": geo_loss.detach(),
            "mask": (total_mask_loss / n_views).detach() if isinstance(total_mask_loss, torch.Tensor) else total_mask_loss,
            "vq": (total_vq_loss / n_views).detach() if isinstance(total_vq_loss, torch.Tensor) else total_vq_loss,
        }

    def _masked_pose_forward(
        self,
        tokens: torch.Tensor,
        flat_poses: torch.Tensor,
        B: int,
        S: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to pose tokens and predict the masked ones.

        Args:
            tokens: (B, S, K, D) tokenised pose embeddings.
            flat_poses: (B*S, J, 2) original poses for reconstruction target.
            B: batch size.
            S: sequence length (N_persons * T).

        Returns:
            mask_loss: MSE reconstruction loss on masked poses.
            tokens_out: (B, S, K, D) tokens with some replaced by mask_token.
        """
        K, D = tokens.shape[2], tokens.shape[3]
        num_mask = max(1, int(S * self.mask_ratio))

        # Random mask indices per batch
        mask_indices = torch.stack([
            torch.randperm(S, device=tokens.device)[:num_mask] for _ in range(B)
        ])  # (B, num_mask)

        # Replace masked positions with learnable mask token
        tokens_out = tokens.clone()
        mask_expand = self.mask_token.expand(B, 1, K, D)
        for b in range(B):
            tokens_out[b, mask_indices[b]] = mask_expand[b].expand(num_mask, K, D)

        # Extract encoder features for masked positions
        # (simplified: use the masked tokens directly for decoding)
        masked_feats = tokens_out.reshape(B, S * K, D)

        # Decode
        decoded = self.mask_decoder(
            self.mask_token.expand(B, num_mask, -1),
            masked_feats,
        )  # (B, num_mask, D)

        pred_coords = self.mask_pred_head(decoded)  # (B, num_mask, J*2)

        # Target: original poses at masked positions
        target_poses = flat_poses.reshape(B, S, -1)  # (B, S, J*2)
        target = torch.stack([
            target_poses[b, mask_indices[b]] for b in range(B)
        ])  # (B, num_mask, J*2)

        mask_loss = F.mse_loss(pred_coords, target)

        return mask_loss, tokens_out

    # ----- Finetuning Forward -----

    def forward(
        self,
        video_clips: torch.Tensor,
        pose_sequences: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        person_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Finetuning forward pass (single view, classification).

        Args:
            video_clips: (B, T, C, H, W) video clips.
            pose_sequences: Optional (B, N*T, J, 2) pose coordinates.
            time_ids: Optional (B, N*T) frame indices.
            person_ids: Optional (B, N*T) person indices.

        Returns:
            logits: (B, num_classes) classification logits.
        """
        embeddings = []

        # Video
        v_emb = self.video_encoder(video_clips)
        embeddings.append(v_emb)

        # Pose (optional)
        if pose_sequences is not None:
            B, S, J, _ = pose_sequences.shape
            flat_poses = pose_sequences.reshape(B * S, J, 2)
            tokens, _, _ = self.pose_tokenizer(flat_poses)
            tokens = tokens.reshape(B, S, -1, self.embed_dim)

            cls_emb, _ = self.pose_encoder(
                tokens, time_ids=time_ids, person_ids=person_ids
            )
            embeddings.append(cls_emb)

        # Average pool all available embeddings
        combined = torch.stack(embeddings, dim=0).mean(dim=0)  # (B, D)

        return self.classifier(combined)

    # ----- Utilities -----

    def freeze_video_encoder(self, num_layers: int = 12) -> None:
        """Freeze early layers of the video encoder."""
        for i, (name, param) in enumerate(self.video_encoder.named_parameters()):
            if i < num_layers:
                param.requires_grad = False

    def get_num_params(self) -> Dict[str, int]:
        """Return parameter counts per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "video_encoder": _count(self.video_encoder),
            "pose_tokenizer": _count(self.pose_tokenizer),
            "pose_encoder": _count(self.pose_encoder),
            "mask_decoder": _count(self.mask_decoder),
            "classifier": _count(self.classifier),
            "total": sum(p.numel() for p in self.parameters()),
        }
