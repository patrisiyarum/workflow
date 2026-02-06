"""
Multi-View Surgery Phase Detection Network
============================================

An architecture designed for multi-view operating room data (MVOR).
Processes multiple camera views through a shared CNN backbone and fuses
the features before temporal modeling with an LSTM.

Fusion strategies:
    - 'concat': Concatenate features from all views
    - 'mean': Average features across views
    - 'max': Max-pool features across views
    - 'attention': Learned attention-weighted fusion
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ViewAttentionFusion(nn.Module):
    """Learned attention-based fusion of multi-view features.

    Computes a soft attention weight for each view and returns the
    weighted sum of view features.

    Args:
        feature_dim: Dimensionality of per-view features.
        num_views: Number of camera views.
    """

    def __init__(self, feature_dim: int, num_views: int = 3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
        )
        self.num_views = num_views

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_features: (B, num_views, feature_dim)

        Returns:
            Fused features: (B, feature_dim)
        """
        # Compute attention scores: (B, num_views, 1)
        scores = self.attention(view_features)
        weights = torch.softmax(scores, dim=1)  # (B, num_views, 1)

        # Weighted sum: (B, feature_dim)
        fused = (view_features * weights).sum(dim=1)
        return fused


class MultiViewSurgeryNet(nn.Module):
    """Multi-view ResNet + LSTM for operating room phase detection.

    Processes each camera view through a shared ResNet backbone, fuses the
    view features, then models temporal dependencies with an LSTM.

    Args:
        num_classes: Number of surgical phases.
        backbone: ResNet variant.
        pretrained: Use ImageNet pretrained weights.
        num_views: Number of camera views (3 for MVOR).
        fusion: Fusion strategy ('concat', 'mean', 'max', 'attention').
        lstm_hidden: LSTM hidden dimension.
        lstm_layers: Number of LSTM layers.
        dropout: Dropout probability.
        freeze_backbone: Freeze CNN parameters initially.
    """

    BACKBONE_FEATURES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_views: int = 3,
        fusion: str = "attention",
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_views = num_views
        self.fusion_type = fusion
        self.feature_dim = self.BACKBONE_FEATURES.get(backbone, 2048)

        # Shared CNN backbone
        self.cnn = self._build_backbone(backbone, pretrained)

        if freeze_backbone:
            self.freeze_cnn()

        # View fusion
        if fusion == "concat":
            fused_dim = self.feature_dim * num_views
        elif fusion in ("mean", "max"):
            fused_dim = self.feature_dim
        elif fusion == "attention":
            self.view_attention = ViewAttentionFusion(self.feature_dim, num_views)
            fused_dim = self.feature_dim
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.fused_dim = fused_dim

        # Temporal model
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def _build_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            model = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        modules = list(model.children())[:-1]
        return nn.Sequential(*modules)

    def freeze_cnn(self) -> None:
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self) -> None:
        for param in self.cnn.parameters():
            param.requires_grad = True

    def _fuse_views(self, view_features: torch.Tensor) -> torch.Tensor:
        """Fuse features from multiple camera views.

        Args:
            view_features: (B, num_views, feature_dim)

        Returns:
            Fused feature vector: (B, fused_dim)
        """
        if self.fusion_type == "concat":
            return view_features.reshape(view_features.size(0), -1)
        elif self.fusion_type == "mean":
            return view_features.mean(dim=1)
        elif self.fusion_type == "max":
            return view_features.max(dim=1).values
        elif self.fusion_type == "attention":
            return self.view_attention(view_features)
        else:
            return view_features.mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor.
               For multi-view: (B, T, V, C, H, W) where V = num_views
               For single-view: (B, T, C, H, W)
            hidden: Optional LSTM initial hidden state.

        Returns:
            Tuple of (logits, hidden_state).
        """
        if x.dim() == 6:
            # Multi-view input: (B, T, V, C, H, W)
            B, T, V, C, H, W = x.shape
            # Process all frames and views through CNN
            x_flat = x.reshape(B * T * V, C, H, W)
            features = self.cnn(x_flat).flatten(1)  # (B*T*V, feat_dim)
            features = features.reshape(B, T, V, self.feature_dim)

            # Fuse views at each time step
            fused = []
            for t in range(T):
                fused.append(self._fuse_views(features[:, t]))
            fused = torch.stack(fused, dim=1)  # (B, T, fused_dim)
        elif x.dim() == 5:
            # Single-view input: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
            features = self.cnn(x_flat).flatten(1)
            fused = features.reshape(B, T, self.feature_dim)
        else:
            raise ValueError(f"Expected 5D or 6D input, got {x.dim()}D")

        # LSTM temporal modeling
        lstm_out, hidden = self.lstm(fused, hidden)
        last_output = lstm_out[:, -1, :]

        # Classify
        logits = self.classifier(last_output)

        return logits, hidden
