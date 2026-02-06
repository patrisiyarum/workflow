"""
ResNet + LSTM Model for Surgery Phase Detection
=================================================

A two-stage architecture:
1. ResNet-50 extracts spatial features from each video frame.
2. A multi-layer LSTM models temporal dependencies across a sequence of frames.

The final classification is produced by a fully connected layer on top of
the LSTM output for the last time step.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class SurgeryPhaseNet(nn.Module):
    """ResNet + LSTM network for temporal surgical phase recognition.

    Args:
        num_classes: Number of surgical phase classes.
        backbone: ResNet variant to use ('resnet18', 'resnet34', 'resnet50').
        pretrained: Whether to use ImageNet-pretrained weights.
        feature_dim: Dimensionality of CNN features (set automatically).
        lstm_hidden: LSTM hidden state dimension.
        lstm_layers: Number of stacked LSTM layers.
        dropout: Dropout probability.
        bidirectional: Whether the LSTM is bidirectional.
        freeze_backbone: Whether to freeze the CNN backbone initially.
    """

    BACKBONE_FEATURES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "resnet50",
        pretrained: bool = True,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # ---- CNN Backbone ----
        self.feature_dim = self.BACKBONE_FEATURES.get(backbone, 2048)
        self.cnn = self._build_backbone(backbone, pretrained)

        if freeze_backbone:
            self.freeze_cnn()

        # ---- Temporal Model ----
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- Classifier Head ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _build_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Construct the CNN backbone without its classification head."""
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            model = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final FC layer; keep up to avgpool
        modules = list(model.children())[:-1]
        return nn.Sequential(*modules)

    def _init_weights(self) -> None:
        """Initialize LSTM and classifier weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_cnn(self) -> None:
        """Freeze all parameters in the CNN backbone."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("CNN backbone frozen.")

    def unfreeze_cnn(self) -> None:
        """Unfreeze all parameters in the CNN backbone."""
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("CNN backbone unfrozen.")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features from a batch of images.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        features = self.cnn(x)
        return features.flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the full ResNet + LSTM model.

        Args:
            x: Input tensor of shape (B, T, C, H, W) where T is sequence length.
            hidden: Optional initial LSTM hidden state (h_0, c_0).

        Returns:
            Tuple of:
                - logits: Classification logits of shape (B, num_classes).
                - hidden: Final LSTM hidden state (h_n, c_n).
        """
        batch_size, seq_len, C, H, W = x.shape

        # Reshape to process all frames through CNN at once
        x = x.view(batch_size * seq_len, C, H, W)

        # Extract spatial features: (B*T, feature_dim)
        features = self.extract_features(x)

        # Reshape back to sequences: (B, T, feature_dim)
        features = features.view(batch_size, seq_len, self.feature_dim)

        # Temporal modeling with LSTM
        lstm_out, hidden = self.lstm(features, hidden)

        # Use the output of the last time step: (B, lstm_output_dim)
        last_output = lstm_out[:, -1, :]

        # Classification
        logits = self.classifier(last_output)

        return logits, hidden

    def predict_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for single-frame prediction (no temporal context).

        Creates a sequence of length 1 and runs the full model.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Predicted class probabilities of shape (B, num_classes).
        """
        x = x.unsqueeze(1)  # (B, 1, C, H, W)
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)


class SurgeryPhaseNetFrameLevel(nn.Module):
    """A simpler frame-level model (CNN-only) for baseline comparison.

    Uses a ResNet backbone with a single FC layer for per-frame classification.
    No temporal modeling.

    Args:
        num_classes: Number of surgical phase classes.
        backbone: ResNet variant to use.
        pretrained: Whether to use ImageNet-pretrained weights.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_dim = SurgeryPhaseNet.BACKBONE_FEATURES.get(backbone, 2048)

        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet50":
            model = models.resnet50(weights=weights)
        elif backbone == "resnet18":
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet34(weights=weights)

        modules = list(model.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.cnn(x).flatten(1)
        return self.classifier(features)
