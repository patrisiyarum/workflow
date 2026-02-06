"""
Feature Extractor
=================

Standalone feature extraction module. Extracts CNN features from video frames
and saves them to disk. This enables faster training of the temporal model
by precomputing spatial features.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class FrameDataset(Dataset):
    """Simple dataset that loads frames from a directory."""

    def __init__(self, frame_paths: List[str], transform: Optional[transforms.Compose] = None):
        self.frame_paths = frame_paths
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.frame_paths[idx]).convert("RGB")
        return self.transform(image)


class FeatureExtractor:
    """Extract CNN features from video frames and cache them.

    Uses a pretrained ResNet backbone to extract feature vectors from frames.
    Features can be saved to disk for faster subsequent training.

    Args:
        backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50').
        device: Torch device to use ('cuda' or 'cpu').
        batch_size: Batch size for feature extraction.
    """

    FEATURE_DIMS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.backbone_name = backbone
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.feature_dim = self.FEATURE_DIMS[backbone]

        self.model = self._build_model(backbone)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, backbone: str) -> nn.Module:
        """Build the feature extraction model (ResNet without FC layer)."""
        weights = "IMAGENET1K_V1"

        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            model = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the classification head
        modules = list(model.children())[:-1]
        return nn.Sequential(*modules)

    @torch.no_grad()
    def extract_from_frames(
        self,
        frame_paths: List[str],
        transform: Optional[transforms.Compose] = None,
    ) -> np.ndarray:
        """Extract features from a list of frame image paths.

        Args:
            frame_paths: List of paths to frame images.
            transform: Optional image transform pipeline.

        Returns:
            Feature array of shape (num_frames, feature_dim).
        """
        dataset = FrameDataset(frame_paths, transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        all_features = []
        for batch in tqdm(loader, desc="Extracting features"):
            batch = batch.to(self.device)
            features = self.model(batch).flatten(1)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def extract_and_save(
        self,
        frames_dir: str,
        output_dir: str,
        video_ids: List[int],
        sample_rate: int = 25,
    ) -> None:
        """Extract features for multiple videos and save to disk.

        For each video, saves a .npy file containing the feature matrix.

        Args:
            frames_dir: Root directory containing per-video frame folders.
            output_dir: Directory to save extracted feature files.
            video_ids: List of video IDs to process.
            sample_rate: Frame sampling rate.
        """
        os.makedirs(output_dir, exist_ok=True)

        for vid_id in sorted(video_ids):
            video_name = f"video{vid_id:02d}"
            video_frames_dir = Path(frames_dir) / video_name

            if not video_frames_dir.exists():
                print(f"Skipping {video_name}: frames directory not found")
                continue

            # Collect frame paths in sorted order
            frame_paths = sorted(
                [
                    str(p)
                    for p in video_frames_dir.glob("frame_*.jpg")
                    if int(p.stem.split("_")[1]) % sample_rate == 0
                ]
            )

            if not frame_paths:
                print(f"Skipping {video_name}: no frames found")
                continue

            print(f"Extracting features for {video_name} ({len(frame_paths)} frames)...")
            features = self.extract_from_frames(frame_paths)

            output_path = Path(output_dir) / f"{video_name}_features.npy"
            np.save(str(output_path), features)
            print(f"  Saved: {output_path} — shape {features.shape}")

        print("Feature extraction complete.")
