"""
Cholec80 Dataset Loaders
========================

Provides PyTorch Dataset classes for loading the Cholec80 surgical video dataset.
Supports both single-frame and sequence-based (temporal) loading for LSTM models.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class Cholec80Dataset(Dataset):
    """Single-frame dataset for the Cholec80 surgical video dataset.

    Loads individual frames and their phase annotations. Suitable for training
    CNN-only models or for feature extraction.

    Args:
        frames_dir: Root directory containing per-video frame folders.
        annotations_dir: Directory containing phase annotation text files.
        video_ids: List of video IDs to include (1-indexed).
        transform: Optional image transform pipeline.
        sample_rate: Load every Nth frame (default 1 = all frames).
    """

    def __init__(
        self,
        frames_dir: str,
        annotations_dir: str,
        video_ids: List[int],
        transform: Optional[Callable] = None,
        sample_rate: int = 1,
    ):
        self.frames_dir = Path(frames_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        self.sample_rate = sample_rate

        self.samples: List[Tuple[str, int]] = []  # (frame_path, phase_label)
        self.video_metadata: Dict[int, dict] = {}

        self._load_annotations(video_ids)

    def _load_annotations(self, video_ids: List[int]) -> None:
        """Parse annotation files and build the list of (frame_path, label) tuples."""
        for vid_id in sorted(video_ids):
            video_name = f"video{vid_id:02d}"
            annotation_file = self.annotations_dir / f"{video_name}-phase.txt"

            if not annotation_file.exists():
                print(f"Warning: Annotation file not found: {annotation_file}")
                continue

            video_frames_dir = self.frames_dir / video_name
            if not video_frames_dir.exists():
                print(f"Warning: Frames directory not found: {video_frames_dir}")
                continue

            # Read annotation file (tab-separated: Frame\tPhase)
            df = pd.read_csv(
                annotation_file,
                sep="\t",
                header=0,
                names=["frame", "phase"],
                dtype={"frame": int, "phase": int},
            )

            frame_count = 0
            for _, row in df.iterrows():
                frame_idx = int(row["frame"])
                phase = int(row["phase"])

                # Apply sampling rate
                if frame_idx % self.sample_rate != 0:
                    continue

                frame_path = video_frames_dir / f"frame_{frame_idx:06d}.jpg"
                if frame_path.exists():
                    self.samples.append((str(frame_path), phase))
                    frame_count += 1

            self.video_metadata[vid_id] = {
                "name": video_name,
                "num_frames": frame_count,
                "annotation_file": str(annotation_file),
            }

        print(
            f"Loaded {len(self.samples)} frames from "
            f"{len(self.video_metadata)} videos"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frame_path, label = self.samples[idx]

        image = Image.open(frame_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[int, int]:
        """Return the count of samples per class."""
        distribution: Dict[int, int] = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return dict(sorted(distribution.items()))

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for handling class imbalance."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        num_classes = max(dist.keys()) + 1
        weights = torch.zeros(num_classes)
        for cls, count in dist.items():
            weights[cls] = total / (num_classes * count)
        return weights


class Cholec80SequenceDataset(Dataset):
    """Sequence-based dataset for temporal models (LSTM).

    Returns sequences of consecutive frames for temporal phase recognition.
    Each sample is a sequence of `sequence_length` frames with the label of the
    last frame in the sequence.

    Args:
        frames_dir: Root directory containing per-video frame folders.
        annotations_dir: Directory containing phase annotation text files.
        video_ids: List of video IDs to include (1-indexed).
        sequence_length: Number of consecutive frames per sequence.
        transform: Optional image transform pipeline.
        sample_rate: Load every Nth frame (default 25 for 1 fps from 25 fps video).
        stride: Step between consecutive sequences (default 1).
    """

    def __init__(
        self,
        frames_dir: str,
        annotations_dir: str,
        video_ids: List[int],
        sequence_length: int = 10,
        transform: Optional[Callable] = None,
        sample_rate: int = 25,
        stride: int = 1,
    ):
        self.frames_dir = Path(frames_dir)
        self.annotations_dir = Path(annotations_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.sample_rate = sample_rate
        self.stride = stride

        # Per-video data: list of (frame_path, label) sorted by frame index
        self.video_data: Dict[int, List[Tuple[str, int]]] = {}
        # Flat index mapping: (video_id, start_index_in_video_data)
        self.sequences: List[Tuple[int, int]] = []

        self._load_data(video_ids)

    def _load_data(self, video_ids: List[int]) -> None:
        """Load frame paths and annotations, then build sequence indices."""
        for vid_id in sorted(video_ids):
            video_name = f"video{vid_id:02d}"
            annotation_file = self.annotations_dir / f"{video_name}-phase.txt"
            video_frames_dir = self.frames_dir / video_name

            if not annotation_file.exists() or not video_frames_dir.exists():
                continue

            df = pd.read_csv(
                annotation_file,
                sep="\t",
                header=0,
                names=["frame", "phase"],
                dtype={"frame": int, "phase": int},
            )

            frames: List[Tuple[str, int]] = []
            for _, row in df.iterrows():
                frame_idx = int(row["frame"])
                phase = int(row["phase"])

                if frame_idx % self.sample_rate != 0:
                    continue

                frame_path = video_frames_dir / f"frame_{frame_idx:06d}.jpg"
                if frame_path.exists():
                    frames.append((str(frame_path), phase))

            if len(frames) < self.sequence_length:
                continue

            self.video_data[vid_id] = frames

            # Build sequence start indices for this video
            num_sequences = (len(frames) - self.sequence_length) // self.stride + 1
            for i in range(num_sequences):
                start = i * self.stride
                self.sequences.append((vid_id, start))

        total_frames = sum(len(v) for v in self.video_data.values())
        print(
            f"Loaded {total_frames} frames from {len(self.video_data)} videos, "
            f"yielding {len(self.sequences)} sequences of length {self.sequence_length}"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        vid_id, start = self.sequences[idx]
        frames_data = self.video_data[vid_id]

        images = []
        for i in range(self.sequence_length):
            frame_path, _ = frames_data[start + i]
            image = Image.open(frame_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # Stack into (sequence_length, C, H, W)
        sequence = torch.stack(images, dim=0)

        # Label is the phase of the last frame in the sequence
        _, label = frames_data[start + self.sequence_length - 1]

        return sequence, label

    def get_class_distribution(self) -> Dict[int, int]:
        """Return the count of sequences per class (based on last-frame label)."""
        distribution: Dict[int, int] = {}
        for vid_id, start in self.sequences:
            _, label = self.video_data[vid_id][start + self.sequence_length - 1]
            distribution[label] = distribution.get(label, 0) + 1
        return dict(sorted(distribution.items()))

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        num_classes = max(dist.keys()) + 1
        weights = torch.zeros(num_classes)
        for cls, count in dist.items():
            weights[cls] = total / (num_classes * count)
        return weights
