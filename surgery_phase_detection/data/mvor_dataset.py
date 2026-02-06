"""
MVOR Dataset Loader
====================

PyTorch Dataset classes for the MVOR (Multi-View Operating Room) dataset.
Parses the COCO-style JSON annotations and provides frame-level and
sequence-level loading with automatic surgical phase labeling.

The MVOR dataset contains 732 synchronized multi-view frames from 3 RGB-D
cameras across 4 days, annotated with person bounding boxes, 2D/3D keypoints,
and person roles (clinician / patient).

Since MVOR does not include explicit surgical phase labels, this module
derives phase labels from scene context:
    - Number and roles of people present
    - Temporal position within the day
    - Person pose features (keypoint positions)
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .phase_labeler import MVORPhaseLabeler


class MVORDataset(Dataset):
    """Single-frame MVOR dataset for surgery phase detection.

    Each sample is a single camera view from the operating room, labeled with
    the derived surgical phase.

    Args:
        data_root: Root directory of the MVOR dataset.
        annotation_path: Path to camma_mvor_2018.json.
        camera_ids: Which cameras to include (1, 2, 3). Default: all.
        day_ids: Which recording days to include (1-4). Default: all.
        transform: Image transform pipeline.
        phase_labeler: Optional custom phase labeler. Uses default if None.
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        camera_ids: Optional[List[int]] = None,
        day_ids: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        phase_labeler: Optional[MVORPhaseLabeler] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.camera_ids = camera_ids or [1, 2, 3]
        self.day_ids = day_ids or [1, 2, 3, 4]

        # Load annotations
        with open(annotation_path) as f:
            self.coco_data = json.load(f)

        # Phase labeler
        self.phase_labeler = phase_labeler or MVORPhaseLabeler()

        # Build indices
        self._build_index()

    def _build_index(self) -> None:
        """Build fast lookup indices from COCO annotations."""
        # Image lookup
        self.images_by_id: Dict[int, dict] = {
            img["id"]: img for img in self.coco_data["images"]
        }

        # Annotations per image
        self.anns_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        # Multiview frame groupings
        self.multiview_frames = self.coco_data["multiview_images"]

        # Build per-multiview-frame annotation summaries for phase labeling
        self.mv_summaries: Dict[str, dict] = {}
        for mv_frame in self.multiview_frames:
            mv_id = mv_frame["id"]
            all_anns = []
            for img_info in mv_frame["images"]:
                all_anns.extend(self.anns_by_image[img_info["id"]])

            timestamp = mv_frame["images"][0].get("date_captured", "")
            day_id = mv_frame["images"][0].get("day_id", 0)

            self.mv_summaries[mv_id] = {
                "timestamp": timestamp,
                "day_id": day_id,
                "annotations": all_anns,
                "image_ids": [img["id"] for img in mv_frame["images"]],
            }

        # Assign phase labels to each multiview frame
        self.mv_phases = self.phase_labeler.label_all_frames(
            self.multiview_frames, self.anns_by_image
        )

        # Build the sample list: (image_id, file_name, phase_label)
        self.samples: List[Tuple[int, str, int]] = []
        for mv_frame in self.multiview_frames:
            mv_id = mv_frame["id"]
            phase = self.mv_phases.get(mv_id, 0)

            for img_info in mv_frame["images"]:
                cam_id = img_info.get("cam_id", 0)
                day_id = img_info.get("day_id", 0)

                if cam_id in self.camera_ids and day_id in self.day_ids:
                    self.samples.append((
                        img_info["id"],
                        img_info["file_name"],
                        phase,
                    ))

        print(
            f"MVORDataset: {len(self.samples)} samples "
            f"(cameras={self.camera_ids}, days={self.day_ids})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_id, file_name, phase = self.samples[idx]

        # Load image
        img_path = self.data_root / "camma_mvor_dataset" / file_name
        image = Image.open(str(img_path)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, phase

    def get_annotations(self, idx: int) -> List[dict]:
        """Get the raw annotations for a sample."""
        img_id, _, _ = self.samples[idx]
        return self.anns_by_image[img_id]

    def get_class_distribution(self) -> Dict[int, int]:
        """Count samples per phase class."""
        dist: Dict[int, int] = {}
        for _, _, phase in self.samples:
            dist[phase] = dist.get(phase, 0) + 1
        return dict(sorted(dist.items()))

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        num_classes = max(dist.keys()) + 1
        weights = torch.ones(num_classes)
        for cls, count in dist.items():
            weights[cls] = total / (num_classes * count)
        return weights


class MVORSequenceDataset(Dataset):
    """Sequence-based MVOR dataset for temporal phase recognition.

    Returns sequences of consecutive multi-view frames ordered by timestamp,
    suitable for LSTM-based temporal modeling. Each sequence contains frames
    from a single camera within a single day.

    Args:
        data_root: Root directory of the MVOR dataset.
        annotation_path: Path to camma_mvor_2018.json.
        sequence_length: Number of frames per sequence.
        camera_id: Camera to use (1, 2, or 3). Default: 1.
        day_ids: Which recording days to include. Default: all.
        stride: Step between consecutive sequences.
        transform: Image transform pipeline.
        phase_labeler: Optional custom phase labeler.
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        sequence_length: int = 10,
        camera_id: int = 1,
        day_ids: Optional[List[int]] = None,
        stride: int = 1,
        transform: Optional[Callable] = None,
        phase_labeler: Optional[MVORPhaseLabeler] = None,
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.camera_id = camera_id
        self.day_ids = day_ids or [1, 2, 3, 4]
        self.stride = stride
        self.transform = transform

        # Load annotations
        with open(annotation_path) as f:
            self.coco_data = json.load(f)

        self.phase_labeler = phase_labeler or MVORPhaseLabeler()

        # Build index
        self.anns_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        # Get phase labels
        self.mv_phases = self.phase_labeler.label_all_frames(
            self.coco_data["multiview_images"], self.anns_by_image
        )

        # Organize frames per day, sorted by timestamp
        self._build_sequences()

    def _build_sequences(self) -> None:
        """Group frames by day and build temporal sequences."""
        # Collect frames for the target camera, organized by day
        day_frames: Dict[int, List[Tuple[str, str, int]]] = defaultdict(list)

        for mv_frame in self.coco_data["multiview_images"]:
            mv_id = mv_frame["id"]
            phase = self.mv_phases.get(mv_id, 0)

            for img_info in mv_frame["images"]:
                if img_info.get("cam_id") == self.camera_id:
                    day_id = img_info.get("day_id", 0)
                    if day_id in self.day_ids:
                        timestamp = img_info.get("date_captured", "")
                        day_frames[day_id].append((
                            img_info["file_name"],
                            timestamp,
                            phase,
                        ))

        # Sort each day's frames by timestamp
        self.per_day_frames: Dict[int, List[Tuple[str, str, int]]] = {}
        for day_id in sorted(day_frames.keys()):
            frames = sorted(day_frames[day_id], key=lambda x: x[1])
            self.per_day_frames[day_id] = frames

        # Build sequence indices: (day_id, start_index)
        self.sequences: List[Tuple[int, int]] = []
        for day_id, frames in self.per_day_frames.items():
            if len(frames) < self.sequence_length:
                continue
            n_seqs = (len(frames) - self.sequence_length) // self.stride + 1
            for i in range(n_seqs):
                self.sequences.append((day_id, i * self.stride))

        total_frames = sum(len(f) for f in self.per_day_frames.values())
        print(
            f"MVORSequenceDataset: {total_frames} frames across "
            f"{len(self.per_day_frames)} days, {len(self.sequences)} sequences "
            f"(length={self.sequence_length}, cam={self.camera_id})"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        day_id, start = self.sequences[idx]
        frames = self.per_day_frames[day_id]

        images = []
        for i in range(self.sequence_length):
            file_name, _, _ = frames[start + i]
            img_path = self.data_root / "camma_mvor_dataset" / file_name
            image = Image.open(str(img_path)).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # Stack into (sequence_length, C, H, W)
        sequence = torch.stack(images, dim=0)

        # Label is the phase of the last frame
        _, _, phase = frames[start + self.sequence_length - 1]

        return sequence, phase

    def get_class_distribution(self) -> Dict[int, int]:
        """Count sequences per phase class (based on last-frame label)."""
        dist: Dict[int, int] = {}
        for day_id, start in self.sequences:
            _, _, phase = self.per_day_frames[day_id][start + self.sequence_length - 1]
            dist[phase] = dist.get(phase, 0) + 1
        return dict(sorted(dist.items()))

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        num_classes = max(dist.keys()) + 1
        weights = torch.ones(num_classes)
        for cls, count in dist.items():
            weights[cls] = total / (num_classes * count)
        return weights


class MVORMultiViewDataset(Dataset):
    """Multi-view dataset that returns all 3 camera views per time step.

    Each sample consists of 3 images (one per camera) from the same
    synchronized multi-view frame, plus the derived phase label.

    Args:
        data_root: Root of the MVOR dataset.
        annotation_path: Path to camma_mvor_2018.json.
        day_ids: Which days to include.
        transform: Image transform pipeline.
        phase_labeler: Optional custom phase labeler.
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        day_ids: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        phase_labeler: Optional[MVORPhaseLabeler] = None,
    ):
        self.data_root = Path(data_root)
        self.day_ids = day_ids or [1, 2, 3, 4]
        self.transform = transform

        with open(annotation_path) as f:
            self.coco_data = json.load(f)

        self.phase_labeler = phase_labeler or MVORPhaseLabeler()

        self.anns_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        self.mv_phases = self.phase_labeler.label_all_frames(
            self.coco_data["multiview_images"], self.anns_by_image
        )

        # Build sample list: (mv_id, [file_name_cam1, file_name_cam2, file_name_cam3], phase)
        self.samples: List[Tuple[str, List[str], int]] = []
        for mv_frame in self.coco_data["multiview_images"]:
            day_id = mv_frame["images"][0].get("day_id", 0)
            if day_id not in self.day_ids:
                continue

            mv_id = mv_frame["id"]
            phase = self.mv_phases.get(mv_id, 0)
            file_names = [img["file_name"] for img in mv_frame["images"]]
            self.samples.append((mv_id, file_names, phase))

        print(f"MVORMultiViewDataset: {len(self.samples)} multi-view frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        mv_id, file_names, phase = self.samples[idx]

        images = []
        for fn in file_names:
            img_path = self.data_root / "camma_mvor_dataset" / fn
            image = Image.open(str(img_path)).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # Stack views: (num_views, C, H, W)
        views = torch.stack(images, dim=0)

        return views, phase
