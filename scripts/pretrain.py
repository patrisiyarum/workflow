#!/usr/bin/env python3
"""
Video-Pose Pretraining Script (PreViPS-inspired)
==================================================

Pretrains the dual-encoder model by aligning video and tokenized 2D pose
embeddings across multiple camera views. Uses CLIP-style contrastive
learning, geometric consistency regularisation, and masked pose prediction.

Usage:
    python scripts/pretrain.py --config configs/previps.yaml

Reference: "Multi-view Video-Pose Pretraining for Operating Room Surgical
Activity Recognition" (Hamoud et al., 2025, arXiv:2502.13883)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery_phase_detection.models.previps import PreViPS
from surgery_phase_detection.data.transforms import get_train_transforms, get_val_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreViPS Video-Pose Pretraining")
    parser.add_argument("--config", type=str, default="configs/previps.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


class MVORMultiViewPoseDataset(Dataset):
    """Multi-view dataset that returns video frames AND pose coordinates.

    Each sample contains synchronised multi-view images and the corresponding
    2D pose keypoints extracted from MVOR annotations.

    Args:
        data_root: Root of the MVOR dataset.
        annotation_path: Path to camma_mvor_2018.json.
        day_ids: Which days to include.
        camera_ids: Which cameras to include.
        sequence_length: Frames per temporal clip.
        transform: Image transform pipeline.
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        day_ids: list,
        camera_ids: list = None,
        sequence_length: int = 8,
        transform=None,
    ):
        from PIL import Image

        self.data_root = Path(data_root)
        self.camera_ids = camera_ids or [1, 2, 3]
        self.seq_len = sequence_length
        self.transform = transform
        self.Image = Image

        with open(annotation_path) as f:
            coco = json.load(f)

        # Build indices
        self.anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        # Group multi-view frames by day, sorted temporally
        day_frames = defaultdict(list)
        for mv in coco["multiview_images"]:
            day_id = mv["images"][0]["day_id"]
            if day_id in day_ids:
                day_frames[day_id].append(mv)

        for d in day_frames:
            day_frames[d].sort(key=lambda m: m["images"][0]["date_captured"])

        # Build temporal sequences
        self.sequences = []  # list of lists of mv_frames
        for day_id, frames in day_frames.items():
            for start in range(0, len(frames) - sequence_length + 1):
                self.sequences.append(frames[start : start + sequence_length])

        print(f"PretrainDataset: {len(self.sequences)} sequences "
              f"(days={day_ids}, seq_len={sequence_length})")

    def __len__(self):
        return len(self.sequences)

    def _extract_poses(self, anns, num_joints=17, max_persons=8):
        """Extract 2D keypoints from annotations for a single image."""
        poses = np.zeros((max_persons, num_joints, 2), dtype=np.float32)
        count = 0
        for ann in anns:
            if count >= max_persons:
                break
            kps = ann.get("keypoints", [])
            if kps and not ann.get("only_bbox", 0):
                kps_arr = np.array(kps, dtype=np.float32).reshape(-1, 3)
                poses[count, : len(kps_arr), :] = kps_arr[:num_joints, :2]
                count += 1
        return poses  # (max_persons, J, 2)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Per-view outputs
        video_clips = {}   # {cam_id: (T, C, H, W)}
        pose_seqs = {}     # {cam_id: (max_persons*T, J, 2)}
        time_ids_out = {}  # {cam_id: (max_persons*T,)}
        person_ids_out = {}

        max_persons = 8
        num_joints = 10  # CAMMA keypoints

        for cam_id in self.camera_ids:
            frames = []
            all_poses = []
            all_time = []
            all_person = []

            for t, mv_frame in enumerate(seq):
                # Find the image for this camera
                img_info = None
                for img in mv_frame["images"]:
                    if img.get("cam_id") == cam_id:
                        img_info = img
                        break
                if img_info is None:
                    continue

                # Load image
                img_path = self.data_root / "camma_mvor_dataset" / img_info["file_name"]
                if img_path.exists():
                    image = self.Image.open(str(img_path)).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    frames.append(image)
                else:
                    # Placeholder
                    frames.append(torch.zeros(3, 224, 224))

                # Extract poses
                anns = self.anns_by_image.get(img_info["id"], [])
                poses = self._extract_poses(anns, num_joints=num_joints, max_persons=max_persons)
                for p in range(max_persons):
                    all_poses.append(poses[p])
                    all_time.append(t)
                    all_person.append(p)

            if frames:
                video_clips[cam_id] = torch.stack(frames)  # (T, C, H, W)
            pose_seqs[cam_id] = torch.tensor(np.array(all_poses), dtype=torch.float32)
            time_ids_out[cam_id] = torch.tensor(all_time, dtype=torch.long)
            person_ids_out[cam_id] = torch.tensor(all_person, dtype=torch.long)

        return video_clips, pose_seqs, time_ids_out, person_ids_out


def collate_multiview(batch):
    """Custom collate for multi-view data with per-view dictionaries."""
    video_clips = defaultdict(list)
    pose_seqs = defaultdict(list)
    time_ids = defaultdict(list)
    person_ids = defaultdict(list)

    for vc, ps, ti, pi in batch:
        for cam_id in vc:
            video_clips[cam_id].append(vc[cam_id])
            pose_seqs[cam_id].append(ps[cam_id])
            time_ids[cam_id].append(ti[cam_id])
            person_ids[cam_id].append(pi[cam_id])

    out_vc = {k: torch.stack(v) for k, v in video_clips.items()}
    out_ps = {k: torch.stack(v) for k, v in pose_seqs.items()}
    out_ti = {k: torch.stack(v) for k, v in time_ids.items()}
    out_pi = {k: torch.stack(v) for k, v in person_ids.items()}

    return out_vc, out_ps, out_ti, out_pi


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    pretrain_cfg = config["pretraining"]
    log_cfg = config.get("logging", {})

    os.makedirs(log_cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    os.makedirs(log_cfg.get("log_dir", "logs"), exist_ok=True)

    # Data
    print("\n--- Loading Pretraining Data ---")
    preprocess_cfg = config.get("preprocessing", {})
    transform = get_train_transforms(
        image_size=preprocess_cfg.get("image_size", 224),
        mean=preprocess_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=preprocess_cfg.get("std", [0.229, 0.224, 0.225]),
    )

    all_days = (
        dataset_cfg.get("train_days", [2, 3])
        + dataset_cfg.get("val_days", [4])
        + dataset_cfg.get("test_days", [1])
    )

    dataset = MVORMultiViewPoseDataset(
        data_root=dataset_cfg["data_root"],
        annotation_path=dataset_cfg["annotation_path"],
        day_ids=all_days,
        camera_ids=dataset_cfg.get("camera_ids", [1, 2, 3]),
        sequence_length=pretrain_cfg.get("sequence_length", 8),
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=pretrain_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=pretrain_cfg.get("num_workers", 2),
        collate_fn=collate_multiview,
        drop_last=True,
    )

    # Model
    print("\n--- Building PreViPS Model ---")
    model = PreViPS(
        num_classes=dataset_cfg.get("num_classes", 4),
        embed_dim=model_cfg.get("embed_dim", 256),
        backbone=model_cfg.get("backbone", "resnet50"),
        pretrained=model_cfg.get("pretrained", True),
        num_joints=model_cfg.get("num_joints", 10),
        codebook_size=model_cfg.get("codebook_size", 512),
        num_pose_tokens=model_cfg.get("num_pose_tokens", 4),
        pose_layers=model_cfg.get("pose_layers", 6),
        pose_heads=model_cfg.get("pose_heads", 8),
        max_persons=model_cfg.get("max_persons", 8),
        max_frames=model_cfg.get("max_frames", 16),
        max_views=model_cfg.get("max_views", 6),
        temperature=pretrain_cfg.get("temperature", 0.07),
        lambda_geo=pretrain_cfg.get("lambda_geo", 0.5),
        lambda_mask=pretrain_cfg.get("lambda_mask", 0.5),
        mask_ratio=pretrain_cfg.get("mask_ratio", 0.15),
    ).to(device)

    param_counts = model.get_num_params()
    print(f"Parameter counts: {param_counts}")

    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr=pretrain_cfg.get("learning_rate", 0.01),
        weight_decay=pretrain_cfg.get("weight_decay", 1e-5),
        momentum=0.9,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=pretrain_cfg.get("epochs", 50)
    )

    writer = SummaryWriter(
        log_dir=os.path.join(
            log_cfg.get("log_dir", "logs"),
            log_cfg.get("experiment_name", "previps_pretrain"),
        )
    ) if log_cfg.get("tensorboard", True) else None

    # Training loop
    print("\n--- Pretraining ---")
    epochs = pretrain_cfg.get("epochs", 50)

    for epoch in range(epochs):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for video_clips, pose_seqs, time_ids, person_ids in pbar:
            # Move to device
            vc = {k: v.to(device) for k, v in video_clips.items()}
            ps = {k: v.to(device) for k, v in pose_seqs.items()}
            ti = {k: v.to(device) for k, v in time_ids.items()}
            pi = {k: v.to(device) for k, v in person_ids.items()}

            optimizer.zero_grad()
            losses = model.pretrain_forward(vc, ps, ti, pi)
            losses["total"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                epoch_losses[k] += val
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{epoch_losses['total']/n_batches:.4f}",
                "con": f"{epoch_losses['contrastive']/n_batches:.4f}",
            })

        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}: ", end="")
        for k in ["total", "contrastive", "geometric", "mask", "vq"]:
            avg = epoch_losses[k] / max(n_batches, 1)
            print(f"{k}={avg:.4f} ", end="")
            if writer:
                writer.add_scalar(f"Pretrain/{k}", avg, epoch)
        print(f"lr={lr:.6f}")

        if writer:
            writer.add_scalar("Pretrain/lr", lr, epoch)

        # Save checkpoint
        ckpt_dir = log_cfg.get("checkpoint_dir", "checkpoints")
        if (epoch + 1) % log_cfg.get("save_every", 10) == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, os.path.join(ckpt_dir, f"pretrain_epoch{epoch+1}.pth"))

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, os.path.join(ckpt_dir, "pretrain_last.pth"))

    if writer:
        writer.close()

    print(f"\nPretraining complete. Checkpoint: {ckpt_dir}/pretrain_last.pth")


if __name__ == "__main__":
    main()
