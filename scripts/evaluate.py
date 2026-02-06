#!/usr/bin/env python3
"""
Evaluation Script for Surgery Phase Detection
===============================================

Evaluates a trained model on the MVOR or Cholec80 test set and generates
a detailed metrics report with visualizations.

Usage:
    python scripts/evaluate.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/best_model.pth \
        --output_dir results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery_phase_detection.data.transforms import get_val_transforms
from surgery_phase_detection.data.phase_labeler import MVORPhaseLabeler
from surgery_phase_detection.utils.metrics import compute_metrics, print_metrics_report
from surgery_phase_detection.utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Surgery Phase Detection Model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


def build_model(config: dict, device: torch.device) -> nn.Module:
    """Build model and load checkpoint."""
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    model_type = model_cfg.get("type", "single_view")

    if model_type == "multi_view":
        from surgery_phase_detection.models.multiview_net import MultiViewSurgeryNet
        model = MultiViewSurgeryNet(
            num_classes=dataset_cfg["num_classes"],
            backbone=model_cfg["backbone"],
            pretrained=False,
            num_views=model_cfg.get("num_views", 3),
            fusion=model_cfg.get("fusion", "attention"),
            lstm_hidden=model_cfg["lstm_hidden"],
            lstm_layers=model_cfg["lstm_layers"],
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        from surgery_phase_detection.models.resnet_lstm import SurgeryPhaseNet
        model = SurgeryPhaseNet(
            num_classes=dataset_cfg["num_classes"],
            backbone=model_cfg["backbone"],
            pretrained=False,
            lstm_hidden=model_cfg["lstm_hidden"],
            lstm_layers=model_cfg["lstm_layers"],
            dropout=model_cfg.get("dropout", 0.3),
            bidirectional=model_cfg.get("bidirectional", False),
        )

    return model


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for sequences, labels in tqdm(loader, desc="Evaluating"):
        sequences = sequences.to(device, non_blocking=True)
        logits, _ = model(sequences)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return {
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
        "metrics": compute_metrics(
            np.array(all_labels), np.array(all_preds), num_classes=num_classes
        ),
    }


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    # Load model
    print("Loading model...")
    model = build_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Build test dataloader
    preprocess_cfg = config["preprocessing"]
    val_transform = get_val_transforms(
        image_size=preprocess_cfg["image_size"],
        mean=preprocess_cfg["mean"],
        std=preprocess_cfg["std"],
    )

    phase_labeler = MVORPhaseLabeler(
        **config.get("phase_labeler", {})
    )

    if dataset_cfg["name"] == "mvor":
        from surgery_phase_detection.data.mvor_dataset import MVORSequenceDataset

        day_ids = (
            dataset_cfg["test_days"] if args.split == "test"
            else dataset_cfg["val_days"]
        )

        test_dataset = MVORSequenceDataset(
            data_root=dataset_cfg["data_root"],
            annotation_path=dataset_cfg["annotation_path"],
            sequence_length=model_cfg["sequence_length"],
            camera_id=dataset_cfg.get("primary_camera", 1),
            day_ids=day_ids,
            transform=val_transform,
            phase_labeler=phase_labeler,
        )
    else:
        from surgery_phase_detection.data.dataset import Cholec80SequenceDataset

        video_ids = (
            dataset_cfg["test_videos"] if args.split == "test"
            else dataset_cfg["val_videos"]
        )

        test_dataset = Cholec80SequenceDataset(
            frames_dir=dataset_cfg["frames_dir"],
            annotations_dir=dataset_cfg["annotations_dir"],
            video_ids=video_ids,
            sequence_length=model_cfg["sequence_length"],
            transform=val_transform,
            sample_rate=dataset_cfg.get("sample_rate", 25),
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=True,
    )

    # Evaluate
    print(f"\nEvaluating on {args.split} set ({len(test_dataset)} sequences)...")
    results = evaluate_model(model, test_loader, device, dataset_cfg["num_classes"])

    # Print report
    phase_names = dataset_cfg.get("phase_names")
    print_metrics_report(results["metrics"], phase_names)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_path = os.path.join(args.output_dir, f"{args.split}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    preds_path = os.path.join(args.output_dir, f"{args.split}_predictions.npz")
    np.savez(
        preds_path,
        predictions=results["predictions"],
        labels=results["labels"],
        probabilities=results["probabilities"],
    )
    print(f"Predictions saved to {preds_path}")

    # Plots
    cm = np.array(results["metrics"]["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        phase_names=phase_names,
        save_path=os.path.join(args.output_dir, f"{args.split}_confusion_matrix.png"),
    )

    plot_per_class_metrics(
        results["metrics"],
        metric_name="jaccard",
        phase_names=phase_names,
        save_path=os.path.join(args.output_dir, f"{args.split}_per_class_jaccard.png"),
    )

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
