#!/usr/bin/env python3
"""
Evaluation Script for Surgery Phase Detection
===============================================

Evaluates a trained model on the test set and generates a detailed metrics report.

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

from surgery_phase_detection.data.dataset import Cholec80SequenceDataset
from surgery_phase_detection.data.transforms import get_val_transforms
from surgery_phase_detection.models.resnet_lstm import SurgeryPhaseNet
from surgery_phase_detection.utils.metrics import compute_metrics, print_metrics_report
from surgery_phase_detection.utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Surgery Phase Detection Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Run model evaluation over the entire dataset.

    Returns:
        Dictionary containing predictions, labels, and computed metrics.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for sequences, labels in tqdm(loader, desc="Evaluating"):
        sequences = sequences.to(device, non_blocking=True)

        logits, _ = model(sequences)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "metrics": metrics,
    }


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    model = SurgeryPhaseNet(
        num_classes=dataset_cfg["num_classes"],
        backbone=model_cfg["backbone"],
        pretrained=False,  # Don't need pretrained weights, loading checkpoint
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", False),
    )

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

    video_ids = dataset_cfg["test_videos"] if args.split == "test" else dataset_cfg["val_videos"]

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
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Evaluate
    print(f"\nEvaluating on {args.split} set ({len(test_dataset)} sequences)...")
    results = evaluate_model(model, test_loader, device, dataset_cfg["num_classes"])

    # Print report
    print_metrics_report(results["metrics"], dataset_cfg.get("phase_names"))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, f"{args.split}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    # Save predictions
    preds_path = os.path.join(args.output_dir, f"{args.split}_predictions.npz")
    np.savez(
        preds_path,
        predictions=results["predictions"],
        labels=results["labels"],
        probabilities=results["probabilities"],
    )
    print(f"Predictions saved to {preds_path}")

    # Generate plots
    cm = np.array(results["metrics"]["confusion_matrix"])
    phase_names = dataset_cfg.get("phase_names")

    fig_cm = plot_confusion_matrix(
        cm,
        phase_names=phase_names,
        save_path=os.path.join(args.output_dir, f"{args.split}_confusion_matrix.png"),
    )

    fig_jaccard = plot_per_class_metrics(
        results["metrics"],
        metric_name="jaccard",
        phase_names=phase_names,
        save_path=os.path.join(args.output_dir, f"{args.split}_per_class_jaccard.png"),
    )

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
