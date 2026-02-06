#!/usr/bin/env python3
"""
Training Script for Surgery Phase Detection
=============================================

Trains a model on the MVOR or Cholec80 dataset for surgical phase detection.
Supports both single-view (ResNet+LSTM) and multi-view fusion architectures.

Usage:
    # Single-view on MVOR (default)
    python scripts/train.py --config configs/default.yaml

    # Multi-view on MVOR
    python scripts/train.py --config configs/mvor_multiview.yaml

    # Resume training
    python scripts/train.py --config configs/default.yaml --resume checkpoints/last.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery_phase_detection.data.transforms import get_train_transforms, get_val_transforms
from surgery_phase_detection.data.phase_labeler import MVORPhaseLabeler
from surgery_phase_detection.utils.metrics import compute_metrics, print_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Surgery Phase Detection Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_phase_labeler(config: dict) -> MVORPhaseLabeler:
    """Create the MVOR phase labeler from config."""
    pl_cfg = config.get("phase_labeler", {})
    return MVORPhaseLabeler(
        num_clinician_threshold=pl_cfg.get("num_clinician_threshold", 2),
        temporal_smooth_window=pl_cfg.get("temporal_smooth_window", 3),
        use_temporal_context=pl_cfg.get("use_temporal_context", True),
    )


def build_dataloaders(config: dict):
    """Create training and validation data loaders."""
    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocessing"]
    augment_cfg = config.get("augmentation", {})
    train_cfg = config["training"]
    model_cfg = config["model"]

    train_transform = get_train_transforms(
        image_size=preprocess_cfg["image_size"],
        mean=preprocess_cfg["mean"],
        std=preprocess_cfg["std"],
        horizontal_flip=augment_cfg.get("horizontal_flip", 0.5),
        rotation=augment_cfg.get("rotation", 10),
        brightness=augment_cfg.get("brightness", 0.2),
        contrast=augment_cfg.get("contrast", 0.2),
        saturation=augment_cfg.get("saturation", 0.2),
        hue=augment_cfg.get("hue", 0.05),
    )

    val_transform = get_val_transforms(
        image_size=preprocess_cfg["image_size"],
        mean=preprocess_cfg["mean"],
        std=preprocess_cfg["std"],
    )

    phase_labeler = build_phase_labeler(config)

    if dataset_cfg["name"] == "mvor":
        from surgery_phase_detection.data.mvor_dataset import (
            MVORSequenceDataset,
        )

        train_dataset = MVORSequenceDataset(
            data_root=dataset_cfg["data_root"],
            annotation_path=dataset_cfg["annotation_path"],
            sequence_length=model_cfg["sequence_length"],
            camera_id=dataset_cfg.get("primary_camera", 1),
            day_ids=dataset_cfg.get("train_days", [2, 3]),
            transform=train_transform,
            phase_labeler=phase_labeler,
        )

        val_dataset = MVORSequenceDataset(
            data_root=dataset_cfg["data_root"],
            annotation_path=dataset_cfg["annotation_path"],
            sequence_length=model_cfg["sequence_length"],
            camera_id=dataset_cfg.get("primary_camera", 1),
            day_ids=dataset_cfg.get("val_days", [4]),
            transform=val_transform,
            phase_labeler=phase_labeler,
        )
    else:
        # Cholec80 fallback
        from surgery_phase_detection.data.dataset import Cholec80SequenceDataset

        train_dataset = Cholec80SequenceDataset(
            frames_dir=dataset_cfg["frames_dir"],
            annotations_dir=dataset_cfg["annotations_dir"],
            video_ids=dataset_cfg["train_videos"],
            sequence_length=model_cfg["sequence_length"],
            transform=train_transform,
            sample_rate=dataset_cfg.get("sample_rate", 25),
        )

        val_dataset = Cholec80SequenceDataset(
            frames_dir=dataset_cfg["frames_dir"],
            annotations_dir=dataset_cfg["annotations_dir"],
            video_ids=dataset_cfg["val_videos"],
            sequence_length=model_cfg["sequence_length"],
            transform=val_transform,
            sample_rate=dataset_cfg.get("sample_rate", 25),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    return train_loader, val_loader, train_dataset, val_dataset


def build_model(config: dict) -> nn.Module:
    """Build the model from configuration."""
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    model_type = model_cfg.get("type", "single_view")

    if model_type == "multi_view":
        from surgery_phase_detection.models.multiview_net import MultiViewSurgeryNet

        model = MultiViewSurgeryNet(
            num_classes=dataset_cfg["num_classes"],
            backbone=model_cfg["backbone"],
            pretrained=model_cfg.get("pretrained", True),
            num_views=model_cfg.get("num_views", 3),
            fusion=model_cfg.get("fusion", "attention"),
            lstm_hidden=model_cfg["lstm_hidden"],
            lstm_layers=model_cfg["lstm_layers"],
            dropout=model_cfg.get("dropout", 0.3),
            freeze_backbone=True,
        )
    else:
        from surgery_phase_detection.models.resnet_lstm import SurgeryPhaseNet

        model = SurgeryPhaseNet(
            num_classes=dataset_cfg["num_classes"],
            backbone=model_cfg["backbone"],
            pretrained=model_cfg.get("pretrained", True),
            lstm_hidden=model_cfg["lstm_hidden"],
            lstm_layers=model_cfg["lstm_layers"],
            dropout=model_cfg.get("dropout", 0.3),
            bidirectional=model_cfg.get("bidirectional", False),
            freeze_backbone=True,
        )

    return model


def build_optimizer_and_scheduler(model: nn.Module, config: dict):
    train_cfg = config["training"]

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    scheduler_name = train_cfg.get("scheduler", "cosine")
    scheduler_params = train_cfg.get("scheduler_params", {})

    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", train_cfg["epochs"]),
        )
    elif scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_params.get("step_size", 15),
            gamma=scheduler_params.get("gamma", 0.1),
        )
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    else:
        scheduler = None

    return optimizer, scheduler


def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler, use_amp,
    gradient_clip=1.0, log_every=20,
):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch_idx, (sequences, labels) in enumerate(tqdm(loader, desc="Training")):
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(sequences)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % log_every == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(loader)} — "
                f"Loss: {total_loss / total_samples:.4f}, "
                f"Acc: {total_correct / total_samples:.4f}"
            )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=4):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for sequences, labels in tqdm(loader, desc="Validating"):
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(sequences)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "metrics": metrics,
        "predictions": all_preds,
        "labels": all_labels,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "config": config,
    }, path)


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    log_cfg = config["logging"]
    os.makedirs(log_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(log_cfg["log_dir"], exist_ok=True)

    # Data
    print("\n--- Loading Data ---")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(config)

    # Model
    print("\n--- Building Model ---")
    model = build_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]

    if train_cfg.get("class_weights") == "auto":
        class_weights = train_dataset.get_class_weights().to(device)
        print(f"Using auto class weights: {class_weights}")
    elif isinstance(train_cfg.get("class_weights"), list):
        class_weights = torch.tensor(train_cfg["class_weights"], dtype=torch.float).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    writer = None
    if log_cfg.get("tensorboard", True):
        writer = SummaryWriter(
            log_dir=os.path.join(log_cfg["log_dir"], log_cfg.get("experiment_name", "default"))
        )

    # Resume
    start_epoch = 0
    best_metric = 0.0

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", 0.0)
        print(f"Resumed at epoch {start_epoch}, best metric: {best_metric:.4f}")

    # Training loop
    print("\n--- Training ---")
    freeze_epochs = config["model"].get("freeze_backbone_epochs", 3)
    patience = train_cfg.get("early_stopping_patience", 8)
    patience_counter = 0

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        print(f"{'='*60}")

        # Unfreeze backbone
        if epoch == freeze_epochs:
            print("Unfreezing CNN backbone...")
            if hasattr(model, "unfreeze_cnn"):
                model.unfreeze_cnn()
            optimizer, scheduler = build_optimizer_and_scheduler(model, config)

        # Train
        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp,
            gradient_clip=train_cfg.get("gradient_clip", 1.0),
            log_every=log_cfg.get("log_every", 20),
        )

        # Validate
        val_results = validate(
            model, val_loader, criterion, device,
            num_classes=dataset_cfg["num_classes"],
        )

        # Scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_results["loss"])
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['accuracy']:.4f}")
        print(f"  Val Loss:   {val_results['loss']:.4f} | Val Acc:   {val_results['accuracy']:.4f}")
        print(f"  Val F1: {val_results['metrics']['macro_f1']:.4f} | "
              f"Val Jaccard: {val_results['metrics']['mean_jaccard']:.4f}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        if writer:
            writer.add_scalar("Loss/train", train_results["loss"], epoch)
            writer.add_scalar("Loss/val", val_results["loss"], epoch)
            writer.add_scalar("Accuracy/train", train_results["accuracy"], epoch)
            writer.add_scalar("Accuracy/val", val_results["accuracy"], epoch)
            writer.add_scalar("F1/val", val_results["metrics"]["macro_f1"], epoch)
            writer.add_scalar("Jaccard/val", val_results["metrics"]["mean_jaccard"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Best model check
        current_metric = val_results["metrics"]["macro_f1"]

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric, config,
                os.path.join(log_cfg["checkpoint_dir"], "best_model.pth"),
            )
            print(f"  ** New best model! F1: {best_metric:.4f} **")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        # Periodic checkpoint
        if (epoch + 1) % log_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric, config,
                os.path.join(log_cfg["checkpoint_dir"], f"checkpoint_epoch{epoch + 1}.pth"),
            )

        save_checkpoint(
            model, optimizer, scheduler, epoch, best_metric, config,
            os.path.join(log_cfg["checkpoint_dir"], "last.pth"),
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs.")
            break

    # Final evaluation
    print("\n--- Final Evaluation ---")
    best_ckpt = os.path.join(log_cfg["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_results = validate(
        model, val_loader, criterion, device,
        num_classes=dataset_cfg["num_classes"],
    )

    print_metrics_report(
        val_results["metrics"],
        dataset_cfg.get("phase_names"),
    )

    if writer:
        writer.close()

    print(f"\nTraining complete. Best F1: {best_metric:.4f}")
    print(f"Best model saved to: {best_ckpt}")


if __name__ == "__main__":
    main()
