#!/usr/bin/env python3
"""
Training Script for Surgery Phase Detection
=============================================

Trains the ResNet + LSTM model on the Cholec80 dataset.

Usage:
    python scripts/train.py --config configs/default.yaml
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

from surgery_phase_detection.data.dataset import Cholec80SequenceDataset
from surgery_phase_detection.data.transforms import get_train_transforms, get_val_transforms
from surgery_phase_detection.models.resnet_lstm import SurgeryPhaseNet
from surgery_phase_detection.utils.metrics import compute_metrics, print_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Surgery Phase Detection Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def build_dataloaders(config: dict) -> tuple:
    """Create training and validation data loaders."""
    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocessing"]
    augment_cfg = config.get("augmentation", {})
    train_cfg = config["training"]

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

    model_cfg = config["model"]

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
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    return train_loader, val_loader, train_dataset, val_dataset


def build_model(config: dict) -> SurgeryPhaseNet:
    """Build the model from configuration."""
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    model = SurgeryPhaseNet(
        num_classes=dataset_cfg["num_classes"],
        backbone=model_cfg["backbone"],
        pretrained=model_cfg.get("pretrained", True),
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", False),
        freeze_backbone=True,  # Start with frozen CNN
    )

    return model


def build_optimizer_and_scheduler(model: nn.Module, config: dict):
    """Build optimizer and learning rate scheduler."""
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
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_params.get("gamma", 0.1),
            patience=5,
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_clip: float = 1.0,
    log_every: int = 50,
) -> dict:
    """Train the model for one epoch.

    Returns:
        Dictionary with training metrics for this epoch.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

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

        # Track metrics
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % log_every == 0:
            running_loss = total_loss / total_samples
            running_acc = total_correct / total_samples
            print(
                f"  Batch {batch_idx + 1}/{len(loader)} — "
                f"Loss: {running_loss:.4f}, Acc: {running_acc:.4f}"
            )

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 7,
) -> dict:
    """Validate the model on the validation set.

    Returns:
        Dictionary with validation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

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

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    # Compute full metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "metrics": metrics,
        "predictions": all_preds,
        "labels": all_labels,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    config: dict,
    path: str,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "config": config,
    }
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup device
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Setup directories
    log_cfg = config["logging"]
    os.makedirs(log_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(log_cfg["log_dir"], exist_ok=True)

    # Build data loaders
    print("\n--- Loading Data ---")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(config)

    # Build model
    print("\n--- Building Model ---")
    model = build_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with optional class weighting
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

    # Optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    # Mixed precision
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Tensorboard
    writer = None
    if log_cfg.get("tensorboard", True):
        writer = SummaryWriter(
            log_dir=os.path.join(log_cfg["log_dir"], log_cfg.get("experiment_name", "default"))
        )

    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0.0

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", 0.0)
        print(f"Resumed at epoch {start_epoch}, best metric: {best_metric:.4f}")

    # Training loop
    print("\n--- Training ---")
    freeze_epochs = config["model"].get("freeze_backbone_epochs", 5)
    patience = train_cfg.get("early_stopping_patience", 10)
    patience_counter = 0

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        print(f"{'='*60}")

        # Unfreeze backbone after warmup period
        if epoch == freeze_epochs:
            print("Unfreezing CNN backbone...")
            model.unfreeze_cnn()
            # Rebuild optimizer to include all parameters
            optimizer, scheduler = build_optimizer_and_scheduler(model, config)

        # Train
        train_results = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip=train_cfg.get("gradient_clip", 1.0),
            log_every=log_cfg.get("log_every", 50),
        )

        # Validate
        val_results = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=dataset_cfg["num_classes"],
        )

        # Update scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_results["loss"])
            else:
                scheduler.step()

        # Logging
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['accuracy']:.4f}")
        print(f"  Val Loss:   {val_results['loss']:.4f} | Val Acc:   {val_results['accuracy']:.4f}")
        print(f"  Val Jaccard: {val_results['metrics']['mean_jaccard']:.4f}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        if writer:
            writer.add_scalar("Loss/train", train_results["loss"], epoch)
            writer.add_scalar("Loss/val", val_results["loss"], epoch)
            writer.add_scalar("Accuracy/train", train_results["accuracy"], epoch)
            writer.add_scalar("Accuracy/val", val_results["accuracy"], epoch)
            writer.add_scalar("Jaccard/val", val_results["metrics"]["mean_jaccard"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Check for improvement (using mean Jaccard as primary metric)
        current_metric = val_results["metrics"]["mean_jaccard"]

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric, config,
                os.path.join(log_cfg["checkpoint_dir"], "best_model.pth"),
            )
            print(f"  ** New best model! Jaccard: {best_metric:.4f} **")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        # Save periodic checkpoint
        if (epoch + 1) % log_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric, config,
                os.path.join(log_cfg["checkpoint_dir"], f"checkpoint_epoch{epoch + 1}.pth"),
            )

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_metric, config,
            os.path.join(log_cfg["checkpoint_dir"], "last.pth"),
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    # Final evaluation with best model
    print("\n--- Final Evaluation ---")
    best_ckpt = os.path.join(log_cfg["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_results = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=dataset_cfg["num_classes"],
    )

    print_metrics_report(val_results["metrics"], dataset_cfg.get("phase_names"))

    if writer:
        writer.close()

    print(f"\nTraining complete. Best Jaccard: {best_metric:.4f}")
    print(f"Best model saved to: {best_ckpt}")


if __name__ == "__main__":
    main()
