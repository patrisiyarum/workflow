"""
Visualization Utilities
========================

Tools for plotting confusion matrices, phase timelines, and training curves.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

# Color palette for surgical phases (colorblind-friendly)
PHASE_COLORS = [
    "#4C72B0",  # Preparation — blue
    "#DD8452",  # CalotTriangleDissection — orange
    "#55A868",  # ClippingCutting — green
    "#C44E52",  # GallbladderDissection — red
    "#8172B3",  # GallbladderPackaging — purple
    "#937860",  # CleaningCoagulation — brown
    "#DA8BC3",  # GallbladderRetraction — pink
]


def plot_confusion_matrix(
    cm: np.ndarray,
    phase_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array of shape (num_classes, num_classes).
        phase_names: List of phase names for axis labels.
        normalize: Whether to normalize rows to percentages.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    if phase_names is None:
        phase_names = PHASE_NAMES[: cm.shape[0]]

    cm = np.array(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_display = cm.astype(float) / row_sums * 100
        fmt = ".1f"
        title_suffix = " (Normalized %)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = ""

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=phase_names,
        yticklabels=phase_names,
        ax=ax,
        cbar_kws={"label": "Percentage (%)" if normalize else "Count"},
        linewidths=0.5,
    )

    ax.set_xlabel("Predicted Phase", fontsize=12)
    ax.set_ylabel("True Phase", fontsize=12)
    ax.set_title(f"{title}{title_suffix}", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    return fig


def plot_phase_timeline(
    true_phases: np.ndarray,
    pred_phases: Optional[np.ndarray] = None,
    phase_names: Optional[List[str]] = None,
    title: str = "Surgery Phase Timeline",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
    fps: float = 1.0,
) -> plt.Figure:
    """Plot a temporal timeline of surgical phases (ground truth and predictions).

    Args:
        true_phases: Array of ground truth phase labels over time.
        pred_phases: Optional array of predicted phase labels.
        phase_names: List of phase names.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.
        fps: Frames per second for time axis labeling.

    Returns:
        The matplotlib Figure object.
    """
    if phase_names is None:
        phase_names = PHASE_NAMES

    n_rows = 2 if pred_phases is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)

    if n_rows == 1:
        axes = [axes]

    time_axis = np.arange(len(true_phases)) / fps / 60  # Convert to minutes

    # --- Ground Truth ---
    _draw_phase_ribbon(axes[0], time_axis, true_phases, phase_names, "Ground Truth")

    # --- Predictions ---
    if pred_phases is not None:
        _draw_phase_ribbon(axes[1], time_axis, pred_phases, phase_names, "Prediction")

        # Highlight errors
        errors = true_phases != pred_phases
        if np.any(errors):
            axes[1].fill_between(
                time_axis,
                0,
                1,
                where=errors,
                color="red",
                alpha=0.15,
                transform=axes[1].get_xaxis_transform(),
                label="Error",
            )

    axes[-1].set_xlabel("Time (minutes)", fontsize=11)
    fig.suptitle(title, fontsize=14, y=1.02)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=PHASE_COLORS[i])
        for i in range(len(phase_names))
    ]
    fig.legend(
        handles,
        phase_names,
        loc="center right",
        bbox_to_anchor=(1.18, 0.5),
        fontsize=8,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Timeline saved to {save_path}")

    return fig


def _draw_phase_ribbon(
    ax: plt.Axes,
    time_axis: np.ndarray,
    phases: np.ndarray,
    phase_names: List[str],
    label: str,
) -> None:
    """Draw a colored ribbon showing phase assignments over time."""
    num_classes = len(phase_names)
    for c in range(num_classes):
        mask = phases == c
        if np.any(mask):
            ax.fill_between(
                time_axis,
                0,
                1,
                where=mask,
                color=PHASE_COLORS[c % len(PHASE_COLORS)],
                alpha=0.8,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_ylabel(label, fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(time_axis[0], time_axis[-1])


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot training and validation loss/accuracy curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        train_accuracies: Optional list of training accuracies.
        val_accuracies: Optional list of validation accuracies.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    has_acc = train_accuracies is not None and val_accuracies is not None
    n_cols = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # --- Loss ---
    axes[0].plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    if has_acc:
        axes[1].plot(epochs, train_accuracies, "b-", label="Train Acc", linewidth=2)
        axes[1].plot(epochs, val_accuracies, "r-", label="Val Acc", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    return fig


def plot_per_class_metrics(
    metrics: Dict,
    metric_name: str = "jaccard",
    phase_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot a bar chart of per-class metrics.

    Args:
        metrics: Metrics dictionary from compute_metrics().
        metric_name: Which metric to plot ('jaccard', 'f1', 'precision', 'recall').
        phase_names: List of phase names.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    if phase_names is None:
        phase_names = PHASE_NAMES

    if title is None:
        title = f"Per-Class {metric_name.capitalize()}"

    per_class = metrics.get("per_class", {})
    values = [per_class.get(name, {}).get(metric_name, 0.0) for name in phase_names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        range(len(phase_names)),
        values,
        color=PHASE_COLORS[: len(phase_names)],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_name.capitalize(), fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Add mean line
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.3f}")
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Per-class metrics plot saved to {save_path}")

    return fig
