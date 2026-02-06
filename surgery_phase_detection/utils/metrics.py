"""
Evaluation Metrics
==================

Comprehensive metrics for surgical phase recognition, including accuracy,
precision, recall, F1-score, and Jaccard index (the standard Cholec80 metric).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    phase_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute precision, recall, and F1-score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method ('macro', 'micro', 'weighted').
        phase_names: Optional list of phase names for per-class report.

    Returns:
        Dictionary with precision, recall, and f1 scores.
    """
    return {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def compute_jaccard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> Tuple[np.ndarray, float]:
    """Compute the Jaccard index (IoU) per class and the mean Jaccard.

    The Jaccard index is the standard evaluation metric for the Cholec80 dataset.
    For each phase: Jaccard = TP / (TP + FP + FN)

    Args:
        y_true: Ground truth labels array.
        y_pred: Predicted labels array.
        num_classes: Number of classes.

    Returns:
        Tuple of (per_class_jaccard, mean_jaccard).
    """
    per_class_jaccard = np.zeros(num_classes)

    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)

        tp = np.sum(true_c & pred_c)
        fp = np.sum(~true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)

        denominator = tp + fp + fn
        if denominator == 0:
            per_class_jaccard[c] = 1.0  # No samples and no predictions = perfect
        else:
            per_class_jaccard[c] = tp / denominator

    mean_jaccard = np.mean(per_class_jaccard)
    return per_class_jaccard, float(mean_jaccard)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
    phase_names: Optional[List[str]] = None,
) -> Dict:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels array.
        y_pred: Predicted labels array.
        num_classes: Number of phase classes.
        phase_names: Optional list of phase names.

    Returns:
        Dictionary containing all metrics.
    """
    if phase_names is None:
        phase_names = PHASE_NAMES[:num_classes]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = compute_accuracy(y_true, y_pred)
    prf_macro = compute_precision_recall_f1(y_true, y_pred, average="macro")
    prf_weighted = compute_precision_recall_f1(y_true, y_pred, average="weighted")
    per_class_jaccard, mean_jaccard = compute_jaccard(y_true, y_pred, num_classes)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    # Per-class metrics
    per_class = {}
    for c in range(num_classes):
        c_true = (y_true == c).astype(int)
        c_pred = (y_pred == c).astype(int)

        if c_true.sum() == 0 and c_pred.sum() == 0:
            per_class[phase_names[c]] = {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "jaccard": 1.0,
                "support": 0,
            }
        else:
            prf = compute_precision_recall_f1(c_true, c_pred, average="binary")
            per_class[phase_names[c]] = {
                **prf,
                "jaccard": float(per_class_jaccard[c]),
                "support": int(c_true.sum()),
            }

    return {
        "accuracy": accuracy,
        "macro_precision": prf_macro["precision"],
        "macro_recall": prf_macro["recall"],
        "macro_f1": prf_macro["f1"],
        "weighted_precision": prf_weighted["precision"],
        "weighted_recall": prf_weighted["recall"],
        "weighted_f1": prf_weighted["f1"],
        "mean_jaccard": mean_jaccard,
        "per_class_jaccard": {
            phase_names[c]: float(per_class_jaccard[c])
            for c in range(num_classes)
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def print_metrics_report(
    metrics: Dict,
    phase_names: Optional[List[str]] = None,
) -> None:
    """Pretty-print the evaluation metrics.

    Args:
        metrics: Dictionary returned by compute_metrics().
        phase_names: Optional phase names list.
    """
    if phase_names is None:
        phase_names = PHASE_NAMES

    print("=" * 70)
    print("SURGERY PHASE DETECTION — EVALUATION REPORT")
    print("=" * 70)

    print(f"\n{'Overall Metrics':^70}")
    print("-" * 70)
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Macro Precision:     {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:        {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")
    print(f"  Mean Jaccard (IoU):  {metrics['mean_jaccard']:.4f}")

    print(f"\n{'Per-Class Metrics':^70}")
    print("-" * 70)
    header = f"  {'Phase':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Jacc':>6} {'Supp':>7}"
    print(header)
    print("  " + "-" * 62)

    for phase_name, class_metrics in metrics["per_class"].items():
        print(
            f"  {phase_name:<30} "
            f"{class_metrics['precision']:>6.3f} "
            f"{class_metrics['recall']:>6.3f} "
            f"{class_metrics['f1']:>6.3f} "
            f"{class_metrics['jaccard']:>6.3f} "
            f"{class_metrics['support']:>7d}"
        )

    print("=" * 70)
