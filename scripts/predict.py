#!/usr/bin/env python3
"""
Inference Script for Surgery Phase Detection
==============================================

Run phase prediction on a new surgery video or directory of frames.

Usage:
    # From video file
    python scripts/predict.py \
        --video_path path/to/video.mp4 \
        --checkpoint checkpoints/best_model.pth \
        --output_path results/prediction.json

    # From pre-extracted frames directory
    python scripts/predict.py \
        --frames_dir path/to/frames/ \
        --checkpoint checkpoints/best_model.pth \
        --output_path results/prediction.json
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery_phase_detection.data.transforms import get_val_transforms
from surgery_phase_detection.models.resnet_lstm import SurgeryPhaseNet
from surgery_phase_detection.utils.visualization import plot_phase_timeline


PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict surgical phases")
    parser.add_argument("--video_path", type=str, help="Path to input video file.")
    parser.add_argument("--frames_dir", type=str, help="Path to pre-extracted frames directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file.")
    parser.add_argument("--output_path", type=str, default="results/prediction.json")
    parser.add_argument("--sample_rate", type=int, default=25, help="Sample every Nth frame from video.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true", help="Generate timeline visualization.")
    return parser.parse_args()


def extract_frames_from_video(
    video_path: str,
    sample_rate: int = 25,
) -> Tuple[List[np.ndarray], float]:
    """Extract frames from a video file at the given sample rate.

    Args:
        video_path: Path to the video file.
        sample_rate: Extract every Nth frame.

    Returns:
        Tuple of (list of BGR frames, video fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"  Sampling every {sample_rate} frames ({fps / sample_rate:.1f} fps effective)")

    frames = []
    frame_idx = 0

    with tqdm(total=total_frames // sample_rate, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frames.append(frame)
                pbar.update(1)

            frame_idx += 1

    cap.release()
    print(f"  Extracted {len(frames)} frames")

    return frames, fps


def load_frames_from_directory(frames_dir: str) -> List[str]:
    """Load sorted frame paths from a directory.

    Args:
        frames_dir: Directory containing frame images.

    Returns:
        Sorted list of frame file paths.
    """
    frames_dir = Path(frames_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    frame_paths = sorted(
        [str(p) for p in frames_dir.iterdir() if p.suffix.lower() in extensions]
    )

    print(f"Found {len(frame_paths)} frames in {frames_dir}")
    return frame_paths


@torch.no_grad()
def predict_phases(
    model: SurgeryPhaseNet,
    frames,
    transform,
    device: torch.device,
    sequence_length: int = 10,
) -> Dict:
    """Run phase prediction over a sequence of frames.

    Uses a sliding window approach: for each position, takes the previous
    `sequence_length` frames and predicts the phase of the current frame.

    Args:
        model: Trained SurgeryPhaseNet model.
        frames: List of PIL images or numpy arrays (BGR).
        transform: Image transform pipeline.
        device: Torch device.
        sequence_length: Temporal window size.

    Returns:
        Dictionary with predictions, probabilities, and per-frame data.
    """
    model.eval()

    n_frames = len(frames)
    predictions = []
    probabilities = []

    # Convert all frames to PIL images if they're numpy
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            # BGR to RGB
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(f))
        elif isinstance(f, str):
            pil_frames.append(Image.open(f).convert("RGB"))
        else:
            pil_frames.append(f)

    # Transform all frames
    transformed = [transform(img) for img in pil_frames]

    print(f"Running inference on {n_frames} frames...")

    for i in tqdm(range(n_frames), desc="Predicting"):
        # Build sequence ending at frame i
        start = max(0, i - sequence_length + 1)
        seq_frames = transformed[start : i + 1]

        # Pad with first frame if not enough history
        while len(seq_frames) < sequence_length:
            seq_frames.insert(0, seq_frames[0])

        # Stack into (1, T, C, H, W)
        sequence = torch.stack(seq_frames).unsqueeze(0).to(device)

        logits, _ = model(sequence)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()

        predictions.append(pred)
        probabilities.append(probs.cpu().numpy()[0])

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    # Apply temporal smoothing (majority vote over a small window)
    smoothed = _temporal_smooth(predictions, window_size=5)

    return {
        "predictions": predictions.tolist(),
        "smoothed_predictions": smoothed.tolist(),
        "probabilities": probabilities.tolist(),
        "phase_names": PHASE_NAMES,
        "num_frames": n_frames,
    }


def _temporal_smooth(predictions: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply temporal smoothing via majority voting in a sliding window.

    Args:
        predictions: Array of frame-level predictions.
        window_size: Size of the smoothing window.

    Returns:
        Smoothed predictions array.
    """
    smoothed = predictions.copy()
    half = window_size // 2

    for i in range(len(predictions)):
        start = max(0, i - half)
        end = min(len(predictions), i + half + 1)
        window = predictions[start:end]

        # Majority vote
        values, counts = np.unique(window, return_counts=True)
        smoothed[i] = values[np.argmax(counts)]

    return smoothed


def generate_summary(result: Dict) -> Dict:
    """Generate a human-readable summary of the prediction.

    Args:
        result: Prediction result dictionary.

    Returns:
        Summary dictionary with phase durations and transitions.
    """
    preds = np.array(result["smoothed_predictions"])
    phase_names = result["phase_names"]
    n_frames = len(preds)

    # Phase durations (as fraction of total)
    durations = {}
    for i, name in enumerate(phase_names):
        count = np.sum(preds == i)
        durations[name] = {
            "frame_count": int(count),
            "percentage": round(count / n_frames * 100, 2),
        }

    # Phase transitions
    transitions = []
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            transitions.append({
                "frame": i,
                "from_phase": phase_names[preds[i - 1]],
                "to_phase": phase_names[preds[i]],
            })

    return {
        "total_frames": n_frames,
        "num_transitions": len(transitions),
        "phase_durations": durations,
        "transitions": transitions,
    }


def main():
    args = parse_args()

    if not args.video_path and not args.frames_dir:
        print("Error: Provide either --video_path or --frames_dir")
        sys.exit(1)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    model = SurgeryPhaseNet(
        num_classes=dataset_cfg["num_classes"],
        backbone=model_cfg["backbone"],
        pretrained=False,
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", False),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print("Model loaded successfully.")

    # Get frames
    if args.video_path:
        frames, fps = extract_frames_from_video(args.video_path, args.sample_rate)
    else:
        frame_paths = load_frames_from_directory(args.frames_dir)
        frames = frame_paths  # Will be loaded as paths in predict_phases
        fps = 25.0 / args.sample_rate  # Assumed

    # Transform
    preprocess_cfg = config["preprocessing"]
    transform = get_val_transforms(
        image_size=preprocess_cfg["image_size"],
        mean=preprocess_cfg["mean"],
        std=preprocess_cfg["std"],
    )

    # Predict
    result = predict_phases(
        model=model,
        frames=frames,
        transform=transform,
        device=device,
        sequence_length=model_cfg["sequence_length"],
    )

    # Generate summary
    summary = generate_summary(result)
    result["summary"] = summary

    # Save results
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nPredictions saved to {args.output_path}")

    # Print summary
    print(f"\n--- Prediction Summary ---")
    print(f"Total frames analyzed: {summary['total_frames']}")
    print(f"Phase transitions: {summary['num_transitions']}")
    print(f"\nPhase durations:")
    for phase, info in summary["phase_durations"].items():
        if info["frame_count"] > 0:
            print(f"  {phase}: {info['percentage']:.1f}% ({info['frame_count']} frames)")

    # Visualization
    if args.visualize:
        preds = np.array(result["smoothed_predictions"])
        fig = plot_phase_timeline(
            true_phases=preds,
            phase_names=PHASE_NAMES,
            title="Predicted Surgery Phases",
            save_path=args.output_path.replace(".json", "_timeline.png"),
            fps=fps / args.sample_rate if args.video_path else 1.0,
        )
        print(f"Timeline visualization saved.")


if __name__ == "__main__":
    main()
