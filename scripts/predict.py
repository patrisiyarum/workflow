#!/usr/bin/env python3
"""
Inference Script for Surgery Phase Detection
==============================================

Run phase prediction on new operating room images, video files, or
directories of frames.

Usage:
    # Predict on a directory of OR images
    python scripts/predict.py \
        --frames_dir path/to/frames/ \
        --checkpoint checkpoints/best_model.pth

    # Predict on a video file
    python scripts/predict.py \
        --video_path path/to/video.mp4 \
        --checkpoint checkpoints/best_model.pth \
        --output_path results/prediction.json
"""

import argparse
import json
import os
import sys
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
from surgery_phase_detection.data.phase_labeler import MVOR_PHASE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict surgical phases")
    parser.add_argument("--video_path", type=str, help="Path to input video file.")
    parser.add_argument("--frames_dir", type=str, help="Path to pre-extracted frames.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_path", type=str, default="results/prediction.json")
    parser.add_argument("--sample_rate", type=int, default=25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def extract_frames_from_video(
    video_path: str, sample_rate: int = 25
) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}, FPS: {fps:.1f}, Frames: {total}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames, fps


def load_frames_from_directory(frames_dir: str) -> List[str]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(
        str(p) for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in extensions
    )
    print(f"Found {len(paths)} frames in {frames_dir}")
    return paths


@torch.no_grad()
def predict_phases(model, frames, transform, device, sequence_length=5):
    model.eval()
    n_frames = len(frames)
    predictions, probabilities = [], []

    # Convert to PIL
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(f))
        elif isinstance(f, str):
            pil_frames.append(Image.open(f).convert("RGB"))
        else:
            pil_frames.append(f)

    transformed = [transform(img) for img in pil_frames]

    for i in tqdm(range(n_frames), desc="Predicting"):
        start = max(0, i - sequence_length + 1)
        seq = transformed[start: i + 1]
        while len(seq) < sequence_length:
            seq.insert(0, seq[0])

        sequence = torch.stack(seq).unsqueeze(0).to(device)
        logits, _ = model(sequence)
        probs = torch.softmax(logits, dim=-1)
        predictions.append(logits.argmax(dim=-1).item())
        probabilities.append(probs.cpu().numpy()[0])

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    smoothed = _temporal_smooth(predictions)

    return {
        "predictions": predictions.tolist(),
        "smoothed_predictions": smoothed.tolist(),
        "probabilities": probabilities.tolist(),
        "phase_names": MVOR_PHASE_NAMES,
        "num_frames": n_frames,
    }


def _temporal_smooth(predictions, window_size=5):
    smoothed = predictions.copy()
    half = window_size // 2
    for i in range(len(predictions)):
        start = max(0, i - half)
        end = min(len(predictions), i + half + 1)
        window = predictions[start:end]
        values, counts = np.unique(window, return_counts=True)
        smoothed[i] = values[np.argmax(counts)]
    return smoothed


def generate_summary(result):
    preds = np.array(result["smoothed_predictions"])
    phase_names = result["phase_names"]
    n = len(preds)

    durations = {}
    for i, name in enumerate(phase_names):
        count = int(np.sum(preds == i))
        durations[name] = {
            "frame_count": count,
            "percentage": round(count / n * 100, 2) if n > 0 else 0,
        }

    transitions = []
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            transitions.append({
                "frame": i,
                "from_phase": phase_names[int(preds[i - 1])],
                "to_phase": phase_names[int(preds[i])],
            })

    return {
        "total_frames": n,
        "num_transitions": len(transitions),
        "phase_durations": durations,
        "transitions": transitions,
    }


def main():
    args = parse_args()

    if not args.video_path and not args.frames_dir:
        print("Error: Provide --video_path or --frames_dir")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
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
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print("Model loaded.")

    # Get frames
    if args.video_path:
        frames, fps = extract_frames_from_video(args.video_path, args.sample_rate)
    else:
        frames = load_frames_from_directory(args.frames_dir)
        fps = 1.0

    # Transform
    preprocess_cfg = config["preprocessing"]
    transform = get_val_transforms(
        image_size=preprocess_cfg["image_size"],
        mean=preprocess_cfg["mean"],
        std=preprocess_cfg["std"],
    )

    # Predict
    result = predict_phases(
        model, frames, transform, device,
        sequence_length=model_cfg["sequence_length"],
    )

    summary = generate_summary(result)
    result["summary"] = summary

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nPredictions saved to {args.output_path}")

    # Print summary
    print(f"\n--- Prediction Summary ---")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Phase transitions: {summary['num_transitions']}")
    print(f"\nPhase durations:")
    for phase, info in summary["phase_durations"].items():
        if info["frame_count"] > 0:
            print(f"  {phase}: {info['percentage']:.1f}% ({info['frame_count']} frames)")

    # Visualization
    if args.visualize:
        from surgery_phase_detection.utils.visualization import plot_phase_timeline
        preds = np.array(result["smoothed_predictions"])
        plot_phase_timeline(
            true_phases=preds,
            phase_names=MVOR_PHASE_NAMES,
            title="Predicted OR Activity Phases",
            save_path=args.output_path.replace(".json", "_timeline.png"),
            fps=fps / args.sample_rate if args.video_path else 1.0,
        )


if __name__ == "__main__":
    main()
