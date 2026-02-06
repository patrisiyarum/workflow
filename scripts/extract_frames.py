#!/usr/bin/env python3
"""
Frame Extraction Script
========================

Extract frames from Cholec80 surgery videos at a specified sample rate.
Saves frames as JPEG images for dataset loading.

Usage:
    python scripts/extract_frames.py \
        --video_dir data/videos/ \
        --output_dir data/frames/ \
        --sample_rate 25
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from surgery videos")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing surgery video files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted frames.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=25,
        help="Extract every Nth frame (default: 25 for ~1 fps from 25 fps video).",
    )
    parser.add_argument(
        "--video_ext",
        type=str,
        default=".mp4",
        help="Video file extension to look for.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        help="Optional resize dimensions (width height).",
    )
    return parser.parse_args()


def extract_video_frames(
    video_path: str,
    output_dir: str,
    sample_rate: int = 25,
    quality: int = 95,
    resize: tuple = None,
) -> int:
    """Extract frames from a single video.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        sample_rate: Save every Nth frame.
        quality: JPEG compression quality.
        resize: Optional (width, height) to resize frames.

    Returns:
        Number of frames extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {os.path.basename(video_path)}")
    print(f"    Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"    Extracting every {sample_rate}th frame (~{total_frames // sample_rate} frames)")

    os.makedirs(output_dir, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    frame_idx = 0
    extracted = 0

    pbar = tqdm(total=total_frames, desc=f"    Extracting", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame, encode_params)
            extracted += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"    Extracted {extracted} frames to {output_dir}")
    return extracted


def main():
    args = parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)

    # Find all video files
    video_files = sorted(video_dir.glob(f"*{args.video_ext}"))
    if not video_files:
        # Try common alternatives
        for ext in [".mp4", ".avi", ".mkv", ".mov"]:
            video_files = sorted(video_dir.glob(f"*{ext}"))
            if video_files:
                break

    if not video_files:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    print(f"Found {len(video_files)} video files in {video_dir}")
    print(f"Sample rate: every {args.sample_rate} frames")
    print(f"Output directory: {args.output_dir}")
    print()

    total_extracted = 0
    resize = tuple(args.resize) if args.resize else None

    for video_path in video_files:
        # Create per-video output directory
        video_name = video_path.stem
        video_output_dir = os.path.join(args.output_dir, video_name)

        # Skip if already extracted
        if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
            existing = len([f for f in os.listdir(video_output_dir) if f.endswith(".jpg")])
            print(f"  Skipping {video_name}: {existing} frames already extracted")
            total_extracted += existing
            continue

        n = extract_video_frames(
            video_path=str(video_path),
            output_dir=video_output_dir,
            sample_rate=args.sample_rate,
            quality=args.quality,
            resize=resize,
        )
        total_extracted += n
        print()

    print(f"\nDone! Total frames extracted: {total_extracted}")


if __name__ == "__main__":
    main()
