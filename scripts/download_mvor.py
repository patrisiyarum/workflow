#!/usr/bin/env python3
"""
Download and Setup the MVOR Dataset
=====================================

Downloads the MVOR (Multi-View Operating Room) dataset from the CAMMA
public S3 server and sets up the directory structure.

The MVOR dataset contains 732 synchronized multi-view frames from 3 RGB-D
cameras recorded over 4 days in a hybrid operating room during procedures
such as vertebroplasty and lung biopsy.

Usage:
    python scripts/download_mvor.py --output_dir data/mvor
    python scripts/download_mvor.py --output_dir data/mvor --annotations_only
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


MVOR_DATASET_URL = "https://s3.unistra.fr/camma_public/datasets/mvor/camma_mvor_dataset.zip"
MVOR_REPO_URL = "https://github.com/CAMMA-public/MVOR.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MVOR dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mvor",
        help="Root directory to store the MVOR dataset.",
    )
    parser.add_argument(
        "--annotations_only",
        action="store_true",
        help="Only clone the repo for annotations (skip large image download).",
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Skip downloading the image dataset zip (use if already downloaded).",
    )
    return parser.parse_args()


def download_with_progress(url: str, dest: str) -> None:
    """Download a file with a progress indicator."""
    def reporthook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size) if total_size > 0 else 0
        mb_done = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
        sys.stdout.write(f"\r  Downloading: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()

    print(f"  URL: {url}")
    print(f"  Destination: {dest}")
    try:
        urlretrieve(url, dest, reporthook)
        print()  # newline after progress
    except URLError as e:
        print(f"\nDownload failed: {e}")
        print("You can manually download from:")
        print(f"  {url}")
        sys.exit(1)


def clone_mvor_repo(output_dir: str) -> str:
    """Clone the MVOR repository for annotations and evaluation scripts."""
    repo_dir = os.path.join(output_dir, "MVOR")
    if os.path.exists(repo_dir):
        print(f"  MVOR repo already exists at {repo_dir}")
        return repo_dir

    print(f"  Cloning MVOR repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", MVOR_REPO_URL, repo_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Git clone failed: {result.stderr}")
        sys.exit(1)

    print(f"  Cloned to {repo_dir}")
    return repo_dir


def setup_annotations(repo_dir: str, output_dir: str) -> None:
    """Copy annotation files to the expected location."""
    src_annotations = os.path.join(repo_dir, "annotations", "camma_mvor_2018.json")
    dst_annotations = os.path.join(output_dir, "annotations", "camma_mvor_2018.json")

    os.makedirs(os.path.dirname(dst_annotations), exist_ok=True)

    if not os.path.exists(dst_annotations):
        shutil.copy2(src_annotations, dst_annotations)
        print(f"  Annotations copied to {dst_annotations}")
    else:
        print(f"  Annotations already exist at {dst_annotations}")

    # Also copy evaluation scripts
    src_lib = os.path.join(repo_dir, "lib")
    dst_lib = os.path.join(output_dir, "eval_lib")
    if not os.path.exists(dst_lib):
        shutil.copytree(src_lib, dst_lib)
        print(f"  Evaluation scripts copied to {dst_lib}")


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MVOR Dataset Setup")
    print("=" * 60)

    # Step 1: Clone the MVOR repo (for annotations)
    print("\n[1/3] Cloning MVOR repository for annotations...")
    repo_dir = clone_mvor_repo(output_dir)

    # Step 2: Setup annotations
    print("\n[2/3] Setting up annotations...")
    setup_annotations(repo_dir, output_dir)

    if args.annotations_only:
        print("\n  Skipping image download (--annotations_only).")
        print("\nSetup complete (annotations only).")
        print(f"  Annotations: {output_dir}/annotations/camma_mvor_2018.json")
        return

    # Step 3: Download the dataset images
    dataset_dir = os.path.join(output_dir, "camma_mvor_dataset")
    zip_path = os.path.join(output_dir, "camma_mvor_dataset.zip")

    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        print("\n[3/3] Dataset images already downloaded.")
    elif args.skip_images:
        print("\n[3/3] Skipping image download (--skip_images).")
    else:
        print("\n[3/3] Downloading MVOR dataset images (~2 GB)...")
        download_with_progress(MVOR_DATASET_URL, zip_path)

        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        print(f"  Extracted to {output_dir}")

        # Clean up zip
        os.remove(zip_path)
        print(f"  Removed {zip_path}")

    # Verify structure
    print("\n" + "=" * 60)
    print("MVOR Dataset Structure:")
    print("=" * 60)
    expected = {
        "Annotations": os.path.join(output_dir, "annotations", "camma_mvor_2018.json"),
        "Day 1 images": os.path.join(output_dir, "camma_mvor_dataset", "day1"),
        "Day 2 images": os.path.join(output_dir, "camma_mvor_dataset", "day2"),
        "Day 3 images": os.path.join(output_dir, "camma_mvor_dataset", "day3"),
        "Day 4 images": os.path.join(output_dir, "camma_mvor_dataset", "day4"),
    }

    all_ok = True
    for name, path in expected.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status}] {name}: {path}")

    if all_ok:
        print("\nSetup complete! All files present.")
    else:
        print("\nSome files are missing. See above for details.")

    print(f"\nDataset root: {output_dir}")


if __name__ == "__main__":
    main()
