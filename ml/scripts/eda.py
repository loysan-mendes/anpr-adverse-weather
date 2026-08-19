"""
Phase 1 - Step 2: Exploratory Data Analysis on the downloaded dataset.

This script doesn't assume a fixed folder/annotation format (Kaggle ANPR
datasets vary a lot: some ship XML/Pascal-VOC boxes, some CSV with plate
text, some just raw images). It first *profiles* what you actually have,
then produces stats + plots so we know what Phase 2 (weather augmentation)
and Phase 5 (YOLO training) need to work with.

Usage:
    python ml/scripts/eda.py --data_dir ml/data/raw
"""

import argparse
from pathlib import Path
from collections import Counter
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
ANN_EXTS = {".xml", ".json", ".txt", ".csv"}


def scan_dir(data_dir: Path):
    images, annotations, other = [], [], []
    for p in data_dir.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            images.append(p)
        elif ext in ANN_EXTS:
            annotations.append(p)
        else:
            other.append(p)
    return images, annotations, other


def image_stats(images):
    widths, heights, sizes_kb, aspect_ratios = [], [], [], []
    corrupt = []
    sample_n = min(len(images), 500)  # sample for speed on large datasets
    sample = images if len(images) <= sample_n else list(np.random.choice(images, sample_n, replace=False))

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            corrupt.append(str(p))
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(round(w / h, 2))
        sizes_kb.append(p.stat().st_size / 1024)

    return {
        "sampled": len(sample),
        "corrupt": corrupt,
        "widths": widths,
        "heights": heights,
        "sizes_kb": sizes_kb,
        "aspect_ratios": aspect_ratios,
    }


def annotation_preview(annotations, n=3):
    preview = {}
    ext_counts = Counter(p.suffix.lower() for p in annotations)
    for ext, count in ext_counts.items():
        samples = [p for p in annotations if p.suffix.lower() == ext][:n]
        preview[ext] = {"count": count, "sample_files": [str(s) for s in samples]}
        for s in samples:
            try:
                text = s.read_text(errors="ignore")[:500]
                preview[ext].setdefault("content_preview", []).append(text)
            except Exception as e:
                preview[ext].setdefault("content_preview", []).append(f"<unreadable: {e}>")
    return preview


def plot_distributions(stats, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(stats["widths"], bins=30, color="#4C72B0")
    axes[0, 0].set_title("Image Width Distribution")
    axes[0, 0].set_xlabel("width (px)")

    axes[0, 1].hist(stats["heights"], bins=30, color="#55A868")
    axes[0, 1].set_title("Image Height Distribution")
    axes[0, 1].set_xlabel("height (px)")

    axes[1, 0].hist(stats["aspect_ratios"], bins=30, color="#C44E52")
    axes[1, 0].set_title("Aspect Ratio Distribution (w/h)")
    axes[1, 0].set_xlabel("aspect ratio")

    axes[1, 1].hist(stats["sizes_kb"], bins=30, color="#8172B2")
    axes[1, 1].set_title("File Size Distribution (KB)")
    axes[1, 1].set_xlabel("KB")

    plt.tight_layout()
    fig_path = out_dir / "image_distributions.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot -> {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ml/data/raw")
    parser.add_argument("--out_dir", default="ml/data/processed/eda_report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists():
        raise SystemExit(f"{data_dir} does not exist. Run download_dataset.py first.")

    print(f"Scanning {data_dir} ...")
    images, annotations, other = scan_dir(data_dir)
    print(f"Found: {len(images)} images | {len(annotations)} annotation files | {len(other)} other files")

    report = {
        "total_images": len(images),
        "total_annotations": len(annotations),
        "other_files_sample": [str(p) for p in other[:10]],
    }

    if images:
        stats = image_stats(images)
        report["image_stats"] = {
            "sampled": stats["sampled"],
            "num_corrupt": len(stats["corrupt"]),
            "corrupt_files": stats["corrupt"][:10],
            "width_min_max_mean": [int(np.min(stats["widths"])), int(np.max(stats["widths"])), float(np.mean(stats["widths"]))] if stats["widths"] else None,
            "height_min_max_mean": [int(np.min(stats["heights"])), int(np.max(stats["heights"])), float(np.mean(stats["heights"]))] if stats["heights"] else None,
            "most_common_aspect_ratios": Counter(stats["aspect_ratios"]).most_common(5),
        }
        plot_distributions(stats, out_dir)
    else:
        print("WARNING: no images found — check --data_dir path.")

    if annotations:
        report["annotation_preview"] = annotation_preview(annotations)
    else:
        print("NOTE: no annotation files found. Dataset may be images-only "
              "(common for classification-style Kaggle sets) — we'll need "
              "to decide on an annotation strategy for YOLO training in Phase 5.")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved -> {report_path}")
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in report.items() if k != "annotation_preview"}, indent=2, default=str))


if __name__ == "__main__":
    main()