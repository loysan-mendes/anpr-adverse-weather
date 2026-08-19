"""
Phase 5 - Step 1: Convert augmented_manifest.json (VOC-style absolute-pixel
boxes) into the folder structure + label format ultralytics YOLO expects:

    ml/data/yolo_dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt   <- one line per box: "class x_center y_center w h" (all normalized 0-1)
        labels/val/*.txt
        data.yaml

Single class: "number_plate" (class id 0).

Splits BY SOURCE PHOTO (same grouping approach as Phases 3/4) so a photo's
clear/hazy/rainy/blurry variants never straddle train and val -- otherwise
val "accuracy" would partly just be memorized background content.

We train on the FULL augmented set (clear + every weather condition/severity)
rather than clear images only: even with good upstream restoration (Phase 4),
real-world residual haze/blur/rain will slip through imperfectly, so the
detector should be robust on its own, not solely dependent on restoration
being perfect.

Usage:
    python ml/scripts/prepare_yolo_dataset.py
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

ML_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_DIR / "data" / "processed" / "augmented_manifest.json"
YOLO_DIR = ML_DIR / "data" / "yolo_dataset"

CLASS_NAMES = ["number_plate"]

SEED = 42
random.seed(SEED)


def group_key(image_path: str) -> str:
    return Path(image_path).stem


def voc_to_yolo_line(box, img_w, img_h, class_id=0):
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    xmin, xmax = max(0, xmin), min(img_w, xmax)
    ymin, ymax = max(0, ymin), min(img_h, ymax)
    w, h = xmax - xmin, ymax - ymin
    if w <= 1 or h <= 1:
        return None  # degenerate box, skip
    xc = (xmin + xmax) / 2 / img_w
    yc = (ymin + ymax) / 2 / img_h
    wn = w / img_w
    hn = h / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--out_dir", default=str(YOLO_DIR))
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    with open(args.manifest) as f:
        entries = json.load(f)

    groups = defaultdict(list)
    for e in entries:
        groups[group_key(e["image_path"])].append(e)

    stems = list(groups.keys())
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * args.val_frac))
    val_stems = set(stems[:n_val])

    out_dir = Path(args.out_dir)
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    box_counts = {"train": 0, "val": 0}
    skipped_no_boxes = 0

    for stem, items in groups.items():
        split = "val" if stem in val_stems else "train"
        for e in items:
            src_img = ML_DIR / e["image_path"]
            if not src_img.exists():
                print(f"WARNING: missing image {src_img}, skipping")
                continue

            # unique filename: condition_severity_stem.jpg (avoids collisions --
            # the same stem repeats across clear/haze/rain/blur/lowlight folders)
            unique_name = f"{e['condition']}_{e['severity']}_{stem}"
            dst_img = out_dir / "images" / split / f"{unique_name}.jpg"
            shutil.copy2(src_img, dst_img)

            lines = []
            for box in e["boxes"]:
                line = voc_to_yolo_line(box, e["width"], e["height"])
                if line:
                    lines.append(line)

            if not lines:
                skipped_no_boxes += 1
            label_path = out_dir / "labels" / split / f"{unique_name}.txt"
            label_path.write_text("\n".join(lines))

            counts[split] += 1
            box_counts[split] += len(lines)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    print(f"{len(stems) - n_val} train photos, {n_val} val photos")
    print(f"Train: {counts['train']} images, {box_counts['train']} boxes")
    print(f"Val:   {counts['val']} images, {box_counts['val']} boxes")
    if skipped_no_boxes:
        print(f"NOTE: {skipped_no_boxes} images had zero valid boxes after conversion "
              f"(empty label file written -- YOLO treats these as background/negatives)")
    print(f"\nWrote dataset -> {out_dir}")
    print(f"data.yaml -> {data_yaml}")


if __name__ == "__main__":
    main()