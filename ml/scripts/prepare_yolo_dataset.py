"""Build isolated YOLO datasets; never merge new splits into old directories."""
import argparse
import json
import shutil
import uuid
from pathlib import Path
from dataset_utils import load_manifest, split_entries, image_path, source_id, provenance
from model_artifacts import atomic_json

ML_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_DIR / "data" / "processed" / "augmented_manifest.json"
YOLO_DIR = ML_DIR / "data" / "yolo_dataset"

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


def prepare_dataset(manifest, out_dir, val_frac=0.15, test_frac=0.15):
    splits = split_entries(load_manifest(manifest), val_frac, test_frac)
    # Complete preflight before publishing anything. Missing images must not
    # silently shrink validation or turn corrupt annotations into negatives.
    for entries in splits.values():
        for entry in entries:
            if not image_path(entry, ML_DIR).is_file():
                raise FileNotFoundError(image_path(entry, ML_DIR))
            if entry["width"] <= 0 or entry["height"] <= 0:
                raise ValueError("Image dimensions must be positive")
            for box in entry["boxes"]:
                if voc_to_yolo_line(box, entry["width"], entry["height"]) is None:
                    raise ValueError(f"Invalid box in {entry['image_path']}: {box}")
    out_dir = Path(out_dir)
    build_name = "builds/" + uuid.uuid4().hex
    build_dir = out_dir / build_name
    for split, entries in splits.items():
        images, labels = build_dir / "images" / split, build_dir / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index, entry in enumerate(entries):
            src = image_path(entry, ML_DIR)
            name = f"{index:06d}_{src.stem}"
            shutil.copy2(src, images / (name + src.suffix.lower()))
            lines = [voc_to_yolo_line(b, entry["width"], entry["height"]) for b in entry["boxes"]]
            (labels / (name + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        atomic_json(build_dir / (split + "_manifest.json"), entries)
    record = provenance(splits["train"] + splits["val"], ML_DIR)
    atomic_json(build_dir / "provenance.json", record)
    # No 'path' key: Ultralytics resolves train/val relative to this YAML.
    yaml = (f"train: {build_name}/images/train\nval: {build_name}/images/val\n"
            + (f"test: {build_name}/images/test\n" if splits["test"] else "")
            + f"provenance: {build_name}/provenance.json\nnc: 1\nnames: ['number_plate']\n")
    temporary = out_dir / "data.yaml.tmp"
    temporary.write_text(yaml, encoding="utf-8")
    temporary.replace(out_dir / "data.yaml")
    print(f"Published {out_dir / 'data.yaml'}; previous builds remain available.")
    print({split: len(entries) for split, entries in splits.items()})
    return build_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--out_dir", default=str(YOLO_DIR))
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()
    prepare_dataset(args.manifest, args.out_dir, args.val_frac, args.test_frac)


if __name__ == "__main__":
    main()
