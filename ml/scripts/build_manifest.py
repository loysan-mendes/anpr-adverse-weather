"""
Phase 1 - Step 3: Unify both annotation sets into one manifest.

We have two Pascal-VOC style XML sources with the SAME schema
(<object><name>number_plate</name><bndbox>...</bndbox>...</object>)
but one set additionally has a <number_plate_text> attribute per box:

    Indian_Number_Plates/Sample_Images  <-> Annotations/Annotations
        27 images, 1 box/image, no text label

    number_plate_images_ocr             <-> number_plate_annos_ocr
        47 images, multiple boxes/image, WITH plate text label

This script walks both annotation folders, matches each XML to its image
by filename, and writes one unified JSON manifest:

    ml/data/processed/manifest.json

Each entry:
{
    "image_path": "...",
    "width": 2448, "height": 3264,
    "source": "voc_no_text" | "voc_with_text",
    "boxes": [
        {"xmin": .., "ymin": .., "xmax": .., "ymax": .., "text": "KL34A465" or null}
    ]
}

Usage:
    python ml/scripts/build_manifest.py
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]

# (annotation_dir, image_dir, source_label) pairs — matches what we found in EDA
DEFAULT_PAIRS = [
    (
        ML_DIR / "data" / "raw" / "Annotations" / "Annotations",
        ML_DIR / "data" / "raw" / "Indian_Number_Plates" / "Sample_Images",
        "voc_no_text",
    ),
    (
        ML_DIR / "data" / "raw" / "number_plate_annos_ocr" / "number_plate_annos_ocr",
        ML_DIR / "data" / "raw" / "number_plate_images_ocr" / "number_plate_images_ocr",
        "voc_with_text",
    ),
]


def find_image(image_dir: Path, filename: str):
    """Find the image file, tolerating case/extension mismatches."""
    direct = image_dir / filename
    if direct.exists():
        return direct
    stem = Path(filename).stem.lower()
    for p in image_dir.glob("*"):
        if p.stem.lower() == stem:
            return p
    return None


def parse_xml(xml_path: Path, image_dir: Path, source_label: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", default="").strip()
    size = root.find("size")
    width = int(float(size.findtext("width", default="0"))) if size is not None else None
    height = int(float(size.findtext("height", default="0"))) if size is not None else None

    img_path = find_image(image_dir, filename)
    if img_path is None:
        return None, filename  # signal a miss

    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        box = {
            "xmin": float(bnd.findtext("xmin")),
            "ymin": float(bnd.findtext("ymin")),
            "xmax": float(bnd.findtext("xmax")),
            "ymax": float(bnd.findtext("ymax")),
            "text": None,
        }
        for attr in obj.findall("./attributes/attribute"):
            if attr.findtext("name") == "number_plate_text":
                box["text"] = attr.findtext("value")
        boxes.append(box)

    entry = {
        "image_path": str(img_path.relative_to(ML_DIR)),
        "width": width,
        "height": height,
        "source": source_label,
        "boxes": boxes,
    }
    return entry, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ML_DIR / "data" / "processed" / "manifest.json"))
    args = parser.parse_args()

    manifest = []
    misses = []

    for ann_dir, img_dir, label in DEFAULT_PAIRS:
        if not ann_dir.exists():
            print(f"WARNING: annotation dir not found, skipping: {ann_dir}")
            continue
        xml_files = list(ann_dir.glob("*.xml"))
        print(f"[{label}] found {len(xml_files)} XML files in {ann_dir}")
        for xml_path in xml_files:
            entry, missed_filename = parse_xml(xml_path, img_dir, label)
            if entry is None:
                misses.append(f"{label}: {missed_filename} (from {xml_path.name})")
            else:
                manifest.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_boxes = sum(len(e["boxes"]) for e in manifest)
    total_with_text = sum(1 for e in manifest for b in e["boxes"] if b["text"])

    print(f"\nWrote {len(manifest)} entries -> {out_path}")
    print(f"Total boxes: {total_boxes}  |  Boxes with plate text: {total_with_text}")
    if misses:
        print(f"\n{len(misses)} annotation files could not be matched to an image:")
        for m in misses:
            print(" -", m)


if __name__ == "__main__":
    main()