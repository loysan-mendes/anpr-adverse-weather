"""
Phase 1 - Step 1: Download the Indian Number Plates dataset from Kaggle
using kagglehub (handles caching, versioning, and unzipping for you).

Prereqs (run once, on YOUR local machine):
    1. pip install -r ml/requirements.txt      (includes kagglehub)
    2. Get your Kaggle API token:
         Kaggle.com -> Account -> Create New API Token -> downloads kaggle.json
    3. Place kaggle.json at:
         Linux/Mac: ~/.kaggle/kaggle.json   (chmod 600 ~/.kaggle/kaggle.json)
         Windows:   C:\\Users\\<you>\\.kaggle\\kaggle.json

Usage:
    python ml/scripts/download_dataset.py
    python ml/scripts/download_dataset.py --dataset other-user/other-slug

Note: this particular dataset (dataclusterlabs/indian-number-plates-dataset)
is the free SAMPLE release from DataCluster Labs -- images only, likely no
bounding-box annotations (the full 20k+ annotated set is only available by
request from the vendor). The EDA script will confirm exactly what we got.
"""

import argparse
import shutil
from pathlib import Path

import kagglehub

ML_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ML_DIR / "data" / "raw"

DEFAULT_DATASET = "dataclusterlabs/indian-number-plates-dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                         help=f"Kaggle dataset ref (default: {DEFAULT_DATASET})")
    parser.add_argument("--dest", default=str(RAW_DIR),
                         help="Where to copy the dataset (default: ml/data/raw)")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.dataset}' via kagglehub ...")
    cache_path = kagglehub.dataset_download(args.dataset)
    print(f"Downloaded to kagglehub cache: {cache_path}")

    print(f"Copying into project folder: {dest}")
    for item in Path(cache_path).iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    print("\nDone. Contents of", dest, ":")
    for p in sorted(dest.iterdir()):
        print(" -", p.name)

    print("\nNext: python ml/scripts/eda.py --data_dir ml/data/raw")


if __name__ == "__main__":
    main()