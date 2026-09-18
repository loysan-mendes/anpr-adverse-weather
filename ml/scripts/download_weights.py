"""
Helper script to download and install pre-trained model weights from GitHub Releases.

Usage:
    python ml/scripts/download_weights.py
    python ml/scripts/download_weights.py --tag v1.0.0 --force
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "loysan-mendes/anpr-adverse-weather"
DEFAULT_TAG = "v1.0.0"
ASSET_NAME = "models-v1.0.0.zip"

EXPECTED_FILES = [
    Path("ml/models/quality_analyzer/best_model.pt"),
    Path("ml/models/restoration/blur/best_model.pt"),
    Path("ml/models/restoration/haze/best_model.pt"),
    Path("ml/models/restoration/rain/best_model.pt"),
    Path("ml/models/yolo_runs/plate_detector/weights/best.pt"),
]


def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading {ASSET_NAME}: {mb_down:.1f}/{mb_total:.1f} MB ({percent:.1f}%)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\rDownloading {ASSET_NAME}: {mb_down:.1f} MB")
    sys.stdout.flush()


def download_and_extract(tag: str = DEFAULT_TAG, force: bool = False, root_dir: Path = None):
    if root_dir is None:
        # Assume root is parent of ml/
        root_dir = Path(__file__).resolve().parents[2]

    # Check if all files already exist
    all_present = all((root_dir / f).exists() for f in EXPECTED_FILES)
    if all_present and not force:
        print("[INFO] All pre-trained model weights are already present:")
        for f in EXPECTED_FILES:
            print(f"  - {f}")
        print("Use --force to re-download and overwrite.")
        return

    url = f"https://github.com/{REPO}/releases/download/{tag}/{ASSET_NAME}"
    dest_zip = root_dir / ASSET_NAME

    print(f"Fetching weights from: {url}")
    try:
        urllib.request.urlretrieve(url, dest_zip, reporthook=reporthook)
        print("\n[INFO] Download complete. Extracting files...")
    except Exception as exc:
        print(f"\n[ERROR] Failed to download weights from {url}: {exc}")
        print("\nIf the release has not been published yet, you can also manually")
        print(f"download '{ASSET_NAME}' and extract it into the project root.")
        sys.exit(1)

    try:
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(root_dir)
        print("[SUCCESS] Model weights extracted successfully:")
        for f in EXPECTED_FILES:
            full_p = root_dir / f
            status = "OK" if full_p.exists() else "MISSING"
            print(f"  [{status}] {f}")
    finally:
        if dest_zip.exists():
            dest_zip.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained weights for ANPR adverse weather pipeline")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag to download from (default: {DEFAULT_TAG})")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing weights")
    args = parser.parse_args()

    download_and_extract(tag=args.tag, force=args.force)


if __name__ == "__main__":
    main()
