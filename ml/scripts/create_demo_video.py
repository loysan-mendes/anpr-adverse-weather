"""Create a letterboxed MP4 video demo from an image sequence in new-indian-vehicle-dataset.

Usage:
    .\\run_ml.ps1 ml/scripts/create_demo_video.py --prefix video11 --min-idx 980 --max-idx 1380 --out demo_vehicle_passage.mp4
"""

import argparse
import glob
from pathlib import Path
import cv2
import numpy as np


def build_video_clip(
    frames_dir: Path,
    prefix: str,
    min_idx: int,
    max_idx: int,
    output_path: Path,
    fps: float = 10.0,
    target_width: int = 1280,
    target_height: int = 720,
):
    files = glob.glob(str(frames_dir / f"{prefix}_*.jpg"))
    selected = []
    for f in files:
        fname = Path(f).stem
        parts = fname.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            idx = int(parts[-1])
            if min_idx <= idx <= max_idx:
                selected.append((idx, f))

    selected.sort(key=lambda x: x[0])
    if not selected:
        raise ValueError(f"No frames found matching {prefix}_*.jpg in range [{min_idx}, {max_idx}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))

    written = 0
    for idx, fpath in selected:
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(target_width / w, target_height / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        dx = (target_width - nw) // 2
        dy = (target_height - nh) // 2
        canvas[dy : dy + nh, dx : dx + nw] = resized
        writer.write(canvas)
        written += 1

    writer.release()
    print(f"Successfully compiled {written} frames to {output_path} ({target_width}x{target_height} @ {fps} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Compile dataset image sequence into MP4 video.")
    parser.add_argument("--dir", type=Path, default=Path("new-indian-vehicle-dataset/video_images"))
    parser.add_argument("--prefix", type=str, default="video11")
    parser.add_argument("--min-idx", type=int, default=980)
    parser.add_argument("--max-idx", type=int, default=1380)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("demo_vehicle_passage.mp4"))
    args = parser.parse_args()

    build_video_clip(
        frames_dir=args.dir,
        prefix=args.prefix,
        min_idx=args.min_idx,
        max_idx=args.max_idx,
        output_path=args.out,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
