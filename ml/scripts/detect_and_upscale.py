"""
Phase 6 - Step 2: Detect plates (Phase 5's YOLO model) then super-resolve
each detected crop (Phase 6's Real-ESRGAN) -- so you can see what actually
reaches the OCR stage (Phase 7) at the end of the pipeline so far.

Usage:
    python ml/scripts/detect_and_upscale.py --image path/to/photo.jpg
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from super_resolve import upscale_image

ML_DIR = Path(__file__).resolve().parents[1]
YOLO_WEIGHTS = ML_DIR / "models" / "yolo_runs" / "plate_detector" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default=str(YOLO_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--out_dir", default=None, help="default: <image_dir>/<image_stem>_crops/")
    args = parser.parse_args()

    img_path = Path(args.image)
    out_dir = Path(args.out_dir) if args.out_dir else img_path.parent / f"{img_path.stem}_crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    results = model.predict(img, conf=args.conf, verbose=False)[0]
    boxes = results.boxes

    if len(boxes) == 0:
        print("No plates detected.")
        return

    print(f"Detected {len(boxes)} plate(s)")
    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        x1, y1, x2, y2 = xyxy
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue

        crop_path = out_dir / f"plate_{i}_conf{conf:.2f}_raw.jpg"
        cv2.imwrite(str(crop_path), crop)

        upscaled = upscale_image(crop, outscale=4)
        upscaled_path = out_dir / f"plate_{i}_conf{conf:.2f}_upscaled.jpg"
        cv2.imwrite(str(upscaled_path), upscaled)

        print(f"  Plate {i}: conf={conf:.2f}, box=({x1},{y1},{x2},{y2}), "
              f"raw crop {crop.shape[1]}x{crop.shape[0]} -> "
              f"upscaled {upscaled.shape[1]}x{upscaled.shape[0]}")
        print(f"    raw      -> {crop_path}")
        print(f"    upscaled -> {upscaled_path}")

    print(f"\nAll crops saved -> {out_dir}")


if __name__ == "__main__":
    main()