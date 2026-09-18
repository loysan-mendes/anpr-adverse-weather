"""
Phase 5 - Step 2: Fine-tune a COCO-pretrained YOLOv11 on our plate
detection dataset (prepared by prepare_yolo_dataset.py).

We fine-tune rather than train from scratch -- with ~40 training photos
(x13 weather variants each), training a detector from random weights is
not viable. Starting from COCO-pretrained weights means the model already
knows general "what is an object / where are its edges" features; we're
just teaching it to specialize on "number_plate" as a class.

4a change: Default model raised from yolo11n (nano) to yolo11s (small).
  - Nano was chosen to prevent overfitting on 47 photos with the same
    clear image appearing in every weather variant (low diversity).
  - Small has ~2x more parameters than nano and is still under 30ms on CPU.
  - With 500+ source photos (or augmented variants already in the dataset)
    the extra capacity pays off -- better feature extraction without
    meaningfully higher overfitting risk.
  - Pass --model yolo11n.pt explicitly if you're still on a very small
    dataset or want the absolute fastest inference.

First run will auto-download the selected .pt weights (needs internet).

Usage:
    python ml/scripts/train_yolo.py --epochs 100
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ML_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = ML_DIR / "data" / "yolo_dataset" / "data.yaml"
RUN_PROJECT = ML_DIR / "models" / "yolo_runs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--model", default="yolo11s.pt",
                         help="starting weights: yolo11n.pt (nano, least overfit risk on tiny datasets) "
                              "up to yolo11x.pt (largest). Default is yolo11s (small) -- 2x params vs nano, "
                              "still <30ms on CPU, better suited once the dataset exceeds ~100 source photos.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20,
                         help="early stopping: stop if val metric doesn't improve for N epochs")
    parser.add_argument("--name", default="plate_detector")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(RUN_PROJECT),
        name=args.name,
        seed=42,
        # small-dataset-friendly augmentation: keep it, but not too aggressive
        # since our images already carry synthetic weather degradation
        mosaic=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
    )

    # run final validation explicitly so metrics are printed clearly at the end
    metrics = model.val(data=args.data)
    print("\n=== Final validation metrics ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    run_dir = RUN_PROJECT / args.name
    print(f"\nBest weights -> {run_dir / 'weights' / 'best.pt'}")
    print(f"Training plots/results -> {run_dir}")


if __name__ == "__main__":
    main()