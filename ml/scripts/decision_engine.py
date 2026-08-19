"""
AI Decision Engine: full pipeline integration, tying together everything
built in Phases 3-8.

Flow:
    Quality Analyzer (Phase 3)
      -> conditionally: Dehaze / Derain / Deblur (Phase 4)
      -> YOLOv11 Detection (Phase 5)
      -> per detected plate crop:
           conditionally: Real-ESRGAN upscale (Phase 6)
           -> PaddleOCR (Phase 7)
           -> Plate Validation (Phase 8)

Decision rules (empirically justified by our Phase 3/4/6/7/8 evaluations,
not arbitrary defaults):

  - Restoration (dehaze/derain/deblur) runs when the Quality Analyzer
    flags a matching condition AND either:
      a) severity != "none"  (the normal path), OR
      b) condition_confidence >= CONDITION_CONF_THRESHOLD (the fallback).
    Rationale for the fallback: the severity branch of the Quality Analyzer
    can misfire and predict "none" even when the condition branch correctly
    identifies blur/haze/rain with high confidence (observed empirically --
    the two heads are trained jointly but optimise independently, and the
    severity head has fewer training examples for "none" vs. the real
    degradation classes). Skipping restoration in that case leaves a
    visibly degraded image going into YOLO/OCR with no benefit.
    We still refuse to restore "clear" images (condition="clear" has
    low confidence for any degradation class) and we have no restoration
    branch for "lowlight", so that case is unchanged.

  - Upscaling only runs on crops narrower than UPSCALE_MIN_WIDTH_PX.
    Phase 7's evaluation showed blanket upscaling HURT OCR accuracy on
    already-adequate crops (56%->28% exact match) -- GAN-hallucinated
    texture confusing the OCR more often than resolution helped it. It's
    only worth the risk when a crop is genuinely too small to read
    reliably otherwise.

KNOWN LIMITATION: restoration models operate at a fixed 256x256 working
resolution (how they were trained, for speed/memory). Applying them to a
full high-res photo means downsizing to 256px, restoring, then upsampling
the result back to original size -- this loses some fine detail before
detection runs. Consistent with how we validated these models in Phase 4,
but a real tradeoff, not free.

Usage:
    python ml/scripts/decision_engine.py --image path/to/photo.jpg
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from predict_quality import load_model as load_quality_model, predict as predict_quality
from unet_model import UNet
from ocr_plate import recognize_plate_candidates
from plate_validator import best_candidate
from super_resolve import upscale_image

ML_DIR = Path(__file__).resolve().parents[1]
YOLO_WEIGHTS = ML_DIR / "models" / "yolo_runs" / "plate_detector" / "weights" / "best.pt"
RESTORATION_DIR = ML_DIR / "models" / "restoration"
RESTORATION_CONDITIONS = {"haze", "rain", "blur"}  # matches the architecture's 3 branches

UPSCALE_MIN_WIDTH_PX = 150  # crops narrower than this get upscaled before OCR

# If the quality model is this confident about a degradation condition,
# apply restoration even when the severity head says "none" -- the severity
# head sometimes misfires while the condition head is clearly right.
# Empirically calibrated at 0.55: a blur/moderate image scores 0.606 on
# "blur" vs 0.296 on "clear" -- clearly dominant, restoration should fire.
# A genuinely clear image typically has ~0.85+ on "clear" and <0.1 on any
# degradation class, so 0.55 gives plenty of margin against false triggers.
CONDITION_CONF_THRESHOLD = 0.55

_restoration_models = {}  # condition -> (model, img_size)
_yolo_model = None


def load_restoration_model(condition, device):
    if condition in _restoration_models:
        return _restoration_models[condition]
    ckpt_path = RESTORATION_DIR / condition / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet(base=32)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    _restoration_models[condition] = (model, ckpt["img_size"])
    return _restoration_models[condition]


def restore_image(img_bgr, condition, device):
    model, img_size = load_restoration_model(condition, device)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)[0].cpu().permute(1, 2, 0).numpy()
    out = (out * 255).clip(0, 255).astype("uint8")
    out_full = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(out_full, cv2.COLOR_RGB2BGR)


def load_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
    return _yolo_model


def process_image(image_path, conf_threshold=0.25):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    quality_model, qckpt, device = load_quality_model()
    quality = predict_quality(image_path, quality_model, qckpt, device)
    condition = quality["condition"]
    severity = quality["severity"]
    condition_conf = quality["condition_confidence"]
    severity_conf = quality["severity_confidence"]

    working_img = img
    restoration_applied = None
    restoration_reason = None
    # Apply restoration if:
    #   (a) severity is explicitly non-"none", OR
    #   (b) the condition head is confident enough that we trust it even
    #       when the severity head misfired as "none".
    if condition in RESTORATION_CONDITIONS:
        if severity != "none":
            restoration_reason = "severity"
        elif condition_conf >= CONDITION_CONF_THRESHOLD:
            restoration_reason = "condition_confidence_fallback"
        if restoration_reason:
            working_img = restore_image(img, condition, device)
            restoration_applied = condition

    yolo = load_yolo()
    results = yolo.predict(working_img, conf=conf_threshold, verbose=False)[0]

    plates = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        det_conf = float(box.conf[0])
        x1, y1, x2, y2 = xyxy
        crop = working_img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue

        crop_w = crop.shape[1]
        upscaled_applied = False
        ocr_input = crop
        if crop_w < UPSCALE_MIN_WIDTH_PX:
            ocr_input = upscale_image(crop, outscale=4)
            upscaled_applied = True

        candidates = recognize_plate_candidates(ocr_input)
        plate_text, val_score = best_candidate([c["text"] for c in candidates])

        plates.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "detection_confidence": round(det_conf, 3),
            "crop_width_px": int(crop_w),
            "upscaled": upscaled_applied,
            "plate_text": plate_text,
            "validation_score": round(val_score, 3),
            "raw_ocr_regions": candidates,
        })

    return {
        "image": str(image_path),
        "quality": {
            "condition": condition,
            "condition_confidence": round(condition_conf, 3),
            "severity": severity,
            "severity_confidence": round(severity_conf, 3),
        },
        "restoration_applied": restoration_applied,
        "restoration_reason": restoration_reason,
        "num_plates_detected": len(plates),
        "plates": plates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    result = process_image(args.image, conf_threshold=args.conf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()