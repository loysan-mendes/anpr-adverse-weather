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
    low confidence for any degradation class).

  - Lowlight (3d): handled with a fast CLAHE enhancer in LAB colour space
    rather than a UNet -- no retraining needed. CLAHE on the L channel
    boosts local contrast without shifting hue or saturation.

  - Upscaling only runs on crops narrower than UPSCALE_MIN_WIDTH_PX.
    Phase 7's evaluation showed blanket upscaling HURT OCR accuracy on
    already-adequate crops (56%->28% exact match) -- GAN-hallucinated
    texture confusing the OCR more often than resolution helped it. It's
    only worth the risk when a crop is genuinely too small to read
    reliably otherwise.

KNOWN LIMITATION: UNet restoration models operate at a fixed 512x512 working
resolution (3a change -- was 256x256). Applying them to a full high-res photo
means downsizing, restoring, then upsampling back -- this loses some fine
detail before detection runs. Consistent with how we validated these models
in Phase 4, but a real tradeoff, not free.

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

UPSCALE_MIN_WIDTH_PX = 150   # crops narrower than this always get upscaled before OCR
UPSCALE_DUAL_WIDTH_PX = 200  # 5b: grey zone -- try both raw and upscaled, keep better score

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

# 6a: Quality model singleton -- loaded once at module level so the
# checkpoint is only deserialised and copied to device once per process,
# not on every process_image() call. In a fresh one-shot CLI run the
# savings are small; in a server/batch scenario this eliminates the
# biggest repeated startup cost.
_quality_model = None
_quality_ckpt = None
_quality_device = None


def _load_quality_model_once():
    global _quality_model, _quality_ckpt, _quality_device
    if _quality_model is None:
        _quality_model, _quality_ckpt, _quality_device = load_quality_model()
    return _quality_model, _quality_ckpt, _quality_device


# ----------------------------------------------------------- lowlight (3d) ---
def enhance_lowlight(img_bgr):
    """3d: CLAHE enhancer for lowlight images -- no UNet, no retraining.

    Operates on the L channel of LAB colour space so only luminance is
    boosted; hue and saturation (A, B channels) are untouched. This avoids
    the colour-shift artefacts you get from equalising in BGR directly.
    clipLimit=3.0 and tileGridSize=(8,8) are the standard values from the
    CLAHE literature and work well across a wide range of plate images.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def load_restoration_model(condition, device):
    if condition in _restoration_models:
        return _restoration_models[condition]
    ckpt_path = RESTORATION_DIR / condition / "best_model.pt"
    scripted_path = RESTORATION_DIR / condition / "model_scripted.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if scripted_path.exists():
        # 6b: Prefer the TorchScript-compiled version -- 20-40% faster on CPU,
        # same weights as best_model.pt, no accuracy cost. Run
        # ml/scripts/export_unet.py once after training to generate these.
        model = torch.jit.load(str(scripted_path), map_location=device)
    else:
        is_residual = any("outc_raw" in k for k in ckpt["model_state"].keys())
        model = UNet(base=32, residual=is_residual)
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
    # 6c: fp16 for GPU inference -- ~30-50% faster on NVIDIA GPUs with no
    # accuracy cost. CPU PyTorch does not support fp16 matmul, so only cast
    # when actually on CUDA. The singleton model is cast in-place on first
    # GPU call; subsequent calls reuse the already-half model from the cache.
    if device.type == "cuda":
        tensor = tensor.half()
        model = model.half()
    with torch.no_grad():
        out = model(tensor)[0].cpu().float().permute(1, 2, 0).numpy()
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

    # 6a: Reuse the module-level singleton -- avoids a repeated disk read +
    # checkpoint deserialise on every call inside a long-running process.
    quality_model, qckpt, device = _load_quality_model_once()
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
    if condition == "lowlight":
        # 3d: CLAHE lowlight branch -- fast, no UNet needed.
        if severity != "none" or condition_conf >= CONDITION_CONF_THRESHOLD:
            working_img = enhance_lowlight(img)
            restoration_applied = "lowlight"
            restoration_reason = "severity" if severity != "none" else "condition_confidence_fallback"
    elif condition in RESTORATION_CONDITIONS:
        if severity != "none":
            restoration_reason = "severity"
        elif condition_conf >= CONDITION_CONF_THRESHOLD:
            restoration_reason = "condition_confidence_fallback"
        if restoration_reason:
            working_img = restore_image(img, condition, device)
            restoration_applied = condition

    yolo = load_yolo()
    # 4c: Post-restoration images can score slightly lower YOLO confidence
    # because the 512px working-resolution cycle subtly alters texture/contrast
    # -- a fixed threshold would silently drop valid plates that restoration
    # itself surfaced. Scale down by 20% whenever any restoration ran.
    eff_conf = conf_threshold * 0.8 if restoration_applied else conf_threshold
    results = yolo.predict(working_img, conf=eff_conf, verbose=False)[0]

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
            # Clearly too narrow -- always upscale before OCR.
            ocr_input = upscale_image(crop, outscale=4)
            upscaled_applied = True
        elif crop_w < UPSCALE_DUAL_WIDTH_PX:
            # 5b: Grey zone (150–200px) -- try BOTH raw and upscaled, keep
            # whichever the plate_validator scores higher. The GAN upscaler
            # can hurt wide crops (hallucinated texture confuses OCR) but
            # helps near-threshold ones, so we let the score decide.
            upscaled_crop = upscale_image(crop, outscale=4)
            raw_candidates = recognize_plate_candidates(crop)
            up_candidates = recognize_plate_candidates(upscaled_crop)
            raw_text, raw_score = best_candidate([c["text"] for c in raw_candidates])
            up_text, up_score = best_candidate([c["text"] for c in up_candidates])
            if up_score > raw_score:
                candidates = up_candidates
                plate_text, val_score = up_text, up_score
                upscaled_applied = True
            else:
                candidates = raw_candidates
                plate_text, val_score = raw_text, raw_score
            plates.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "detection_confidence": round(det_conf, 3),
                "crop_width_px": int(crop_w),
                "upscaled": upscaled_applied,
                "plate_text": plate_text,
                "validation_score": round(val_score, 3),
                "raw_ocr_regions": candidates,
            })
            continue  # already appended above -- skip the append below

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