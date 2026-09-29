"""ANPR inference with tiled restoration, original-image fallback and abstention."""
import argparse
import json
import warnings
import time
import numpy as np
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from image_ops import merge_detections, tiled_restore, padded_box
from model_artifacts import resolve_detector, export_is_current
from predict_quality import load_model as load_quality_model, predict as predict_quality
from unet_model import UNet
from ocr_plate import recognize_plate_candidates
from plate_validator import select_candidate, choose_reading, strong_reading
from vehicle_detector import detect_vehicles, associate_plates

ML_DIR = Path(__file__).resolve().parents[1]
RESTORATION_DIR = ML_DIR / "models" / "restoration"
RESTORATION_CONDITIONS = {"haze", "rain", "blur"}
UPSCALE_DUAL_WIDTH_PX = 200
CONDITION_CONF_THRESHOLD = 0.55  # Uncalibrated policy; validate on an independent set.
_restoration_models = {}
_yolo_models = {}
_quality = None


def _load_quality_model_once():
    global _quality
    if _quality is None:
        _quality = load_quality_model()
    return _quality


def enhance_lowlight(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def load_restoration_model(condition, device):
    checkpoint = RESTORATION_DIR / condition / "best_model.pt"
    scripted = RESTORATION_DIR / condition / "model_scripted.pt"
    key = (condition, str(device), checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
    if key not in _restoration_models:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if export_is_current(checkpoint, scripted):
            model = torch.jit.load(str(scripted), map_location=device)
        else:
            config = ckpt.get("architecture", {"base": 32, "residual": any("outc_raw" in k for k in ckpt["model_state"])})
            model = UNet(**config)
            model.load_state_dict(ckpt["model_state"])
        _restoration_models[key] = (model.to(device).eval(), int(ckpt["img_size"]))
    return _restoration_models[key]


def restore_image(img_bgr, condition, device):
    model, size = load_restoration_model(condition, device)

    def predict_batch(tiles):
        rgb = np.ascontiguousarray(tiles[..., ::-1])
        tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        with torch.inference_mode():
            output = model(tensor).permute(0, 2, 3, 1).cpu().numpy()
        return (output[..., ::-1] * 255).clip(0, 255)

    return tiled_restore(img_bgr, None, size, predict_batch=predict_batch,
                         batch_size=2 if device.type == "cuda" else 1)


def load_yolo(weights=None):
    path = resolve_detector(weights)
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key not in _yolo_models:
        _yolo_models[key] = YOLO(str(path))
    return _yolo_models[key]


def read_crop(crop, source, use_upscale=True, profile="balanced"):
    readings = []
    def read(image, variant, upscaled=False):
        regions = recognize_plate_candidates(image)
        readings.append({**select_candidate(regions), "raw_ocr_regions": regions,
                         "upscaled": upscaled, "ocr_source": source, "ocr_variant": variant})
    read(crop, "original")
    if profile == "balanced" and strong_reading(readings[0]):
        return readings
    # Cheap fallback preserves character shapes instead of GAN-generated texture.
    if readings[0]["status"] != "accepted":
        target_height = 64
        scale = min(3.0, max(1.0, target_height / crop.shape[0]))
        larger = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if scale > 1.0:
            read(larger, "bicubic")
        if not strong_reading(choose_reading(readings)):
            lab = cv2.cvtColor(larger, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
            read(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), "contrast")
    selected = choose_reading(readings)
    if use_upscale and crop.shape[1] < UPSCALE_DUAL_WIDTH_PX and (profile == "exhaustive" or not strong_reading(selected)):
        try:
            from super_resolve import upscale_image
            upscaled = upscale_image(crop, outscale=4)
        except (ImportError, OSError, RuntimeError) as exc:
            message = f"Super-resolution unavailable; retained original crop: {exc}"
            warnings.warn(message, stacklevel=2)
            readings[0]["processing_warning"] = message
        else:
            read(upscaled, "realesrgan", True)
    return readings


def process_image(image_path, conf_threshold=0.25, weights=None, use_upscale=True,
                  profile="balanced", imgsz=640, vehicle_weights=None):
    started = time.perf_counter()
    if not 0 < conf_threshold <= 1 or profile not in {"balanced", "exhaustive"} or imgsz < 32:
        raise ValueError("Invalid confidence, profile or image size")
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    quality_model, checkpoint, device = _load_quality_model_once()
    quality = predict_quality(img, quality_model, checkpoint, device)
    quality_seconds = time.perf_counter()-started
    vehicle_started = time.perf_counter()
    vehicles = detect_vehicles(img, weights=vehicle_weights, imgsz=imgsz)
    vehicle_seconds = time.perf_counter()-vehicle_started
    selected_weights = resolve_detector(weights)
    yolo = load_yolo(selected_weights)
    calls = {"detector": 0, "ocr": 0, "restoration": 0, "vehicle_detector": 1}
    def detect(view, source, size):
        calls["detector"] += 1
        result = yolo.predict(view, conf=conf_threshold, imgsz=size, verbose=False)[0]
        return [{"box": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                 "confidence": float(box.conf[0]), "source": source} for box in result.boxes]
    detections = detect(img, "original", imgsz)
    if not detections:
        # One bounded higher-resolution retry for small or distant plates.
        detections = detect(img, "original_highres", min(1280, max(960, imgsz)))
    readings_by_box = {}
    def read_detection(detection, image, source):
        x1, y1, x2, y2 = padded_box(detection["box"], img.shape[1], img.shape[0])
        if x2 <= x1 or y2 <= y1:
            return []
        readings = read_crop(image[y1:y2, x1:x2], source, use_upscale, profile)
        calls["ocr"] += len(readings)
        return readings
    for detection in detections:
        readings_by_box[tuple(detection["box"])] = read_detection(detection, img, "original")
    all_strong = bool(detections) and all(
        readings_by_box[tuple(d["box"])] and strong_reading(choose_reading(readings_by_box[tuple(d["box"])]))
        for d in detections)
    condition = quality["condition"]
    confident_weather = quality["condition_confidence"] >= CONDITION_CONF_THRESHOLD
    severe = quality["severity"] == "severe" and quality["severity_confidence"] >= 0.5
    restore = confident_weather and (profile == "exhaustive" or severe or not all_strong)
    restoration, working_img = None, img
    if restore and condition in RESTORATION_CONDITIONS | {"lowlight"}:
        working_img = enhance_lowlight(img) if condition == "lowlight" else restore_image(img, condition, device)
        restoration = condition
        calls["restoration"] += 1
        detections = merge_detections(detections + detect(working_img, "restored", imgsz))
    plates = []
    for detection in detections:
        key = tuple(detection["box"])
        readings = readings_by_box.get(key)
        if readings is None:
            readings = read_detection(detection, img, "original")
        if restoration:
            readings = readings + read_detection(detection, working_img, "restored")
        if not readings:
            continue
        selected = choose_reading(readings)
        x1, y1, x2, y2 = detection["box"]
        plates.append({"box": [x1, y1, x2, y2], "detection_confidence": detection["confidence"],
                       "crop_width_px": x2-x1, **selected, "validation_score": selected["format_score"], "readings": readings})
    associate_plates(plates, vehicles)
    return {"image": str(image_path), "detector_weights": str(selected_weights), "quality": quality,
            "vehicles": vehicles, "num_vehicles_detected": len(vehicles),
            "restoration_applied": restoration, "restoration_reason": "weather_and_uncertain_or_severe" if restoration else None,
            "num_plates_detected": len(plates), "num_plates_accepted": sum(p["status"] == "accepted" for p in plates),
            "profile": profile, "stage_calls": calls,
            "timings_seconds": {"quality_and_decode": quality_seconds, "vehicle_detection": vehicle_seconds,
                                "total": time.perf_counter()-started}, "plates": plates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--weights", help="Override active detector checkpoint")
    parser.add_argument("--vehicle-weights", help="Override vehicle detector checkpoint (class names are checked)")
    parser.add_argument("--no-upscale", action="store_true")
    parser.add_argument("--profile", choices=["balanced", "exhaustive"], default="balanced")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(json.dumps(process_image(args.image, args.conf, args.weights, not args.no_upscale, args.profile, args.imgsz,
                                   vehicle_weights=args.vehicle_weights), indent=2))


if __name__ == "__main__":
    main()
