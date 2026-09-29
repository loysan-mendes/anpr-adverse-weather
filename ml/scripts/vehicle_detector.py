"""Four-class vehicle detection and conservative, scene-local plate association."""
from pathlib import Path
import json
from model_artifacts import cached_sha256

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "yolo11n.pt"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_models = {}
REGISTRY = DEFAULT_WEIGHTS.parent / 'models' / 'active_vehicle_detector.json'


def resolve_vehicle_weights(weights=None):
    if weights is not None:
        return Path(weights).resolve()
    if REGISTRY.exists():
        record = json.loads(REGISTRY.read_text())
        path = (DEFAULT_WEIGHTS.parent / record['weights']).resolve()
        if not path.is_file() or cached_sha256(path) != record['sha256']:
            raise ValueError('Active vehicle checkpoint is missing or changed; reselect a verified checkpoint.')
        return path
    return DEFAULT_WEIGHTS.resolve()


def load_vehicle_model(weights=None):
    path = resolve_vehicle_weights(weights)
    if not path.is_file():
        raise FileNotFoundError("Vehicle weights missing. Run: python ml/scripts/vehicle_detector.py --download")
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key not in _models:
        from ultralytics import YOLO
        model = YOLO(str(path))
        if not set(VEHICLE_CLASSES.values()).issubset(set(model.names.values())):
            raise ValueError("Vehicle detector must include car, motorcycle, bus, and truck labels.")
        _models[key] = model
    return _models[key]


def detect_vehicles(image, weights=None, conf=0.35, imgsz=640):
    model = load_vehicle_model(weights)
    classes = {i: ('unknown' if name == 'other' else name) for i,name in model.names.items()
               if name in set(VEHICLE_CLASSES.values()) | {'other'}}
    result = model.predict(image, classes=list(classes), conf=conf,
                           imgsz=imgsz, agnostic_nms=True, verbose=False)[0]
    vehicles = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id not in classes:
            continue
        vehicles.append({"id": len(vehicles) + 1,
                         "box": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                         "vehicle_type": classes[class_id],
                         "confidence": float(box.conf[0])})
    return vehicles


def associate_plates(plates, vehicles, min_confidence=0.5):
    """Require a unique containing box; abstain on shared/overlapping assignments.

    Thresholds are uncalibrated policy choices, not correctness probabilities.
    All boxes must use the same full-image coordinates.
    """
    assignments = []
    for plate in plates:
        x1, y1, x2, y2 = plate["box"]
        area = max(0, x2-x1) * max(0, y2-y1)
        candidates = []
        for vehicle in vehicles:
            a, b, c, d = vehicle["box"]
            intersection = max(0, min(x2, c)-max(x1, a)) * max(0, min(y2, d)-max(y1, b))
            if area and intersection/area >= 0.9 and a <= (x1+x2)/2 <= c and b <= (y1+y2)/2 <= d:
                candidates.append(vehicle)
        status = "unmatched"
        match = None
        if len(candidates) > 1:
            status = "ambiguous"
        elif candidates:
            if candidates[0]['vehicle_type'] == 'unknown':
                status = 'unsupported'
            elif candidates[0]["confidence"] >= min_confidence:
                status, match = "matched", candidates[0]
            else:
                status = "low_confidence"
        assignments.append((status, match))
    for plate, (status, match) in zip(plates, assignments):
        if match and sum(other is not None and other["id"] == match["id"] for _, other in assignments) > 1:
            status, match = "ambiguous", None
        plate.update(vehicle_id=match["id"] if match else None,
                     vehicle_type=match["vehicle_type"] if match else "unknown",
                     vehicle_confidence=match["confidence"] if match else None,
                     vehicle_match_status=status)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    from ultralytics import YOLO
    YOLO(str(DEFAULT_WEIGHTS))
    load_vehicle_model()
    print(f"Vehicle detector ready: {DEFAULT_WEIGHTS}")
