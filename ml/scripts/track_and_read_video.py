"""Video tracking and multi-frame plate recognition for vehicle entrance monitoring.

Tracks vehicles with persistent track IDs (ByteTrack), associates license plates,
selects the sharpest observations per passage, and applies temporal consensus fusion.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Optional
import cv2
import numpy as np

from decision_engine import load_yolo, read_crop
from model_artifacts import resolve_detector
from temporal_fusion import estimate_sharpness, fuse_observations
from vehicle_detector import load_vehicle_model, VEHICLE_CLASSES


class VideoANPRTracker:
    def __init__(
        self,
        vehicle_weights: Optional[Path] = None,
        plate_weights: Optional[Path] = None,
        tracker: str = "bytetrack.yaml",
        frame_stride: int = 2,
        max_ocr_per_track: int = 5,
    ):
        self.frame_stride = max(1, frame_stride)
        self.max_ocr_per_track = max(1, max_ocr_per_track)
        self.tracker = tracker

        self.vehicle_model = load_vehicle_model(vehicle_weights)
        self.plate_model = load_yolo(plate_weights)

    def process_video(
        self,
        video_path: Path,
        out_json: Optional[Path] = None,
        out_video: Optional[Path] = None,
        conf_vehicle: float = 0.35,
        conf_plate: float = 0.25,
        acceptance_min_confidence: float = 0.88,
    ) -> Dict:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        writer = None
        if out_video:
            out_video = Path(out_video)
            out_video.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, fps / self.frame_stride, (width, height))

        # Track state store: track_id -> vehicle metadata & plate observation candidates
        tracks: Dict[int, Dict] = {}
        frame_idx = 0
        processed_frames = 0
        started = time.perf_counter()

        # Class IDs of interest for vehicles
        classes = {i: name for i, name in self.vehicle_model.names.items() if name in VEHICLE_CLASSES.values()}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_stride != 0:
                    frame_idx += 1
                    continue

                timestamp_sec = frame_idx / fps
                processed_frames += 1

                # 1. Multi-object vehicle tracking with ByteTrack
                v_res = self.vehicle_model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker,
                    classes=list(classes.keys()),
                    conf=conf_vehicle,
                    verbose=False,
                )[0]

                active_vehicles = []
                for box in v_res.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in classes:
                        continue
                    track_id = int(box.id[0]) if box.id is not None else None
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf[0])
                    v_type = classes[cls_id]

                    if track_id is not None:
                        active_vehicles.append({
                            "track_id": track_id,
                            "box": xyxy,
                            "type": v_type,
                            "conf": conf,
                        })

                        if track_id not in tracks:
                            tracks[track_id] = {
                                "track_id": track_id,
                                "vehicle_type": v_type,
                                "first_seen_frame": frame_idx,
                                "first_seen_sec": round(timestamp_sec, 2),
                                "last_seen_frame": frame_idx,
                                "last_seen_sec": round(timestamp_sec, 2),
                                "confs": [conf],
                                "plate_crops": [],
                            }
                        else:
                            tracks[track_id]["last_seen_frame"] = frame_idx
                            tracks[track_id]["last_seen_sec"] = round(timestamp_sec, 2)
                            tracks[track_id]["confs"].append(conf)

                # 2. Plate detection
                p_res = self.plate_model.predict(
                    frame,
                    conf=conf_plate,
                    verbose=False,
                )[0]

                detected_plates = []
                for box in p_res.boxes:
                    pxyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    detected_plates.append({
                        "box": pxyxy,
                        "conf": float(box.conf[0]),
                    })

                # 3. Associate plates to tracked vehicles
                for plate in detected_plates:
                    px1, py1, px2, py2 = plate["box"]
                    p_area = max(1, (px2 - px1) * (py2 - py1))
                    best_v = None

                    for v in active_vehicles:
                        vx1, vy1, vx2, vy2 = v["box"]
                        inter_w = max(0, min(px2, vx2) - max(px1, vx1))
                        inter_h = max(0, min(py2, vy2) - max(py1, vy1))
                        intersection = inter_w * inter_h
                        if intersection / p_area >= 0.85:
                            best_v = v
                            break

                    if best_v:
                        t_id = best_v["track_id"]
                        crop = frame[max(0, py1):min(height, py2), max(0, px1):min(width, px2)]
                        if crop.size > 0:
                            sharpness = estimate_sharpness(crop)
                            tracks[t_id]["plate_crops"].append({
                                "frame_idx": frame_idx,
                                "timestamp_sec": round(timestamp_sec, 2),
                                "plate_box": plate["box"],
                                "crop": crop,
                                "sharpness": sharpness,
                            })

                # Optional video annotation frame rendering
                if writer:
                    annotated = frame.copy()
                    for v in active_vehicles:
                        vx1, vy1, vx2, vy2 = v["box"]
                        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (180, 80, 240), 2)
                        label = f"V{v['track_id']} {v['type'].upper()} {round(v['conf']*100)}%"
                        cv2.putText(annotated, label, (vx1, max(20, vy1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 80, 240), 2)

                    for p in detected_plates:
                        px1, py1, px2, py2 = p["box"]
                        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 230, 100), 2)

                    writer.write(annotated)

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()

        # 4. Multi-frame OCR consensus per tracked vehicle passage
        passages = []
        for track_id, t_info in sorted(tracks.items(), key=lambda item: item[1]["first_seen_frame"]):
            crops = t_info.pop("plate_crops", [])
            mean_v_conf = float(np.mean(t_info.pop("confs", [0.0])))

            if not crops:
                consensus = {
                    "plate_text": "",
                    "proposed_text": "",
                    "status": "unreadable",
                    "review_reason": "no_plate_detected_for_vehicle",
                    "fused_confidence": 0.0,
                    "format_score": 0.0,
                    "agreement_count": 0,
                    "total_observations": 0,
                }
            else:
                # Select top sharpest candidate frames to minimize redundant OCR
                sorted_crops = sorted(crops, key=lambda c: c["sharpness"], reverse=True)
                selected_for_ocr = sorted_crops[:self.max_ocr_per_track]

                observations = []
                for item in selected_for_ocr:
                    readings = read_crop(item["crop"], "original", use_upscale=False, profile="balanced")
                    if readings:
                        observations.append({
                            "frame_idx": item["frame_idx"],
                            "timestamp_sec": item["timestamp_sec"],
                            "sharpness": item["sharpness"],
                            "reading": readings[0],
                        })

                consensus = fuse_observations(
                    observations,
                    acceptance_min_confidence=acceptance_min_confidence,
                )

            passages.append({
                "track_id": track_id,
                "vehicle_type": t_info["vehicle_type"],
                "vehicle_confidence": round(mean_v_conf, 4),
                "first_seen_sec": t_info["first_seen_sec"],
                "last_seen_sec": t_info["last_seen_sec"],
                "total_plate_frames": len(crops),
                **{k: v for k, v in consensus.items() if k != "observations"},
            })

        total_time = round(time.perf_counter() - started, 2)
        result = {
            "video_path": str(video_path),
            "video_metadata": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": fps,
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}",
            },
            "processing_time_seconds": total_time,
            "num_vehicles_tracked": len(passages),
            "num_accepted_plates": sum(1 for p in passages if p["status"] == "accepted"),
            "num_review_plates": sum(1 for p in passages if p["status"] == "uncertain"),
            "passages": passages,
        }

        if out_json:
            out_json = Path(out_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Input video file")
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--out-video", type=Path, default=None, help="Output annotated video path")
    parser.add_argument("--frame-stride", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--conf-vehicle", type=float, default=0.35)
    parser.add_argument("--conf-plate", type=float, default=0.25)
    args = parser.parse_args()

    tracker = VideoANPRTracker(frame_stride=args.frame_stride)
    results = tracker.process_video(
        args.video,
        out_json=args.out_json,
        out_video=args.out_video,
        conf_vehicle=args.conf_vehicle,
        conf_plate=args.conf_plate,
    )
    print(f"Processed {results['video_metadata']['processed_frames']} frames in {results['processing_time_seconds']}s")
    print(f"Tracked {results['num_vehicles_tracked']} vehicles -> {results['num_accepted_plates']} accepted plates")
    print(json.dumps(results["passages"], indent=2))


if __name__ == "__main__":
    main()
