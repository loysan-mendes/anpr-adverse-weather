"""Evaluate complete ANPR on independently labeled full images, never GT crops."""

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dataset_utils import load_manifest, source_id, image_path, provenance, leakage_group
from image_ops import box_iou
from model_artifacts import sha256_file, atomic_json, resolve_detector
from plate_validator import clean

ML_DIR = Path(__file__).resolve().parents[1]


def collect_seen(checkpoints, historical_manifest, root=ML_DIR):
    """Older checkpoints have no provenance: conservatively exclude the old corpus."""
    combined = {"source_ids": set(), "image_sha256": set(), "source_sha256": set(),
                "leakage_groups": set(), "pixel_sha256": set()}
    fallback = None
    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint)
        sidecar = checkpoint.with_suffix(".provenance.json")
        if sidecar.is_file():
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if record.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise ValueError(f"Stale training provenance: {checkpoint}")
        else:
            if fallback is None:
                # Include ALL historical entries regardless of any later split labels.
                entries = [{k: v for k, v in e.items() if k != "split"}
                           for e in load_manifest(historical_manifest)]
                fallback = provenance(entries, root)
            record = fallback
        for key in combined:
            combined[key].update(record.get(key, []))
    return combined


def check_independent(entries, seen, root=ML_DIR):
    known_hashes = seen["image_sha256"] | seen["source_sha256"]
    for entry in entries:
        path = image_path(entry, root)
        if not path.is_file():
            raise FileNotFoundError(path)
        if entry.get("full_image_labels_verified") is False or entry.get("annotation_scope") == "provided_target_only":
            raise ValueError(f"Full-image labels are not verified; use component diagnostics or complete annotations: {path}")
        if entry.get("split") in {"train", "val"}:
            raise ValueError(f"Evaluation entry is marked {entry['split']}: {path}")
        if (source_id(entry) in seen["source_ids"] or sha256_file(path) in known_hashes
                or entry.get("source_sha256") in known_hashes
                or any(digest in known_hashes for digest in entry.get("duplicate_source_sha256", []))
                or leakage_group(entry) in seen.get("leakage_groups", set())
                or entry.get("pixel_sha256") in seen.get("pixel_sha256", set())
                or entry.get("source_pixel_sha256") in seen.get("pixel_sha256", set())):
            raise ValueError(f"Test overlaps training/model-selection data: {path}")
        for box in entry["boxes"]:
            if not box.get("text") or not clean(box["text"]):
                raise ValueError(f"Every test plate needs a transcription: {path}")
            if not (0 <= box["xmin"] < box["xmax"] <= entry["width"]
                    and 0 <= box["ymin"] < box["ymax"] <= entry["height"]):
                raise ValueError(f"Invalid test box: {path}")


def match_boxes(truth, predictions, threshold=0.5):
    """Maximum-cardinality one-to-one matching; prefer higher IoU edges."""
    neighbors = []
    for box in truth:
        coords = [box[k] for k in ("xmin", "ymin", "xmax", "ymax")]
        edges = [(i, box_iou(coords, p["box"])) for i, p in enumerate(predictions)]
        neighbors.append([i for i, overlap in sorted(edges, key=lambda e: -e[1]) if overlap >= threshold])
    owners = {}

    def assign(gt_index, visited):
        for prediction in neighbors[gt_index]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in owners or assign(owners[prediction], visited):
                owners[prediction] = gt_index
                return True
        return False

    for index in range(len(truth)):
        assign(index, set())
    return [(gt, pred) for pred, gt in owners.items()]


def score_image(entry, result, threshold=0.5):
    truth, predictions = entry["boxes"], result["plates"]
    matches = match_boxes(truth, predictions, threshold)
    exact = sum(predictions[p].get("status") == "accepted"
                and clean(predictions[p].get("plate_text", "")) == clean(truth[g]["text"])
                for g, p in matches)
    accepted = sum(p.get("status") == "accepted" for p in predictions)
    return {"images": 1, "ground_truth_plates": len(truth), "detections": len(predictions),
            "matched_detections": len(matches), "missed_plates": len(truth)-len(matches),
            "extra_detections": len(predictions)-len(matches), "accepted_readings": accepted,
            "exact_readings": exact, "incorrect_accepted_readings": accepted-exact,
            "abstentions": len(predictions)-accepted}


def summarize(rows):
    counts = {key: sum(row[key] for row in rows) for key in rows[0]} if rows else {}
    if not counts:
        return counts
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None
    return {**counts, "detection_recall": ratio("matched_detections", "ground_truth_plates"),
            "detection_precision": ratio("matched_detections", "detections"),
            "end_to_end_exact_recall": ratio("exact_readings", "ground_truth_plates"),
            "accepted_reading_precision": ratio("exact_readings", "accepted_readings")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Independent full-image test manifest, paths relative to --data-root")
    parser.add_argument("--data-root", type=Path, default=ML_DIR)
    parser.add_argument("--historical-manifest", type=Path, default=ML_DIR / "data/processed/manifest.json",
                        help="Entire original corpus used by legacy checkpoints without provenance")
    parser.add_argument("--weights")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-upscale", action="store_true")
    parser.add_argument("--profile", choices=["balanced", "exhaustive"], default="balanced")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=ML_DIR / "data/processed/pipeline_eval_results.json")
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")
    weights = resolve_detector(args.weights)
    checkpoints = [weights, ML_DIR / "models/quality_analyzer/best_model.pt"]
    checkpoints += [ML_DIR / f"models/restoration/{condition}/best_model.pt" for condition in ("blur", "haze", "rain")]
    entries = load_manifest(args.manifest)
    seen = collect_seen(checkpoints, args.historical_manifest)
    check_independent(entries, seen, args.data_root)
    from decision_engine import process_image

    details, buckets, elapsed = [], defaultdict(list), []
    for entry in entries:
        started = time.perf_counter()
        # Fail the run on inference errors; never silently omit difficult images.
        result = process_image(image_path(entry, args.data_root), args.conf, weights, not args.no_upscale,
                               profile=args.profile, imgsz=args.imgsz)
        elapsed.append(time.perf_counter()-started)
        counts = score_image(entry, result, args.iou)
        condition, severity = entry.get("condition", "unspecified"), entry.get("severity", "unspecified")
        for key in ("overall", f"condition/{condition}", f"condition_severity/{condition}/{severity}"):
            buckets[key].append(counts)
        details.append({"source_id": source_id(entry), "counts": counts, "result": result})
    report = {"evaluation": "full_pipeline_independent_test", "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": sha256_file(args.manifest), "source_photos": len({source_id(e) for e in entries}),
              "checkpoints": {str(p): sha256_file(p) for p in checkpoints},
              "code_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
              "settings": {"iou": args.iou, "detection_confidence": args.conf, "upscale": not args.no_upscale,
                           "profile": args.profile, "imgsz": args.imgsz},
              "metrics": {key: summarize(rows) for key, rows in buckets.items()},
              "latency_seconds": {"mean": sum(elapsed)/len(elapsed),
                                  "p95": sorted(elapsed)[math.ceil(0.95*len(elapsed))-1], "includes_cold_start": True},
              "details": details}
    atomic_json(args.out, report)
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
