"""Repeatable diagnostic benchmark. For independent accuracy use evaluate_pipeline.py."""
import argparse
import inspect
import json
import statistics
import time
from pathlib import Path
from dataset_utils import load_manifest, image_path
from model_artifacts import atomic_json, sha256_file
from evaluate_pipeline import score_image, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-upscale", action="store_true")
    args = parser.parse_args()
    from decision_engine import process_image
    entries = [e for e in load_manifest(args.manifest) if e["boxes"] and all(b.get("text") for b in e["boxes"])]
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError("No fully transcribed images")
    options = {"use_upscale": not args.no_upscale}
    if "profile" in inspect.signature(process_image).parameters:
        options["profile"] = args.profile
    report = {"evaluation": "development_diagnostic_not_independent_accuracy",
              "options": options, "manifest_sha256": sha256_file(args.manifest),
              "code_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
              "images": []}
    print(f"BENCHMARK_READY: {len(entries)} images", flush=True)
    for i, entry in enumerate(entries):
        start = time.perf_counter()
        result = process_image(image_path(entry), **options)
        seconds = time.perf_counter()-start
        counts = score_image(entry, result)
        report["images"].append({"image": entry["image_path"], "seconds": seconds, "counts": counts, "result": result})
        atomic_json(args.out, report)
        print(f"{i+1}/{len(entries)} {seconds:.2f}s exact={counts['exact_readings']}/{counts['ground_truth_plates']}", flush=True)
    report["metrics"] = summarize([r["counts"] for r in report["images"]])
    times = [r["seconds"] for r in report["images"]]
    report["latency"] = {"cold_first_seconds": times[0], "mean_seconds": statistics.mean(times),
                         "warm_mean_seconds": statistics.mean(times[1:]) if len(times)>1 else None}
    atomic_json(args.out, report)
    print(json.dumps({"metrics": report["metrics"], "latency": report["latency"]}, indent=2))


if __name__ == "__main__":
    main()
