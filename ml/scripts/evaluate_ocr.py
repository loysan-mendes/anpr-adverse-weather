"""
Phase 7+8 - Evaluate PaddleOCR (+ Phase 8 plate validation) against our 25
ground-truth plate texts (from the number_plate_text labels in Phase 1's
manifest.json).

Reports FOUR variants so we can see exactly what each stage contributes:
  - raw crop, naive join          (Phase 7 baseline)
  - raw crop, validated           (+ Phase 8 region scoring/correction)
  - upscaled crop, naive join     (Phase 6 + Phase 7 baseline)
  - upscaled crop, validated      (Phase 6 + Phase 7 + Phase 8)

We crop using the GROUND-TRUTH boxes (not YOLO's predicted boxes), so
errors here are isolated to OCR + validation, not detector errors.

Usage:
    python ml/scripts/evaluate_ocr.py
"""

import json
from pathlib import Path

import cv2

from ocr_plate import recognize_plate_candidates
from plate_validator import best_candidate
from super_resolve import upscale_image

ML_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_DIR / "data" / "processed" / "manifest.json"
OUT_PATH = ML_DIR / "data" / "processed" / "ocr_eval_results.json"


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def char_accuracy(gt: str, pred: str) -> float:
    if not gt:
        return 0.0
    dist = levenshtein(gt, pred)
    return max(0.0, 1 - dist / max(len(gt), len(pred), 1))


def naive_join(candidates):
    if not candidates:
        return "", 0.0
    text = "".join(c["text"] for c in candidates)
    conf = sum(c["conf"] for c in candidates) / len(candidates)
    return text, conf


def main():
    with open(MANIFEST_PATH) as f:
        entries = json.load(f)

    results = []
    for e in entries:
        img_path = ML_DIR / e["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue

        for box in e["boxes"]:
            if not box.get("text"):
                continue
            gt = box["text"].upper().replace(" ", "")

            xmin, ymin = int(box["xmin"]), int(box["ymin"])
            xmax, ymax = int(box["xmax"]), int(box["ymax"])
            crop = img[max(0, ymin):ymax, max(0, xmin):xmax]
            if crop.size == 0:
                continue

            raw_cands = recognize_plate_candidates(crop)
            raw_join, raw_conf = naive_join(raw_cands)
            raw_val, raw_val_score = best_candidate([c["text"] for c in raw_cands])

            upscaled = upscale_image(crop, outscale=4)
            up_cands = recognize_plate_candidates(upscaled)
            up_join, up_conf = naive_join(up_cands)
            up_val, up_val_score = best_candidate([c["text"] for c in up_cands])

            results.append({
                "gt": gt,
                "raw_join": raw_join, "raw_join_exact": raw_join == gt, "raw_join_acc": char_accuracy(gt, raw_join),
                "raw_val": raw_val, "raw_val_exact": raw_val == gt, "raw_val_acc": char_accuracy(gt, raw_val),
                "up_join": up_join, "up_join_exact": up_join == gt, "up_join_acc": char_accuracy(gt, up_join),
                "up_val": up_val, "up_val_exact": up_val == gt, "up_val_acc": char_accuracy(gt, up_val),
            })

    n = len(results)
    if n == 0:
        print("No ground-truth-labeled boxes found -- check manifest.json")
        return

    def summarize(key_exact, key_acc):
        exact = sum(r[key_exact] for r in results)
        acc = sum(r[key_acc] for r in results) / n
        return exact, acc

    rows = [
        ("Raw, naive join", "raw_join_exact", "raw_join_acc"),
        ("Raw, validated", "raw_val_exact", "raw_val_acc"),
        ("Upscaled, naive join", "up_join_exact", "up_join_acc"),
        ("Upscaled, validated", "up_val_exact", "up_val_acc"),
    ]

    print(f"Evaluated {n} ground-truth plates\n")
    print(f"{'':24}{'Exact match':>15}   {'Char accuracy':>14}")
    for label, ek, ak in rows:
        exact, acc = summarize(ek, ak)
        print(f"{label:24}{exact}/{n} ({exact/n:.1%}){'':>3}{acc:>10.1%}")

    print("\nPer-plate detail:")
    for r in results:
        print(f"  GT={r['gt']:14} | "
              f"raw={r['raw_join']:14}({'OK' if r['raw_join_exact'] else '  '}) -> "
              f"val={r['raw_val']:14}({'OK' if r['raw_val_exact'] else '  '}) | "
              f"up={r['up_join']:14}({'OK' if r['up_join_exact'] else '  '}) -> "
              f"val={r['up_val']:14}({'OK' if r['up_val_exact'] else '  '})")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results -> {OUT_PATH}")


if __name__ == "__main__":
    main()