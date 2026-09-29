"""Conservative OCR selection. Scores are heuristics, not correctness probabilities."""
import argparse
import itertools
import json
import math
import re

L2D = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0", "D": "0", "T": "7"}
D2L = {digit: tuple(k for k, v in L2D.items() if v == digit) for digit in set(L2D.values())}
TEMPLATES = ["LLDDLDDDD", "LLDDLLDDDD", "LLDLDDDD", "LLDLLDDDD", "LLDDDDDD", "LLDDLDDD"]


def clean(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def coerce_with_cost(text, template):
    """Keep all ambiguous substitutions; never silently choose O/Q/D."""
    if len(text) != len(template):
        return []
    choices, cost = [], 0
    for character, expected in zip(text, template):
        if (expected == "L" and character.isalpha()) or (expected == "D" and character.isdigit()):
            choices.append((character,))
        elif expected == "L" and character in D2L:
            choices.append(D2L[character])
            cost += 1
        elif expected == "D" and character in L2D:
            choices.append((L2D[character],))
            cost += 1
        else:
            return []
    return [("".join(parts), cost) for parts in itertools.product(*choices)]


def template_fits(text):
    fits = {}
    for template in TEMPLATES:
        for proposed, cost in coerce_with_cost(text, template):
            # A dropped series letter can masquerade as a no-series plate.
            # Keep this format for suggestions, requiring human review.
            base = 0.70 if template == "LLDDDDDD" else 1.0
            score = max(0.0, base - 0.08 * cost)
            fits[proposed] = max(score, fits.get(proposed, 0.0))
    return fits


def best_template_fit(text):
    fits = sorted(template_fits(clean(text)).items(), key=lambda item: (-item[1], item[0]))
    if not fits or (len(fits) > 1 and fits[0][1] == fits[1][1]):
        return None
    return fits[0]


def order_regions(regions):
    """Cluster boxes by vertical overlap, then read each row left to right."""
    if not regions or any(not r.get("box") for r in regions):
        return list(regions)
    rows = []
    for region in sorted(regions, key=lambda r: (r["box"][1] + r["box"][3]) / 2):
        x1, y1, x2, y2 = region["box"]
        matches = []
        for row in rows:
            top = sum(r["box"][1] for r in row) / len(row)
            bottom = sum(r["box"][3] for r in row) / len(row)
            overlap = max(0.0, min(bottom, y2)-max(top, y1)) / max(1.0, min(bottom-top, y2-y1))
            if overlap >= 0.5:
                matches.append((overlap, row))
        if matches:
            max(matches, key=lambda item: item[0])[1].append(region)
        else:
            rows.append([region])
    rows.sort(key=lambda row: min(r["box"][1] for r in row))
    return [r for row in rows for r in sorted(row, key=lambda r: r["box"][0])]


def select_candidate(candidates, min_confidence=0.80, min_score=0.75, ambiguity_margin=0.08):
    regions = []
    for candidate in candidates:
        region = dict(candidate) if isinstance(candidate, dict) else {"text": candidate, "conf": 0.0}
        region["text"] = clean(region["text"])
        if not region["text"] or region["text"] == "IND":
            continue
        confidence = float(region.get("conf", 0.0))
        region["conf"] = max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0
        regions.append(region)
    regions = order_regions(regions)
    hypotheses = {}
    for start in range(len(regions)):
        for end in range(start+1, len(regions)+1):
            parts = regions[start:end]
            text = "".join(r["text"] for r in parts)
            if len(text) > 10:
                break
            confidence = min(r["conf"] for r in parts)
            omitted = regions[:start] + regions[end:]
            penalty = 0.08 * sum(len(r["text"]) <= 4 for r in omitted)
            for proposed, format_score in template_fits(text).items():
                score = max(0.0, confidence * format_score - penalty)
                item = {"text": proposed, "format_score": format_score, "ocr_confidence": confidence, "selection_score": score}
                if proposed not in hypotheses or score > hypotheses[proposed]["selection_score"]:
                    hypotheses[proposed] = item
    ranked = sorted(hypotheses.values(), key=lambda r: (-r["selection_score"], r["text"]))
    if not ranked:
        return {"plate_text": "", "proposed_text": "", "status": "unreadable", "format_score": 0.0,
                "ocr_confidence": 0.0, "selection_score": 0.0, "alternatives": []}
    best = ranked[0]
    ambiguous = len(ranked) > 1 and best["selection_score"] - ranked[1]["selection_score"] < ambiguity_margin
    accepted = (not ambiguous and best["ocr_confidence"] >= min_confidence
                and best["selection_score"] >= min_score and best["format_score"] >= 0.84)
    return {"plate_text": best["text"] if accepted else "", "proposed_text": best["text"],
            "status": "accepted" if accepted else "uncertain",
            **{k: best[k] for k in ("format_score", "ocr_confidence", "selection_score")}, "alternatives": ranked[:5]}


def best_candidate(candidates):
    """Compatibility tuple. Supply OCR dictionaries including confidence."""
    result = select_candidate(candidates)
    return result["plate_text"], result["selection_score"]


def strong_reading(reading):
    """Conservative compute gate, not a calibrated guarantee of correctness."""
    alternatives = reading.get("alternatives", [])
    margin = (alternatives[0]["selection_score"] - alternatives[1]["selection_score"]
              if len(alternatives) > 1 else 1.0)
    return (reading.get("status") == "accepted" and reading.get("format_score", 0) == 1.0
            and reading.get("ocr_confidence", 0) >= 0.95 and margin >= 0.15)


def choose_reading(readings, ambiguity_margin=0.08):
    """Preserve disagreements across original/restored/upscaled views."""
    ranked = sorted(readings, key=lambda r: r["selection_score"], reverse=True)
    best = dict(ranked[0])
    competing = [r for r in ranked[1:] if r["proposed_text"] and r["proposed_text"] != best["proposed_text"]]
    if competing and best["selection_score"] - competing[0]["selection_score"] < ambiguity_margin:
        best.update(plate_text="", status="uncertain")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--confidence", type=float, default=0.0, help="Measured OCR confidence; unknown stays uncertain")
    args = parser.parse_args()
    print(json.dumps(select_candidate([{"text": t, "conf": args.confidence} for t in args.candidates]), indent=2))


if __name__ == "__main__":
    main()
