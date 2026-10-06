"""Conservative OCR selection. Scores are heuristics, not correctness probabilities."""
import argparse
import itertools
import json
import math
import re

L2D = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0", "D": "0", "T": "7"}
D2L = {digit: tuple(k for k, v in L2D.items() if v == digit) for digit in set(L2D.values())}
TEMPLATES = ["LLDDLDDDD", "LLDDLLDDDD", "LLDLDDDD", "LLDLLDDDD", "LLDDDDDD", "LLDDLDDD"]
# Additional layouts observed in the supplied corpus remain review-only until
# independently calibrated. Match their literal text; do not invent padding.
REVIEW_LAYOUT = re.compile(r"^[A-Z]{2}(?:\d{1,2}[A-Z]{1,3}\d{1,4}|\d{3,6})$")


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
    if REVIEW_LAYOUT.fullmatch(text):
        fits[text] = max(fits.get(text, 0.0), 0.70)
        # A plausible literal is evidence, even when its layout needs review.
        # Do not replace its digits with letters to earn a higher format score.
        for proposed in fits:
            if proposed != text:
                fits[proposed] = min(fits[proposed], max(0.0, fits[text] - 0.08))
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


def conflicting_literals(best, evidence, min_confidence=0.80, ambiguity_margin=0.08):
    """Compare OCR confidence before layout preference can hide a disagreement."""
    proposed = best.get("proposed_text", best.get("text", ""))
    return sorted({item["text"] for item in evidence
                   if item["text"] != proposed
                   # Contiguous windows also produce fragments of a full read.
                   # Those are not independent competing registrations.
                   and item["text"] not in proposed and proposed not in item["text"]
                   and item["ocr_confidence"] >= min_confidence
                   and best["ocr_confidence"] - item["ocr_confidence"] < ambiguity_margin
                   and item.get("omission_penalty", 0.0) == 0.0})


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
    hypotheses, literals = {}, {}
    for start in range(len(regions)):
        for end in range(start+1, len(regions)+1):
            parts = regions[start:end]
            text = "".join(r["text"] for r in parts)
            if len(text) > 14:
                break
            confidence = min(r["conf"] for r in parts)
            omitted = regions[:start] + regions[end:]
            penalty = 0.08 * sum(len(r["text"]) <= 4 for r in omitted)
            if REVIEW_LAYOUT.fullmatch(text):
                evidence = {"text": text, "ocr_confidence": confidence, "omission_penalty": penalty}
                if text not in literals or confidence-penalty > literals[text]["ocr_confidence"]-literals[text]["omission_penalty"]:
                    literals[text] = evidence
            for proposed, format_score in template_fits(text).items():
                score = max(0.0, confidence * format_score - penalty)
                item = {"text": proposed, "raw_text": text,
                        "literal_layout": proposed == text and bool(REVIEW_LAYOUT.fullmatch(text)),
                        "format_score": format_score, "ocr_confidence": confidence, "selection_score": score}
                if proposed not in hypotheses or score > hypotheses[proposed]["selection_score"]:
                    hypotheses[proposed] = item
    ranked = sorted(hypotheses.values(), key=lambda r: (-r["selection_score"], r["text"]))
    if not ranked:
        # Preserve unusual literal OCR evidence for review. A plausible string
        # is not proof of a registration, so this path cannot auto-accept it.
        literal = "".join(region["text"] for region in regions)
        if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{3,12}", literal) and re.search(r"\d", literal):
            confidence = min(region["conf"] for region in regions)
            candidate = {"text": literal, "format_score": 0.0,
                         "ocr_confidence": confidence, "selection_score": confidence * 0.4}
            return {"plate_text": "", "proposed_text": literal, "status": "uncertain",
                    "review_reason": "unsupported_layout", **{k: candidate[k] for k in
                    ("format_score", "ocr_confidence", "selection_score")}, "alternatives": [candidate]}
        return {"plate_text": "", "proposed_text": "", "status": "unreadable", "format_score": 0.0,
                "ocr_confidence": 0.0, "selection_score": 0.0, "alternatives": []}
    best = ranked[0]
    ambiguous = len(ranked) > 1 and best["selection_score"] - ranked[1]["selection_score"] < ambiguity_margin
    conflicts = conflicting_literals(best, literals.values(), min_confidence, ambiguity_margin)
    ambiguous = ambiguous or bool(conflicts)
    accepted = (not ambiguous and best["ocr_confidence"] >= min_confidence
                and best["selection_score"] >= min_score and best["format_score"] >= 0.84)
    result = {"plate_text": best["text"] if accepted else "", "proposed_text": best["text"],
            "status": "accepted" if accepted else "uncertain",
            **{k: best[k] for k in ("format_score", "ocr_confidence", "selection_score", "raw_text", "literal_layout")},
            "literal_evidence": list(literals.values()), "alternatives": ranked[:5]}
    if conflicts:
        result.update(review_reason="conflicting_literal_readings", conflicting_texts=conflicts)
    elif ambiguous:
        result["review_reason"] = "ambiguous_interpretations"
    elif not accepted:
        result["review_reason"] = "reading_requires_review"
    return result


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
        best.update(plate_text="", status="uncertain", review_reason="conflicting_image_views")
    evidence = [item for reading in readings for item in reading.get("literal_evidence", [])]
    conflicts = conflicting_literals(best, evidence, ambiguity_margin=ambiguity_margin)
    if conflicts:
        best.update(plate_text="", status="uncertain", review_reason="conflicting_literal_readings",
                    conflicting_texts=conflicts)
    if evidence:
        best["literal_evidence"] = evidence
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--confidence", type=float, default=0.0, help="Measured OCR confidence; unknown stays uncertain")
    args = parser.parse_args()
    print(json.dumps(select_candidate([{"text": t, "conf": args.confidence} for t in args.candidates]), indent=2))


if __name__ == "__main__":
    main()
