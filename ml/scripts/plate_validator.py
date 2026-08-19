"""
Phase 8: Plate Validation -- clean up raw OCR output using known Indian
plate format structure.

Two things this fixes, both visible in Phase 7's evaluation:

1. Contamination: OCR often detects extra nearby text (dealer stickers,
   brand badges, hologram serials) as SEPARATE regions alongside the real
   plate number, and naive concatenation glues everything together --
   e.g. "SUZUKIDL3CD1210" instead of "DL3CD1210". Fix: score each
   INDIVIDUALLY DETECTED region (and adjacent pairs, in case the true
   plate was itself split across two regions) against known Indian plate
   templates, keep the best-scoring one. We deliberately do NOT slice
   arbitrary substrings out of the flattened string -- that creates
   nonsense matches that happen to fit a template by coincidence.

2. Common character misreads: OCR confuses visually similar
   letters/digits (O/0, I/1, S/5, B/8, G/6, Z/2). Fix: when a region
   matches a template's length but has a class mismatch at one position,
   try that character's known confusion-pair substitute. Fewer required
   substitutions = higher score, so a clean exact fit always beats a
   heavily-coerced one.

This is a plausibility scorer to pick the best candidate among several
noisy OCR readings -- NOT a legal registry lookup (no real state-code
list, no valid-district-range checking). Known remaining limitation: it
sorts/joins candidate regions left-to-right and doesn't handle two-row
plate layouts (series letters stacked above/below the number), so a
scrambled two-row read won't be fixed by this alone.

Usage (standalone test):
    python ml/scripts/plate_validator.py --candidates "SUZUKI" "DL3CD1210"
"""

import argparse
import re

# letter -> digit, for characters that look alike (used when a template
# position expects a digit but OCR read a similar-looking letter)
L2D = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0"}
# digit -> letter, the reverse direction
D2L = {v: k for k, v in L2D.items()}

# Indian plate structure templates (L=letter position, D=digit position).
# Covers common real-world variants: 1- or 2-digit RTO code, 1- or
# 2-letter series, 3- or 4-digit number.
TEMPLATES = [
    "LLDDLDDDD",   # e.g. KL 35 F 4337
    "LLDDLLDDDD",  # e.g. WB 42 AX 7446
    "LLDLDDDD",    # e.g. DL 3 C 1210
    "LLDLLDDDD",   # e.g. DL 3 CD 1210
    "LLDDDDDD",    # e.g. KL 49 8262 (no series letter)
    "LLDDLDDD",    # e.g. KL 34 A 465 (3-digit number, older format)
]


def clean(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def coerce_with_cost(text, template):
    """Try to fit `text` to `template`'s letter/digit class per position.
    Returns (coerced_string, num_substitutions_needed) or (None, None) if
    some position can't be satisfied even with confusion-substitution."""
    if len(text) != len(template):
        return None, None
    out, cost = [], 0
    for ch, t in zip(text, template):
        if t == "L":
            if ch.isalpha():
                out.append(ch)
            elif ch in D2L:
                out.append(D2L[ch])
                cost += 1
            else:
                return None, None
        else:
            if ch.isdigit():
                out.append(ch)
            elif ch in L2D:
                out.append(L2D[ch])
                cost += 1
            else:
                return None, None
    return "".join(out), cost


def best_template_fit(text):
    """Best (coerced_text, score) across all templates for one string.
    score: 1.0 for an exact fit (no substitutions needed), decreasing
    slightly per substitution required; None if no template matches this
    length at all."""
    best = None
    for t in TEMPLATES:
        coerced, cost = coerce_with_cost(text, t)
        if coerced is None:
            continue
        s = max(0.5, 1.0 - 0.08 * cost)
        if best is None or s > best[1]:
            best = (coerced, s)
    return best


def partial_score(text):
    """Fallback plausibility score for strings that don't fit any
    template length -- some credit for looking plate-ish, always below
    a real template fit."""
    partial = 0.0
    if len(text) >= 2 and text[:2].isalpha():
        partial += 0.15
    digit_frac = sum(c.isdigit() for c in text) / max(len(text), 1)
    partial += 0.1 * digit_frac
    if 7 <= len(text) <= 10:
        partial += 0.1
    return partial


def best_candidate(candidates):
    """candidates: list of raw OCR strings from individually detected text
    regions (NOT pre-joined). Scores each region alone, and each adjacent
    pair joined together (covers a true plate split across 2 regions).
    Returns (best_text, score)."""
    cleaned = [clean(c) for c in candidates if clean(c)]
    if not cleaned:
        return "", 0.0

    pool = list(cleaned)
    for i in range(len(cleaned) - 1):
        pool.append(cleaned[i] + cleaned[i + 1])
        pool.append(cleaned[i + 1] + cleaned[i])  # handles stacked/two-row layouts
    pool.append("".join(cleaned))  # full join, as a last-resort candidate

    best_text, best_score = pool[0], -1.0
    for cand in pool:
        fit = best_template_fit(cand)
        s = fit[1] if fit else partial_score(cand)
        text = fit[0] if fit else cand
        if s > best_score:
            best_text, best_score = text, s

    return best_text, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True,
                         help="one or more individually-detected OCR text regions")
    args = parser.parse_args()

    text, s = best_candidate(args.candidates)
    print(f"Input regions: {args.candidates}")
    print(f"Best guess:    {text}")
    print(f"Score:         {s:.2f}")


if __name__ == "__main__":
    main()