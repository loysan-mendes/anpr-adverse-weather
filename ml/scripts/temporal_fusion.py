"""Multi-frame temporal reading fusion and consensus voting for video ANPR.

Combines sequential plate observations from a tracked vehicle passage,
weights observations by sharpness and OCR confidence, and applies
multi-frame agreement to confirm or review plate numbers.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from plate_validator import best_template_fit, select_candidate, strong_reading



def estimate_sharpness(img_bgr: np.ndarray) -> float:
    """Estimate image sharpness using Laplacian variance.

    Higher variance corresponds to sharper character edges; lower values indicate blur.
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def character_level_consensus(readings: List[Dict], weights: List[float]) -> Tuple[str, float]:
    """Aggregate character predictions position-by-position across frames.

    Returns the aligned consensus text and average alignment confidence.
    """
    valid = [(r["proposed_text"], w, r["ocr_confidence"])
             for r, w in zip(readings, weights) if r.get("proposed_text")]
    if not valid:
        return "", 0.0

    # Group by length to align positions cleanly
    lengths = {}
    for text, w, _ in valid:
        lengths[len(text)] = lengths.get(len(text), 0.0) + w
    best_len = max(lengths, key=lengths.get)
    aligned_candidates = [(text, w, c) for text, w, c in valid if len(text) == best_len]

    consensus_chars = []
    char_confs = []
    for pos in range(best_len):
        pos_votes = {}
        pos_confs = {}
        for text, w, conf in aligned_candidates:
            char = text[pos]
            pos_votes[char] = pos_votes.get(char, 0.0) + w
            pos_confs.setdefault(char, []).append(conf)
        top_char = max(pos_votes, key=pos_votes.get)
        consensus_chars.append(top_char)
        char_confs.append(float(np.mean(pos_confs[top_char])))

    return "".join(consensus_chars), float(np.mean(char_confs)) if char_confs else 0.0


def fuse_observations(
    observations: List[Dict],
    ambiguity_margin: float = 0.08,
    min_agreement_count: int = 2,
    acceptance_min_confidence: float = 0.88,
) -> Dict:
    """Fuse multiple sequential plate readings from a tracked vehicle track.

    Args:
        observations: List of observation dicts, each containing:
            - reading: dict from select_candidate/read_crop
            - sharpness: float
            - frame_idx: int
            - timestamp_sec: float (optional)
        ambiguity_margin: Margin between top candidate and competing readings.
        min_agreement_count: Number of agreeing frames required for temporal confirmation.
        acceptance_min_confidence: Threshold for automatically accepting consensus text.

    Returns:
        Structured passage consensus result with plate_text, status, and evidence.
    """
    if not observations:
        return {
            "plate_text": "",
            "proposed_text": "",
            "status": "unreadable",
            "review_reason": "no_observations",
            "fused_confidence": 0.0,
            "format_score": 0.0,
            "agreement_count": 0,
            "total_observations": 0,
            "sharpest_frame_idx": None,
            "observations": [],
            "alternatives": [],
        }

    valid_obs = []
    weights = []
    sharpest_idx = None
    max_sharpness = -1.0

    for obs in observations:
        reading = obs.get("reading", {})
        sharpness = float(obs.get("sharpness", 0.0))
        frame_idx = obs.get("frame_idx")
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            sharpest_idx = frame_idx

        proposed = reading.get("proposed_text", "")
        ocr_conf = float(reading.get("ocr_confidence", 0.0))
        fmt_score = float(reading.get("format_score", 0.0))

        # Weight combines OCR confidence, format adherence, and focus quality
        # Logarithmic scaling on sharpness prevents single sharp frames from drowning consistent runs
        sharpness_factor = float(np.log1p(max(0.0, sharpness)))
        w = max(0.01, ocr_conf * (1.0 + 0.5 * fmt_score) * (1.0 + 0.1 * sharpness_factor))

        valid_obs.append(obs)
        weights.append(w)

    readings = [obs["reading"] for obs in valid_obs]

    # 1. Text-level vote tally
    candidate_scores = {}
    candidate_counts = {}
    candidate_max_conf = {}
    candidate_confs = {}

    for obs, w in zip(valid_obs, weights):
        reading = obs["reading"]
        text = reading.get("proposed_text", "")
        if not text:
            continue
        conf = float(reading.get("ocr_confidence", 0.0))
        candidate_scores[text] = candidate_scores.get(text, 0.0) + w
        candidate_counts[text] = candidate_counts.get(text, 0) + 1
        candidate_max_conf[text] = max(candidate_max_conf.get(text, 0.0), conf)
        candidate_confs.setdefault(text, []).append(conf)

    if not candidate_scores:
        return {
            "plate_text": "",
            "proposed_text": "",
            "status": "unreadable",
            "review_reason": "unreadable_in_all_frames",
            "fused_confidence": 0.0,
            "format_score": 0.0,
            "agreement_count": 0,
            "total_observations": len(observations),
            "sharpest_frame_idx": sharpest_idx,
            "observations": valid_obs,
            "alternatives": [],
        }

    # Sort texts by total vote score
    ranked_texts = sorted(candidate_scores.keys(), key=lambda t: candidate_scores[t], reverse=True)
    best_text = ranked_texts[0]
    agreement_count = candidate_counts[best_text]
    mean_conf = float(np.mean(candidate_confs[best_text]))

    # Compute format score on consensus candidate
    parsed = best_template_fit(best_text)
    format_score = float(parsed[1]) if parsed else 0.0

    # 2. Multi-frame confidence promotion
    # Independent confirmation across multiple frames builds confidence beyond single-frame noise
    agreement_bonus = 0.03 * min(4, agreement_count - 1) if agreement_count >= min_agreement_count else 0.0
    fused_confidence = min(0.99, mean_conf + agreement_bonus)

    # 3. Competing reading ambiguity check
    competing = [t for t in ranked_texts[1:] if t != best_text]
    conflicts = []
    if competing:
        top_score = candidate_scores[best_text]
        runner_up = competing[0]
        runner_score = candidate_scores[runner_up]
        normalized_gap = (top_score - runner_score) / max(1e-6, top_score)
        if normalized_gap < ambiguity_margin and candidate_counts[runner_up] >= 2:
            conflicts.append(runner_up)

    # Character-level consensus check only for small character confusions (e.g. O/0 or B/8, distance <= 2)
    if (
        conflicts
        and len(best_text) == len(conflicts[0])
        and sum(c1 != c2 for c1, c2 in zip(best_text, conflicts[0])) <= 2
    ):
        char_text, char_conf = character_level_consensus(readings, weights)
        char_parsed = best_template_fit(char_text)
        if char_parsed and float(char_parsed[1]) > format_score:
            best_text = char_text
            format_score = float(char_parsed[1])
            conflicts = []

    # 4. Acceptance Decision Gate
    accepted = (
        not bool(conflicts)
        and format_score >= 0.85
        and fused_confidence >= acceptance_min_confidence
        and (agreement_count >= min_agreement_count or fused_confidence >= 0.92)
    )

    review_reason = None
    if conflicts:
        review_reason = "conflicting_temporal_readings"
    elif format_score < 0.85:
        review_reason = "unsupported_layout"
    elif fused_confidence < acceptance_min_confidence:
        review_reason = "low_temporal_confidence"
    elif agreement_count < min_agreement_count and fused_confidence < 0.92:
        review_reason = "insufficient_frame_agreement"

    status = "accepted" if accepted else "uncertain"
    plate_text = best_text if accepted else ""

    alternatives = [
        {
            "text": t,
            "votes": round(candidate_scores[t], 2),
            "frame_count": candidate_counts[t],
            "max_confidence": round(candidate_max_conf[t], 4),
        }
        for t in ranked_texts[:5]
    ]

    return {
        "plate_text": plate_text,
        "proposed_text": best_text,
        "status": status,
        "review_reason": review_reason,
        "fused_confidence": round(fused_confidence, 4),
        "format_score": round(format_score, 4),
        "agreement_count": agreement_count,
        "total_observations": len(observations),
        "sharpest_frame_idx": sharpest_idx,
        "conflicting_texts": conflicts,
        "alternatives": alternatives,
        "observations": valid_obs,
    }
