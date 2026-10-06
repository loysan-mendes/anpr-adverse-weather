"""Final review gate; never changes OCR evidence or inference routing."""
import math
from numbers import Real

# Development-selected prototype policy, not a calibrated correctness probability.
DEFAULT_ACCEPTANCE_MIN_CONFIDENCE = 0.90


def validate_acceptance_confidence(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Final OCR acceptance confidence must be finite and in (0, 1]")
    return float(value)


def apply_acceptance_gate(reading, min_confidence=DEFAULT_ACCEPTANCE_MIN_CONFIDENCE):
    """Move weak accepted text to review; never promote or reinterpret text."""
    threshold = validate_acceptance_confidence(min_confidence)
    result = dict(reading)
    if result.get("status") != "accepted":
        return result
    confidence = result.get("ocr_confidence")
    valid = (not isinstance(confidence, bool) and isinstance(confidence, Real)
             and math.isfinite(confidence) and 0 <= confidence <= 1)
    if not valid or confidence < threshold:
        result.update(plate_text="", proposed_text=result.get("proposed_text") or result.get("plate_text", ""),
                      status="uncertain", review_reason="low_final_ocr_confidence",
                      acceptance_min_confidence=threshold)
    return result
