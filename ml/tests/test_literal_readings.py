"""Regression cases found in the supplied-target OCR development diagnostic."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plate_validator import choose_reading, select_candidate, strong_reading


def reading(text, confidence=.99):
    return select_candidate([{"text": text, "conf": confidence}])


class LiteralReadingTests(unittest.TestCase):
    def test_observed_no_series_plate_is_not_rewritten_to_a_series(self):
        result = reading("PY036993", .9465173)
        self.assertEqual(result["proposed_text"], "PY036993")
        self.assertEqual(result["raw_text"], "PY036993")
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["plate_text"], "")
        self.assertFalse(strong_reading(result))

    def test_split_no_series_plate_preserves_all_digits(self):
        result = select_candidate([{"text": "PY03", "conf": .95}, {"text": "6993", "conf": .94}])
        self.assertEqual(result["proposed_text"], "PY036993")
        self.assertEqual(result["status"], "uncertain")

    def test_other_literal_review_layouts_are_preserved(self):
        for text in ("MP077524", "KL01CC50", "TN02BL9", "DL3CAY2231"):
            with self.subTest(text=text):
                result = reading(text)
                self.assertEqual(result["proposed_text"], text)
                self.assertEqual(result["status"], "uncertain")

    def test_confident_literal_conflict_is_not_hidden_by_format_scores(self):
        result = choose_reading([reading("PY03G993", .96), reading("PY036993", .95)])
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["plate_text"], "")
        self.assertEqual(result["review_reason"], "conflicting_literal_readings")
        self.assertIn("PY036993", result["conflicting_texts"])

    def test_regions_with_conflicting_layouts_also_require_review(self):
        result = select_candidate([{"text": "PY03G993", "conf": .96}, {"text": "PY036993", "conf": .95}])
        self.assertEqual(result["plate_text"], "")
        self.assertEqual(result["status"], "uncertain")

    def test_low_confidence_disagreement_does_not_veto_a_strong_literal(self):
        result = choose_reading([reading("PY03G993", .99), reading("PY036993", .4)])
        self.assertEqual(result["plate_text"], "PY03G993")

    def test_valid_one_digit_district_and_split_fragments_still_work(self):
        self.assertEqual(reading("DL3CD1210")["plate_text"], "DL3CD1210")
        result = select_candidate([{"text": text, "conf": .99} for text in ("KA01AB", "123", "4")])
        self.assertEqual(result["plate_text"], "KA01AB1234")

    def test_correction_of_implausible_ocr_remains_possible(self):
        result = reading("KAO1AB1234")
        self.assertEqual(result["proposed_text"], "KA01AB1234")
        self.assertEqual(result["raw_text"], "KAO1AB1234")
        self.assertFalse(result["literal_layout"])
        self.assertFalse(strong_reading(result))


if __name__ == "__main__":
    unittest.main()
