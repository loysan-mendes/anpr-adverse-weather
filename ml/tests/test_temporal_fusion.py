"""Unit tests for multi-frame temporal reading fusion and video ANPR consensus."""

import unittest
from pathlib import Path
import sys
import numpy as np
import cv2

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from temporal_fusion import estimate_sharpness, character_level_consensus, fuse_observations


class TemporalFusionTests(unittest.TestCase):
    def test_sharpness_differentiates_blur_from_crisp(self):
        crisp = np.zeros((40, 80, 3), dtype=np.uint8)
        # Sharp high-frequency pattern
        crisp[::2, ::2] = 255
        blurred = cv2.GaussianBlur(crisp, (7, 7), 2.0)

        sharp_val = estimate_sharpness(crisp)
        blur_val = estimate_sharpness(blurred)
        self.assertGreater(sharp_val, blur_val)
        self.assertEqual(estimate_sharpness(None), 0.0)

    def test_multi_frame_agreement_recovers_blurred_typo(self):
        # 3 observations of the same vehicle:
        # Frame 1 (crisp): KA01AB1234 (conf 0.94)
        # Frame 2 (blur typo): KA01A81234 (conf 0.70)
        # Frame 3 (crisp): KA01AB1234 (conf 0.95)
        obs = [
            {
                "frame_idx": 10,
                "sharpness": 150.0,
                "reading": {
                    "plate_text": "KA01AB1234",
                    "proposed_text": "KA01AB1234",
                    "ocr_confidence": 0.94,
                    "format_score": 1.0,
                    "status": "accepted",
                },
            },
            {
                "frame_idx": 12,
                "sharpness": 30.0,
                "reading": {
                    "plate_text": "",
                    "proposed_text": "KA01A81234",
                    "ocr_confidence": 0.70,
                    "format_score": 0.85,
                    "status": "uncertain",
                },
            },
            {
                "frame_idx": 14,
                "sharpness": 180.0,
                "reading": {
                    "plate_text": "KA01AB1234",
                    "proposed_text": "KA01AB1234",
                    "ocr_confidence": 0.95,
                    "format_score": 1.0,
                    "status": "accepted",
                },
            },
        ]

        result = fuse_observations(obs)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["plate_text"], "KA01AB1234")
        self.assertEqual(result["agreement_count"], 2)
        self.assertEqual(result["sharpest_frame_idx"], 14)
        self.assertGreaterEqual(result["fused_confidence"], 0.94)

    def test_conflicting_readings_trigger_uncertain_review(self):
        # 2 frames strongly vote for KA01AB1234, 2 frames strongly vote for MH12CD5678
        obs = [
            {
                "frame_idx": 1,
                "sharpness": 100.0,
                "reading": {"proposed_text": "KA01AB1234", "ocr_confidence": 0.90, "format_score": 1.0},
            },
            {
                "frame_idx": 2,
                "sharpness": 100.0,
                "reading": {"proposed_text": "KA01AB1234", "ocr_confidence": 0.90, "format_score": 1.0},
            },
            {
                "frame_idx": 3,
                "sharpness": 100.0,
                "reading": {"proposed_text": "MH12CD5678", "ocr_confidence": 0.90, "format_score": 1.0},
            },
            {
                "frame_idx": 4,
                "sharpness": 100.0,
                "reading": {"proposed_text": "MH12CD5678", "ocr_confidence": 0.90, "format_score": 1.0},
            },
        ]

        result = fuse_observations(obs)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["review_reason"], "conflicting_temporal_readings")
        self.assertEqual(result["plate_text"], "")

    def test_character_level_consensus_alignment(self):
        readings = [
            {"proposed_text": "DL01AB1234", "ocr_confidence": 0.90},
            {"proposed_text": "DL01A81234", "ocr_confidence": 0.85},
            {"proposed_text": "DL01AB1234", "ocr_confidence": 0.92},
        ]
        weights = [1.0, 0.8, 1.2]
        consensus_text, avg_conf = character_level_consensus(readings, weights)
        self.assertEqual(consensus_text, "DL01AB1234")
        self.assertGreater(avg_conf, 0.85)

    def test_empty_observations(self):
        res = fuse_observations([])
        self.assertEqual(res["status"], "unreadable")
        self.assertEqual(res["plate_text"], "")
        self.assertEqual(res["total_observations"], 0)


if __name__ == "__main__":
    unittest.main()
