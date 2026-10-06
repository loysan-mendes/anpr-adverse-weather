import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acceptance_policy import apply_acceptance_gate, validate_acceptance_confidence
import test_pipeline as fixtures


class AcceptancePolicyTests(unittest.TestCase):
    def reading(self, confidence=.85, status="accepted"):
        return {"plate_text": "KA01AB1234" if status == "accepted" else "",
                "proposed_text": "KA01AB1234", "status": status, "ocr_confidence": confidence,
                "raw_text": "KA01AB1234", "literal_evidence": [{"text": "KA01AB1234"}],
                "alternatives": [{"text": "KA01AB1234", "selection_score": confidence}]}

    def test_gate_preserves_proposal_evidence_and_input(self):
        original = self.reading()
        snapshot = copy.deepcopy(original)
        result = apply_acceptance_gate(original, .9)
        self.assertEqual(original, snapshot)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["plate_text"], "")
        self.assertEqual(result["proposed_text"], "KA01AB1234")
        self.assertEqual(result["review_reason"], "low_final_ocr_confidence")
        for field in ("raw_text", "literal_evidence", "alternatives", "ocr_confidence"):
            self.assertEqual(result[field], original[field])

    def test_threshold_boundary_and_baseline(self):
        self.assertEqual(apply_acceptance_gate(self.reading(.9), .9)["status"], "accepted")
        self.assertEqual(apply_acceptance_gate(self.reading(.899999), .9)["status"], "uncertain")
        self.assertEqual(apply_acceptance_gate(self.reading(.8), .8)["status"], "accepted")

    def test_never_promotes_uncertain_or_unreadable(self):
        for status in ("uncertain", "unreadable"):
            original = {**self.reading(.99, status), "review_reason": "conflicting_literal_readings"}
            self.assertEqual(apply_acceptance_gate(original, .8), original)

    def test_invalid_confidence_cannot_be_accepted(self):
        for value in (None, float("nan"), float("inf"), -.1, 1.1, "0.99", True):
            with self.subTest(value=value):
                self.assertEqual(apply_acceptance_gate(self.reading(value), .9)["status"], "uncertain")

    def test_invalid_thresholds_rejected(self):
        for value in (0, -1, 1.1, float("nan"), float("inf"), None, "0.9", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_acceptance_confidence(value)

    def test_missing_proposal_retains_selected_text_and_is_idempotent(self):
        reading = self.reading()
        del reading["proposed_text"]
        result = apply_acceptance_gate(reading, .9)
        self.assertEqual(result["proposed_text"], reading["plate_text"])
        self.assertEqual(apply_acceptance_gate(result, .8), result)

    def test_pipeline_gate_does_not_change_routing_or_vehicle_association(self):
        outputs = []
        for threshold in (.8, .9):
            fixture = fixtures.PipelineTests()
            engine, _ = fixture.engine()
            detector = Mock()
            detector.predict.return_value = [SimpleNamespace(boxes=[fixture.detection()])]
            engine.load_yolo = Mock(return_value=detector)
            engine.recognize_plate_candidates.return_value = [{"text": "KA01AB1234", "conf": .85}]
            result = engine.process_image("photo.jpg", use_upscale=False, acceptance_min_confidence=threshold)
            self.assertEqual(result["plates"][0]["vehicle_id"], 1)
            self.assertEqual(engine.recognize_plate_candidates.call_count, 2)
            outputs.append(result)
        self.assertEqual(outputs[0]["num_plates_accepted"], 1)
        self.assertEqual(outputs[1]["num_plates_accepted"], 0)
        self.assertEqual(outputs[0]["stage_calls"], outputs[1]["stage_calls"])
        for key in ("readings", "box", "detection_confidence", "proposed_text", "vehicle_match_status"):
            self.assertEqual(outputs[0]["plates"][0][key], outputs[1]["plates"][0][key])

    def test_invalid_gate_fails_before_loading_images_or_models(self):
        engine, _ = fixtures.PipelineTests().engine()
        with self.assertRaises(ValueError):
            engine.process_image("photo.jpg", acceptance_min_confidence=float("nan"))
        engine.cv2.imread.assert_not_called()
        engine.recognize_plate_candidates.assert_not_called()
