"""Integration of real pipeline policy with deterministic model/backend doubles."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name, dependencies):
    spec = importlib.util.spec_from_file_location("tested_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class PipelineTests(unittest.TestCase):
    def engine(self, condition="haze"):
        cv2 = types.ModuleType("cv2")
        original = np.full((60, 250, 3), 100, dtype=np.uint8)
        cv2.imread = Mock(return_value=original)
        quality = types.ModuleType("predict_quality")
        quality.load_model = Mock(return_value=(object(), {}, "cpu"))
        quality.predict = Mock(return_value={"condition": condition, "condition_confidence": .95,
                                            "severity": "severe", "severity_confidence": .95})
        unet = types.ModuleType("unet_model")
        unet.UNet = Mock()
        ocr = types.ModuleType("ocr_plate")
        ocr.recognize_plate_candidates = Mock(return_value=[{"text": "KA01AB1234", "conf": .99}])
        ultralytics = types.ModuleType("ultralytics")
        ultralytics.YOLO = Mock()
        engine = load_module("decision_engine", {"cv2": cv2, "torch": types.ModuleType("torch"),
            "ultralytics": ultralytics, "predict_quality": quality, "unet_model": unet, "ocr_plate": ocr})
        engine.resolve_detector = Mock(return_value=Path("selected.pt"))
        engine.detect_vehicles = Mock(return_value=[{"id": 1, "box": [0, 0, 250, 60],
                                                    "vehicle_type": "car", "confidence": .9}])
        engine.restore_image = Mock(return_value=np.full_like(original, 200))
        return engine, original

    def detection(self):
        coords = Mock()
        coords.cpu.return_value.numpy.return_value = np.array([5, 10, 245, 50])
        return types.SimpleNamespace(xyxy=[coords], conf=[.95])

    def test_restored_detection_failure_retains_original(self):
        engine, original = self.engine()
        detector = Mock()
        detector.predict.side_effect = [[types.SimpleNamespace(boxes=[self.detection()])], [types.SimpleNamespace(boxes=[])]]
        engine.load_yolo = Mock(return_value=detector)
        result = engine.process_image("photo.jpg", use_upscale=False)
        self.assertEqual(result["plates"][0]["plate_text"], "KA01AB1234")
        self.assertEqual(result["num_plates_detected"], 1)
        self.assertEqual(result["plates"][0]["vehicle_type"], "car")
        self.assertEqual(result["plates"][0]["vehicle_id"], 1)
        self.assertEqual(result["num_vehicles_detected"], 1)
        self.assertEqual(engine.recognize_plate_candidates.call_count, 2)
        self.assertTrue(np.all(engine.recognize_plate_candidates.call_args_list[0].args[0] == 100))
        self.assertTrue(np.all(engine.recognize_plate_candidates.call_args_list[1].args[0] == 200))

    def test_cross_view_disagreement_returns_uncertain(self):
        engine, _ = self.engine()
        detector = Mock()
        detector.predict.return_value = [types.SimpleNamespace(boxes=[self.detection()])]
        engine.load_yolo = Mock(return_value=detector)
        engine.recognize_plate_candidates.side_effect = [[{"text": "KA01AB1234", "conf": .99}],
                                                         [{"text": "KA01AB1235", "conf": .99}]]
        result = engine.process_image("photo.jpg", use_upscale=False)
        self.assertEqual(result["plates"][0]["status"], "uncertain")
        self.assertEqual(result["num_plates_accepted"], 0)

    def test_vehicle_results_survive_when_no_plate_is_detected(self):
        engine, _ = self.engine(condition="clear")
        detector = Mock()
        detector.predict.return_value = [types.SimpleNamespace(boxes=[])]
        engine.load_yolo = Mock(return_value=detector)
        result = engine.process_image("photo.jpg", use_upscale=False)
        self.assertEqual(result["plates"], [])
        self.assertEqual(result["vehicles"][0]["vehicle_type"], "car")
        self.assertEqual(result["stage_calls"]["vehicle_detector"], 1)

    def test_optional_upscaler_failure_keeps_original_ocr(self):
        engine, _ = self.engine()
        sr = types.ModuleType("super_resolve")
        sr.upscale_image = Mock(side_effect=RuntimeError("fixture unavailable"))
        with patch.dict(sys.modules, {"super_resolve": sr}), self.assertWarns(UserWarning):
            readings = engine.read_crop(np.zeros((20, 80, 3), dtype=np.uint8), "original", profile="exhaustive")
        self.assertEqual(readings[0]["plate_text"], "KA01AB1234")
        self.assertIn("processing_warning", readings[0])

    def test_quality_checkpoint_load_does_not_request_pretraining(self):
        torch = types.ModuleType("torch")
        torch.device = lambda name: name
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.load = Mock(return_value={"conditions": ["clear"], "severities": ["none"], "model_state": {}})
        nn = types.ModuleType("torch.nn")
        functional = types.ModuleType("torch.nn.functional")
        nn.functional = functional
        torchvision = types.ModuleType("torchvision")
        torchvision.transforms = types.ModuleType("torchvision.transforms")
        training = types.ModuleType("train_quality_analyzer")
        training.QualityNet = Mock()
        training.IMAGENET_MEAN, training.IMAGENET_STD = [], []
        module = load_module("predict_quality", {"cv2": types.ModuleType("cv2"), "torch": torch,
            "torch.nn": nn, "torch.nn.functional": functional, "torchvision": torchvision,
            "torchvision.transforms": torchvision.transforms, "train_quality_analyzer": training})
        module.load_model("local-checkpoint.pt")
        self.assertFalse(training.QualityNet.call_args.kwargs["pretrained"])
        training.QualityNet.return_value.load_state_dict.assert_called_once_with({})


if __name__ == "__main__":
    unittest.main()
