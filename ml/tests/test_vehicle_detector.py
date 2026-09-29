import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vehicle_detector import associate_plates, detect_vehicles, load_vehicle_model


class VehicleTests(unittest.TestCase):
    def vehicle(self, ident=1, kind="car", box=None, confidence=.9):
        return {"id": ident, "vehicle_type": kind, "box": box or [0, 0, 100, 100], "confidence": confidence}

    def test_multiple_vehicles_match_by_position(self):
        plates = [{"box": [20, 70, 40, 80]}, {"box": [120, 70, 140, 80]}]
        associate_plates(plates, [self.vehicle(), self.vehicle(2, "truck", [100, 0, 200, 100])])
        self.assertEqual([p["vehicle_type"] for p in plates], ["car", "truck"])

    def test_abstains_on_overlap_weak_detection_and_missing_vehicle(self):
        for vehicles, expected in [([], "unmatched"), ([self.vehicle(confidence=.4)], "low_confidence"),
                                   ([self.vehicle(), self.vehicle(2, "bus")], "ambiguous")]:
            with self.subTest(expected=expected):
                plates = [{"box": [20, 70, 40, 80]}]
                associate_plates(plates, vehicles)
                self.assertEqual(plates[0]["vehicle_match_status"], expected)
                self.assertEqual(plates[0]["vehicle_type"], "unknown")
                self.assertIsNone(plates[0]["vehicle_id"])

    def test_two_plates_cannot_claim_same_vehicle(self):
        plates = [{"box": [20, 70, 40, 80]}, {"box": [50, 70, 70, 80]}]
        associate_plates(plates, [self.vehicle()])
        self.assertTrue(all(p["vehicle_match_status"] == "ambiguous" for p in plates))

    def test_outside_and_degenerate_plate_boxes(self):
        for box in ([90, 70, 110, 80], [20, 20, 20, 30]):
            plates = [{"box": box}]
            associate_plates(plates, [self.vehicle()])
            self.assertEqual(plates[0]["vehicle_type"], "unknown")

    def test_detector_filters_classes_and_serializes_boxes(self):
        boxes = []
        for cls in [2, 3, 5, 7, 0]:
            coords = Mock()
            coords.cpu.return_value.numpy.return_value = np.array([1, 2, 100, 200])
            boxes.append(SimpleNamespace(cls=[cls], conf=[.8], xyxy=[coords]))
        model = Mock()
        model.names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 0: 'person'}
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        with patch('vehicle_detector.load_vehicle_model', return_value=model):
            vehicles = detect_vehicles(np.zeros((200, 100, 3), dtype=np.uint8))
        self.assertEqual([v["vehicle_type"] for v in vehicles], ["car", "motorcycle", "bus", "truck"])
        self.assertEqual(vehicles[0]["box"], [1, 2, 100, 200])
        self.assertEqual(model.predict.call_args.kwargs["classes"], [2, 3, 5, 7])

    def test_missing_weights_fail_without_downloading(self):
        with self.assertRaisesRegex(FileNotFoundError, "--download"):
            load_vehicle_model(Path(__file__).parent / "absent-vehicle.pt")

    def test_other_vehicle_is_not_reported_as_supported_class(self):
        plates = [{'box': [20, 70, 40, 80]}]
        associate_plates(plates, [self.vehicle(kind='unknown')])
        self.assertEqual(plates[0]['vehicle_match_status'], 'unsupported')
        self.assertIsNone(plates[0]['vehicle_id'])

    def test_custom_class_ids_are_read_from_checkpoint(self):
        coords = Mock()
        coords.cpu.return_value.numpy.return_value = np.array([1, 2, 100, 200])
        model = Mock()
        model.names = {0:'car',1:'motorcycle',2:'bus',3:'truck',4:'other'}
        model.predict.return_value = [SimpleNamespace(boxes=[SimpleNamespace(cls=[4],conf=[.9],xyxy=[coords])])]
        with patch('vehicle_detector.load_vehicle_model',return_value=model):
            vehicles = detect_vehicles(np.zeros((200,100,3),dtype=np.uint8))
        self.assertEqual(vehicles[0]['vehicle_type'],'unknown')
        self.assertEqual(model.predict.call_args.kwargs['classes'],[0,1,2,3,4])


if __name__ == '__main__':
    unittest.main()
