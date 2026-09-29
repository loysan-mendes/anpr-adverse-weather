"""Run with python -m unittest discover -s ml/tests -v (requires numpy)."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from dataset_utils import split_entries, image_path, source_id
from evaluate_pipeline import check_independent, collect_seen, score_image, summarize, match_boxes
from image_ops import tiled_restore, merge_detections
from model_artifacts import activate_detector, resolve_detector, sha256_file, export_is_current, atomic_json
from plate_validator import select_candidate, order_regions, choose_reading, D2L
from training_config import restoration_parser, checkpoint_score, ssim_function
import prepare_yolo_dataset


def region(text, confidence=0.99, box=None):
    return {"text": text, "conf": confidence, "box": box}


class RecognitionTests(unittest.TestCase):
    def test_nonplate_text_is_unreadable(self):
        result = select_candidate([region("SUZUKI")])
        self.assertEqual(result["plate_text"], "")
        self.assertEqual(result["status"], "unreadable")

    def test_low_confidence_format_match_abstains(self):
        result = select_candidate([region("DL3CD1210", 0.2)])
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["plate_text"], "")

    def test_confidence_beats_input_order(self):
        result = select_candidate([region("MH12AB1234", 0.1), region("DL3CD1210", 0.99)])
        self.assertEqual(result["plate_text"], "DL3CD1210")

    def test_competing_strong_readings_abstain(self):
        result = select_candidate([region("MH12AB1234"), region("DL3CD1210")])
        self.assertEqual(result["status"], "uncertain")

    def test_missing_series_never_gets_perfect_acceptance(self):
        result = select_candidate([region("MP077524")])
        self.assertEqual(result["status"], "uncertain")
        self.assertLess(result["format_score"], 1)

    def test_reverse_confusion_keeps_all_alternatives(self):
        self.assertEqual(set(D2L["0"]), {"O", "Q", "D"})
        result = select_candidate([region("0D01AB1234")])
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("OD01AB1234", {r["text"] for r in result["alternatives"]})

    def test_two_rows_with_three_regions(self):
        regions = [region("1234", box=[0, 30, 60, 50]),
                   region("AB", box=[65, 0, 95, 20]), region("KA01", box=[0, 0, 60, 20])]
        self.assertEqual([r["text"] for r in order_regions(regions)], ["KA01", "AB", "1234"])
        self.assertEqual(select_candidate(regions)["plate_text"], "KA01AB1234")

    def test_missing_geometry_keeps_backend_order(self):
        regions = [region("KA01"), region("AB"), region("1234")]
        self.assertEqual(order_regions(regions), regions)

    def test_disagreement_between_image_views_abstains(self):
        readings = [select_candidate([region("KA01AB1234")]), select_candidate([region("KA01AB1235")])]
        self.assertEqual(choose_reading(readings)["plate_text"], "")

    def test_unknown_confidence_and_nonfinite_confidence_abstain(self):
        for regions in (["KA01AB1234"], [region("KA01AB1234", float("nan"))]):
            self.assertNotEqual(select_candidate(regions)["status"], "accepted")


class ImageTests(unittest.TestCase):
    def test_tiling_preserves_small_large_and_nonsquare_pixels(self):
        rng = np.random.default_rng(42)
        for shape in ((1, 1, 3), (17, 80, 3), (73, 101, 3), (128, 128, 3)):
            with self.subTest(shape=shape):
                image = rng.integers(0, 256, shape, dtype=np.uint8)
                actual = tiled_restore(image, lambda tile: tile.copy(), 32, overlap=8)
                np.testing.assert_array_equal(actual, image)

    def test_invalid_restoration_pixels_fail(self):
        with self.assertRaises(ValueError):
            tiled_restore(np.zeros((4, 4, 3)), lambda tile: tile * np.nan, 8)

    def test_original_only_detections_survive_merge(self):
        original = {"box": [0, 0, 20, 20], "confidence": 0.9}
        duplicate = {"box": [1, 1, 21, 21], "confidence": 0.8}
        other = {"box": [40, 40, 60, 60], "confidence": 0.7}
        self.assertEqual(merge_detections([original, duplicate, other]), [original, other])


class ArtifactTests(unittest.TestCase):
    def test_newest_legacy_and_explicit_and_active_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "models/yolo_runs/plate_detector/weights/best.pt"
            new = root / "models/yolo_runs/plate_detector3/weights/best.pt"
            for i, path in enumerate((old, new)):
                path.parent.mkdir(parents=True)
                path.write_bytes(str(i).encode())
                os.utime(path, (100+i, 100+i))
            with self.assertWarns(UserWarning):
                self.assertEqual(resolve_detector(ml_dir=root), new)
            self.assertEqual(resolve_detector(old, root), old)
            activate_detector(old, root)
            self.assertEqual(resolve_detector(ml_dir=root), old)
            old.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                resolve_detector(ml_dir=root)

    def test_changed_checkpoint_or_script_invalidates_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, script = Path(tmp)/"best.pt", Path(tmp)/"model_scripted.pt"
            checkpoint.write_bytes(b"checkpoint")
            script.write_bytes(b"script")
            self.assertFalse(export_is_current(checkpoint, script))
            atomic_json(script.with_suffix(".json"), {"checkpoint_sha256": sha256_file(checkpoint), "scripted_sha256": sha256_file(script)})
            self.assertTrue(export_is_current(checkpoint, script))
            checkpoint.write_bytes(b"new checkpoint")
            self.assertFalse(export_is_current(checkpoint, script))


class DatasetTests(unittest.TestCase):
    def entries(self):
        return [{"image_path": f"data/{condition}/{i}.jpg", "orig_source": "fixture",
                 "condition": condition, "severity": "none" if condition == "clear" else "mild",
                 "width": 100, "height": 100, "boxes": []}
                for i in range(10) for condition in ("clear", "rain")]

    def test_source_split_is_order_independent_and_disjoint(self):
        entries = self.entries()
        first, second = split_entries(entries), split_entries(list(reversed(entries)))
        ids = {key: {source_id(e) for e in values} for key, values in first.items()}
        for key in first:
            self.assertEqual(ids[key], {source_id(e) for e in second[key]})
        self.assertFalse(ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])

    def test_explicit_source_leakage_is_rejected(self):
        entries = self.entries()
        for i, entry in enumerate(entries):
            entry["split"] = "train" if i % 2 else "val"
        with self.assertRaises(ValueError):
            split_entries(entries)

    def test_same_content_different_source_ids_rejected(self):
        entries = self.entries()
        for i, entry in enumerate(entries):
            entry.update(split="train" if i < 10 else "val", source_sha256="same-content")
        with self.assertRaises(ValueError):
            split_entries(entries)

    def test_windows_manifest_path_is_portable(self):
        self.assertEqual(image_path({"image_path": "data\\raw\\a.jpg"}, Path("root")), Path("root/data/raw/a.jpg"))

    def test_rebuild_publishes_new_split_without_stale_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = self.entries()
            for entry in entries:
                path = image_path(entry, root)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry["image_path"].encode())
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(entries))
            with patch.object(prepare_yolo_dataset, "ML_DIR", root), contextlib.redirect_stdout(io.StringIO()):
                first = prepare_yolo_dataset.prepare_dataset(manifest, root/"yolo", .2, 0)
                second = prepare_yolo_dataset.prepare_dataset(manifest, root/"yolo", .5, 0)
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            train = json.loads((second/"train_manifest.json").read_text())
            val = json.loads((second/"val_manifest.json").read_text())
            self.assertFalse({source_id(e) for e in train} & {source_id(e) for e in val})
            self.assertEqual(len(list((second/"images/train").glob("*.jpg"))), len(train))
            self.assertIn(second.name, (root/"yolo/data.yaml").read_text())
            self.assertNotIn("path:", (root/"yolo/data.yaml").read_text())

    def test_failed_rebuild_keeps_published_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root/"yolo/data.yaml"
            target.parent.mkdir()
            target.write_text("previous build")
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps(self.entries()))
            with patch.object(prepare_yolo_dataset, "ML_DIR", root), self.assertRaises(FileNotFoundError):
                prepare_yolo_dataset.prepare_dataset(manifest, target.parent)
            self.assertEqual(target.read_text(), "previous build")


class EvaluationTests(unittest.TestCase):
    def test_misses_and_wrong_acceptances_count_in_end_to_end_metrics(self):
        box = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "text": "KA01AB1234"}
        entry = {"boxes": [box, {**box, "xmin": 20, "xmax": 30}]}
        predictions = [{"box": [0, 0, 10, 10], "plate_text": "KA01AB1234", "status": "accepted"},
                       {"box": [50, 50, 60, 60], "plate_text": "DL3CD1210", "status": "accepted"}]
        result = summarize([score_image(entry, {"plates": predictions})])
        self.assertEqual(result["end_to_end_exact_recall"], .5)
        self.assertEqual(result["accepted_reading_precision"], .5)
        self.assertEqual(result["missed_plates"], 1)
        self.assertEqual(result["extra_detections"], 1)

    def test_one_detection_cannot_match_two_ground_truth_plates(self):
        truth = [{"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}] * 2
        self.assertEqual(len(match_boxes(truth, [{"box": [0, 0, 10, 10]}])), 1)

    def test_overlap_checked_by_identity_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/"image.jpg"
            image.write_bytes(b"photo")
            entry = {"image_path": "image.jpg", "source_id": "external", "boxes": []}
            seen = {"source_ids": set(), "image_sha256": {sha256_file(image)}, "source_sha256": set()}
            with self.assertRaises(ValueError):
                check_independent([entry], seen, root)
            seen["image_sha256"].clear()
            check_independent([entry], seen, root)
            seen["source_ids"].add("external")
            with self.assertRaises(ValueError):
                check_independent([entry], seen, root)

    def test_changed_weights_reject_training_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            path.write_bytes(b"new weights")
            atomic_json(path.with_suffix(".provenance.json"), {"checkpoint_sha256": "old"})
            with self.assertRaises(ValueError):
                collect_seen([path], "unused")


class TrainingConfigTests(unittest.TestCase):
    def test_requested_ssim_cannot_silently_disappear(self):
        with patch("training_config.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "--ssim_weight 0"):
                ssim_function(.3)

    def test_explicit_ssim_opt_out_requires_no_dependency(self):
        with patch("training_config.importlib.import_module", side_effect=AssertionError("unexpected import")):
            self.assertIsNone(ssim_function(0))

    def test_readme_residual_flag_and_explicit_opt_out(self):
        parser = restoration_parser("manifest.json")
        self.assertTrue(parser.parse_args(["--condition", "haze", "--residual"]).residual)
        self.assertFalse(parser.parse_args(["--condition", "blur", "--no-residual"]).residual)

    def test_severity_regression_reduces_checkpoint_score(self):
        self.assertGreater(checkpoint_score(.9, .8), checkpoint_score(.9, .3))
        self.assertGreater(checkpoint_score(.9, .8), checkpoint_score(.95, .3))


if __name__ == "__main__":
    unittest.main()
