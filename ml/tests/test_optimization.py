import sys
import random
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from image_ops import tile_starts, tiled_restore, aligned_patch, padded_box
from plate_validator import select_candidate, strong_reading
from model_artifacts import cached_sha256
import test_pipeline as pipeline_fixtures


class OptimizationTests(unittest.TestCase):
    def test_tile_grid_has_no_redundant_edge_tile(self):
        self.assertEqual(tile_starts(512, 512, 32), [0])
        self.assertEqual(tile_starts(1024, 512, 32), [0, 480, 512])
        self.assertEqual(tile_starts(256, 512, 32), [0])

    def test_batched_restoration_preserves_pixels(self):
        image = np.random.default_rng(1).integers(0, 256, (75, 90, 3), dtype=np.uint8)
        callback = Mock(side_effect=lambda batch: batch)
        output = tiled_restore(image, None, 32, predict_batch=callback, batch_size=3)
        np.testing.assert_array_equal(output, image)
        self.assertEqual(callback.call_count, 4)  # 3 x 4 tiles in batches of three

    def test_native_patches_stay_aligned_and_keep_stroke_values(self):
        image = np.arange(80*100*3, dtype=np.uint16).reshape(80, 100, 3) % 200
        boxes = [{"xmin": 50, "ymin": 40, "xmax": 70, "ymax": 60}]
        first, second = aligned_patch(image, image+1, 32, boxes, random.Random(4))
        np.testing.assert_array_equal(second, first+1)
        self.assertEqual(first.shape, (32, 32, 3))
        first, second = aligned_patch(image[:8, :9], image[:8, :9], 32)
        np.testing.assert_array_equal(first[:8, :9], image[:8, :9])

    def test_crop_context_is_clipped_to_image_bounds(self):
        self.assertEqual(padded_box([0, 0, 100, 50], 100, 50), [0, 0, 100, 50])

    def test_fast_exit_rejects_corrected_or_weak_text(self):
        self.assertTrue(strong_reading(select_candidate([{"text": "KA01AB1234", "conf": .99}])))
        self.assertFalse(strong_reading(select_candidate([{"text": "KAO1AB1234", "conf": .99}])))
        self.assertFalse(strong_reading(select_candidate([{"text": "KA01AB1234", "conf": .85}])))

    def test_digest_cache_invalidates_after_file_change(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights"
            path.write_bytes(b"old")
            first = cached_sha256(path)
            path.write_bytes(b"new contents")
            self.assertNotEqual(first, cached_sha256(path))

    def test_balanced_mode_skips_optional_work_on_strong_nonsevere_read(self):
        fixture = pipeline_fixtures.PipelineTests()
        engine, _ = fixture.engine()
        engine.predict_quality.return_value.update(severity="mild")
        detector = Mock()
        from types import SimpleNamespace
        detector.predict.return_value = [SimpleNamespace(boxes=[fixture.detection()])]
        engine.load_yolo = Mock(return_value=detector)
        result = engine.process_image("photo.jpg", use_upscale=False)
        engine.restore_image.assert_not_called()
        self.assertEqual(result["stage_calls"]["ocr"], 1)
        self.assertEqual(result["stage_calls"]["detector"], 1)

    def test_exhaustive_mode_keeps_weather_restoration(self):
        fixture = pipeline_fixtures.PipelineTests()
        engine, _ = fixture.engine()
        engine.predict_quality.return_value.update(severity="mild")
        detector = Mock()
        from types import SimpleNamespace
        detector.predict.return_value = [SimpleNamespace(boxes=[fixture.detection()])]
        engine.load_yolo = Mock(return_value=detector)
        result = engine.process_image("photo.jpg", use_upscale=False, profile="exhaustive")
        engine.restore_image.assert_called_once()
        self.assertEqual(result["stage_calls"]["detector"], 2)


if __name__ == "__main__":
    unittest.main()
