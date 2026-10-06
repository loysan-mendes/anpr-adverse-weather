"""Tests for blur countermeasures: Unsharp Masking and Morphological Stroke Enhancement."""

import unittest
from pathlib import Path
import sys
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from image_ops import sharpen_unsharp_mask, enhance_plate_strokes
import cv2


class BlurEnhancementTests(unittest.TestCase):
    def test_unsharp_mask_enhances_blurred_step_edge(self):
        # Create a blurred step edge image
        edge_img = np.zeros((30, 60, 3), dtype=np.uint8)
        edge_img[:, 30:] = 200
        blurred = cv2.GaussianBlur(edge_img, (5, 5), 1.5)

        sharpened = sharpen_unsharp_mask(blurred, sigma=1.0, strength=1.5)
        self.assertEqual(sharpened.shape, blurred.shape)
        self.assertEqual(sharpened.dtype, np.uint8)

        # Gradient at the edge should be strictly steeper in the sharpened image
        blur_grad = int(blurred[15, 31, 0]) - int(blurred[15, 29, 0])
        sharp_grad = int(sharpened[15, 31, 0]) - int(sharpened[15, 29, 0])
        self.assertGreater(sharp_grad, blur_grad)

    def test_enhance_plate_strokes_boosts_dark_text_contrast(self):
        # Synthetic white plate with dark simulated text characters
        plate = np.full((30, 80, 3), 220, dtype=np.uint8)
        # Draw dark strokes
        plate[10:20, 20:25] = 40
        plate[10:20, 35:40] = 40
        blurred = cv2.GaussianBlur(plate, (3, 3), 1.0)

        enhanced = enhance_plate_strokes(blurred)
        self.assertEqual(enhanced.shape, plate.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

        # Text region should still be distinctly darker than plate background
        text_pixel = int(enhanced[15, 22, 0])
        bg_pixel = int(enhanced[5, 5, 0])
        self.assertGreater(bg_pixel, text_pixel)

    def test_empty_or_none_safely_handled(self):
        self.assertIsNone(sharpen_unsharp_mask(None))
        self.assertIsNone(enhance_plate_strokes(None))

        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(sharpen_unsharp_mask(empty).size, 0)
        self.assertEqual(enhance_plate_strokes(empty).size, 0)


if __name__ == "__main__":
    unittest.main()
