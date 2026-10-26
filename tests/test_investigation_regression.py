"""
SolarMap-India — Investigation & Failure Analysis Regression Test Suite.
Validates the complete pipeline across all criteria defined in Phase 6:
1. The 637x463 screenshot test across letterbox and tiled inference.
2. 256x256 and 640x640 benchmark image inference.
3. 1920x1080 and 4000x3000 high-resolution images.
4. Probability map range [0.0, 1.0], spatial shape consistency, and monotonic threshold behavior.
5. Mask and overlay dimension preservation across all input geometries.
6. Mathematical coverage percentage calculation adherence.
7. Safe handling of empty masks and rejection of corrupted/invalid uploads.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.model import get_device, load_halfunet_model
from ml.inference.predict import DEFAULT_CHECKPOINT, predict_image
from ml.analytics.components import compute_pixel_area_metrics
from ml.inference.postprocessing import (
    process_logits,
    unpad_and_restore_mask,
    unpad_and_restore_probability_map,
    resize_mask_to_original,
    resize_probability_map_to_original,
)
from ml.inference.preprocessing import (
    MODEL_INPUT_SIZE,
    compute_letterbox_params,
    letterbox_image,
    preprocess_image_letterbox,
)

SAMPLE_637x463_IMAGE = PROJECT_ROOT / "backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png"
SAMPLE_640x640_IMAGE = PROJECT_ROOT / "dataset/Solar Images/Solar Images/images/default/768.0_1.0.png"
TEST_OUTPUT_DIR = PROJECT_ROOT / "outputs/test_investigation_regression"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TestInvestigationRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.model = load_halfunet_model(DEFAULT_CHECKPOINT, device=cls.device)

    def _create_temp_image(self, width: int, height: int, filename: str, color=(100, 120, 140)) -> Path:
        img = Image.new("RGB", (width, height), color=color)
        path = TEST_OUTPUT_DIR / filename
        img.save(path, format="PNG")
        return path

    def test_01_uploaded_637x463_image_letterbox_and_tiled(self):
        """Verify the 637x463 screenshot under both letterbox and tiled strategies."""
        self.assertTrue(SAMPLE_637x463_IMAGE.exists(), f"Missing file: {SAMPLE_637x463_IMAGE}")

        # Letterbox strategy
        res_lb = predict_image(
            SAMPLE_637x463_IMAGE,
            model=self.model,
            output_dir=TEST_OUTPUT_DIR,
            device=self.device,
            strategy="letterbox",
            threshold=0.50,
            min_component_area=10,
        )
        self.assertEqual(res_lb.original_size, (637, 463))
        self.assertEqual(res_lb.binary_mask.shape, (463, 637))
        self.assertEqual(res_lb.solar_probabilities.shape, (463, 637))
        self.assertEqual(res_lb.solar_pixels, 6513)
        self.assertEqual(res_lb.total_pixels, 637 * 463)
        expected_cov = 100.0 * 6513 / (637 * 463)
        self.assertAlmostEqual(res_lb.solar_coverage_percent, expected_cov, places=4)
        self.assertEqual(res_lb.detected_region_count, 14)

        # Tiled strategy
        res_tiled = predict_image(
            SAMPLE_637x463_IMAGE,
            model=self.model,
            output_dir=TEST_OUTPUT_DIR,
            device=self.device,
            strategy="tiled",
            threshold=0.50,
            min_component_area=10,
        )
        self.assertEqual(res_tiled.original_size, (637, 463))
        self.assertEqual(res_tiled.binary_mask.shape, (463, 637))
        self.assertEqual(res_tiled.solar_probabilities.shape, (463, 637))
        # Tiled evaluates at full 1:1 spatial resolution without downscaling
        self.assertGreaterEqual(res_tiled.solar_pixels, 6513)
        self.assertGreaterEqual(res_tiled.detected_region_count, 14)

    def test_02_standard_square_dimensions_256_and_640(self):
        """Verify 256x256 and 640x640 images preserve dimensions and valid values."""
        # 256x256
        p256 = self._create_temp_image(256, 256, "temp_256x256.png")
        res256 = predict_image(p256, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device, strategy="auto")
        self.assertEqual(res256.original_size, (256, 256))
        self.assertEqual(res256.binary_mask.shape, (256, 256))
        self.assertEqual(res256.solar_probabilities.shape, (256, 256))

        # 640x640 real sample
        if SAMPLE_640x640_IMAGE.exists():
            res640 = predict_image(SAMPLE_640x640_IMAGE, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device, strategy="auto")
            self.assertEqual(res640.original_size, (640, 640))
            self.assertEqual(res640.binary_mask.shape, (640, 640))
            self.assertEqual(res640.solar_probabilities.shape, (640, 640))
            self.assertEqual(res640.total_pixels, 640 * 640)

    def test_03_high_resolution_dimensions_1920_and_4000(self):
        """Verify 1920x1080 and 4000x3000 tiled inference executes safely and preserves size."""
        # 1920x1080
        p1080 = self._create_temp_image(1920, 1080, "temp_1920x1080.png")
        res1080 = predict_image(p1080, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device, strategy="auto")
        self.assertEqual(res1080.original_size, (1920, 1080))
        self.assertEqual(res1080.binary_mask.shape, (1080, 1920))
        self.assertEqual(res1080.solar_probabilities.shape, (1080, 1920))
        self.assertEqual(res1080.processing_strategy, "tiled")

        # 4000x3000
        p3000 = self._create_temp_image(4000, 3000, "temp_4000x3000.png")
        res3000 = predict_image(p3000, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device, strategy="auto")
        self.assertEqual(res3000.original_size, (4000, 3000))
        self.assertEqual(res3000.binary_mask.shape, (3000, 4000))
        self.assertEqual(res3000.solar_probabilities.shape, (3000, 4000))
        self.assertEqual(res3000.processing_strategy, "tiled")

    def test_04_probability_map_range_and_threshold_monotonicity(self):
        """Verify probabilities are strictly [0.0, 1.0] and threshold counts are monotonic non-increasing."""
        res = predict_image(SAMPLE_637x463_IMAGE, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device, strategy="letterbox")
        prob_map = res.solar_probabilities

        self.assertGreaterEqual(float(prob_map.min()), 0.0)
        self.assertLessEqual(float(prob_map.max()), 1.0)
        self.assertFalse(np.isnan(prob_map).any())

        # Test threshold sensitivity monotonicity
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
        pixel_counts = []
        for t in thresholds:
            r = predict_image(SAMPLE_637x463_IMAGE, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device, strategy="letterbox", threshold=t)
            pixel_counts.append(r.solar_pixels)

        # Positive pixel count must be non-increasing as threshold increases
        for i in range(len(pixel_counts) - 1):
            self.assertGreaterEqual(
                pixel_counts[i],
                pixel_counts[i + 1],
                f"Pixel count not monotonic: {pixel_counts[i]} at {thresholds[i]} < {pixel_counts[i+1]} at {thresholds[i+1]}"
            )

    def test_05_mask_and_overlay_dimension_preservation(self):
        """Verify saved PNG mask and overlay files match exact original dimensions."""
        sizes = [(100, 100), (333, 444), (512, 256)]
        for w, h in sizes:
            p = self._create_temp_image(w, h, f"dim_test_{w}x{h}.png")
            res = predict_image(p, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device)
            with Image.open(res.mask_path) as m_img:
                self.assertEqual(m_img.size, (w, h))
            with Image.open(res.overlay_path) as o_img:
                self.assertEqual(o_img.size, (w, h))

    def test_06_coverage_formula_correctness(self):
        """Verify solar_coverage_percent strictly equals 100.0 * solar_pixels / (width * height)."""
        w, h = 637, 463
        tot = w * h
        sol_px = 6513
        expected = 100.0 * sol_px / tot

        mask = np.zeros((h, w), dtype=np.uint8)
        mask.flat[:sol_px] = 1

        metrics = compute_pixel_area_metrics(mask, min_component_area=10)
        self.assertEqual(metrics["image_width"], w)
        self.assertEqual(metrics["image_height"], h)
        self.assertEqual(metrics["total_image_pixels"], tot)
        self.assertEqual(metrics["solar_area_pixels"], sol_px)
        self.assertAlmostEqual(metrics["solar_coverage_percent"], expected, places=4)

    def test_07_empty_mask_and_invalid_uploads(self):
        """Verify empty mask safe outputs and corrupted input rejection."""
        # Empty mask
        empty_mask = np.zeros((463, 637), dtype=np.uint8)
        metrics = compute_pixel_area_metrics(empty_mask, min_component_area=10)
        self.assertEqual(metrics["solar_area_pixels"], 0)
        self.assertEqual(metrics["solar_coverage_percent"], 0.0)
        self.assertEqual(metrics["detected_region_count"], 0)
        self.assertEqual(metrics["physical_area_status"], "insufficient_data")

        # Corrupted input file
        corrupt_path = TEST_OUTPUT_DIR / "corrupt.png"
        corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_HEADER")
        with self.assertRaises(ValueError):
            predict_image(corrupt_path, model=self.model, output_dir=TEST_OUTPUT_DIR, device=self.device)


if __name__ == "__main__":
    unittest.main()
