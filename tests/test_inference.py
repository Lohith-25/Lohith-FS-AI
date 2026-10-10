"""
SolarMap-India — Phase 1 Test & Validation Suite.

Validates:
A. Model construction test & parameter count check
B. Checkpoint loading test
C. Dummy tensor forward-pass test ([1, 3, 256, 256] -> [1, 2, 256, 256])
D. Real dataset image inference test
E. Output dimension test (mask shape == original image shape)
F. Binary-mask-value test ({0, 1} unique values)
G. Error handling tests (missing files, invalid inputs)
"""

from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analytics.components import compute_pixel_area_metrics
from ml.inference.model import (
    EXPECTED_PARAM_COUNT,
    HalfUNet,
    get_device,
    load_halfunet_model,
)
from ml.inference.postprocessing import (
    create_overlay,
    mask_to_visual_image,
    process_logits,
    resize_mask_to_original,
)
from ml.inference.predict import predict_image
from ml.inference.preprocessing import (
    MODEL_INPUT_SIZE,
    load_and_validate_image,
    preprocess_image,
)

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
SAMPLE_REAL_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "images"
    / "default"
    / "768.0_1.0.png"
)
TEST_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"


class TestHalfUNetInference(unittest.TestCase):

    def setUp(self):
        self.device = get_device()
        self.checkpoint_path = CHECKPOINT_PATH
        self.sample_image_path = SAMPLE_REAL_IMAGE

    def test_a_model_construction(self):
        """Test A: Verify model architecture instantiation and parameter count."""
        model = HalfUNet(num_classes=2, in_channels=3, base_channels=32)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(
            total_params,
            EXPECTED_PARAM_COUNT,
            f"Expected {EXPECTED_PARAM_COUNT} params, got {total_params}",
        )
        self.assertEqual(model.classifier.out_channels, 2)
        self.assertEqual(model.classifier.in_channels, 32)

    def test_b_checkpoint_loading(self):
        """Test B: Verify checkpoint loads into HalfUNet with strict key matching."""
        self.assertTrue(
            self.checkpoint_path.is_file(),
            f"Checkpoint file missing: {self.checkpoint_path}",
        )
        model = load_halfunet_model(self.checkpoint_path, device=self.device)
        self.assertIsInstance(model, HalfUNet)
        self.assertFalse(model.training, "Model should be in eval() mode")

    def test_c_dummy_forward_pass(self):
        """Test C: Dummy tensor pass with shape [1, 3, 256, 256] -> [1, 2, 256, 256]."""
        model = load_halfunet_model(self.checkpoint_path, device=self.device)
        dummy_tensor = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = model(dummy_tensor)
        self.assertEqual(
            list(output.shape),
            [1, 2, 256, 256],
            f"Expected output shape [1, 2, 256, 256], got {list(output.shape)}",
        )
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf")

    def test_d_real_image_inference(self):
        """Test D: Run full end-to-end pipeline on real dataset image."""
        self.assertTrue(
            self.sample_image_path.is_file(),
            f"Sample real image missing: {self.sample_image_path}",
        )
        result = predict_image(
            image_path=self.sample_image_path,
            checkpoint_path=self.checkpoint_path,
            output_dir=TEST_OUTPUT_DIR,
            device=self.device,
        )
        self.assertEqual(result.model_name, "Half U-Net")
        self.assertTrue(result.mask_path.is_file(), "Mask file was not created on disk")
        self.assertTrue(result.overlay_path.is_file(), "Overlay file was not created on disk")
        self.assertGreater(result.inference_time_ms, 0.0)

    def test_e_output_dimension_preservation(self):
        """Test E: Verify output mask & overlay preserve original image dimensions."""
        orig_img, (orig_w, orig_h) = load_and_validate_image(self.sample_image_path)
        result = predict_image(
            image_path=self.sample_image_path,
            checkpoint_path=self.checkpoint_path,
            output_dir=TEST_OUTPUT_DIR,
            device=self.device,
        )
        self.assertEqual(result.original_size, (orig_w, orig_h))
        self.assertEqual(result.binary_mask.shape, (orig_h, orig_w))

        # Check saved mask image dimensions
        with Image.open(result.mask_path) as mask_img:
            self.assertEqual(mask_img.size, (orig_w, orig_h))

        # Check saved overlay image dimensions
        with Image.open(result.overlay_path) as overlay_img:
            self.assertEqual(overlay_img.size, (orig_w, orig_h))

    def test_f_binary_mask_values(self):
        """Test F: Output mask contains strictly binary values {0, 1}."""
        result = predict_image(
            image_path=self.sample_image_path,
            checkpoint_path=self.checkpoint_path,
            output_dir=TEST_OUTPUT_DIR,
            device=self.device,
        )
        unique_vals = set(np.unique(result.binary_mask))
        self.assertTrue(
            unique_vals.issubset({0, 1}),
            f"Binary mask contains non-binary values: {unique_vals}",
        )
        self.assertEqual(
            result.solar_pixels + int(np.count_nonzero(result.binary_mask == 0)),
            result.total_pixels,
        )

    def test_g_error_handling(self):
        """Test G: Verify robust error handling for missing/invalid paths."""
        # Non-existent image
        with self.assertRaises(FileNotFoundError):
            predict_image("non_existent_image_12345.png")

        # Non-existent checkpoint
        with self.assertRaises(FileNotFoundError):
            load_halfunet_model("non_existent_checkpoint_12345.pth")

    def test_h_connected_solar_regions(self):
        """Test H: Verify connected solar regions are calculated via Phase 2 analytics."""
        result = predict_image(
            image_path=self.sample_image_path,
            checkpoint_path=self.checkpoint_path,
            output_dir=TEST_OUTPUT_DIR,
            device=self.device,
        )
        self.assertIsNotNone(result.component_stats)
        self.assertEqual(result.component_stats.filtered_count, 43)
        self.assertEqual(result.component_stats.raw_count, 43)
        self.assertEqual(result.component_stats.min_component_area_pixels, 20)

    def test_i_standardized_pixel_area_inference_result(self):
        """Test I: Verify standardized pixel-area measurements exposed on InferenceResult."""
        result = predict_image(
            image_path=self.sample_image_path,
            checkpoint_path=self.checkpoint_path,
            output_dir=TEST_OUTPUT_DIR,
            device=self.device,
        )
        pixel_dict = result.to_pixel_area_dict()
        expected_keys = {
            "image_width",
            "image_height",
            "total_image_pixels",
            "solar_area_pixels",
            "solar_coverage_percent",
            "detected_region_count",
            "largest_region_area_pixels",
            "smallest_region_area_pixels",
            "mean_region_area_pixels",
            "physical_area_m2",
            "physical_area_hectares",
            "physical_area_status",
        }
        self.assertEqual(set(pixel_dict.keys()), expected_keys)

        # Verify values from real inference on 768.0_1.0.png (640x640)
        self.assertEqual(pixel_dict["image_width"], 640)
        self.assertEqual(pixel_dict["image_height"], 640)
        self.assertEqual(pixel_dict["total_image_pixels"], 409600)
        self.assertEqual(pixel_dict["solar_area_pixels"], 68451)
        self.assertAlmostEqual(pixel_dict["solar_coverage_percent"], 16.7117, places=3)
        self.assertEqual(pixel_dict["detected_region_count"], 43)
        self.assertEqual(pixel_dict["largest_region_area_pixels"], 4750)
        self.assertEqual(pixel_dict["smallest_region_area_pixels"], 616)
        self.assertAlmostEqual(pixel_dict["mean_region_area_pixels"], 1591.8837, places=2)

        # Physical area safety check
        self.assertIsNone(pixel_dict["physical_area_m2"])
        self.assertIsNone(pixel_dict["physical_area_hectares"])
        self.assertEqual(pixel_dict["physical_area_status"], "insufficient_data")

        # Property accessors match dictionary
        self.assertEqual(result.image_width, pixel_dict["image_width"])
        self.assertEqual(result.image_height, pixel_dict["image_height"])
        self.assertEqual(result.total_image_pixels, pixel_dict["total_image_pixels"])
        self.assertEqual(result.solar_area_pixels, pixel_dict["solar_area_pixels"])
        self.assertEqual(result.detected_region_count, pixel_dict["detected_region_count"])
        self.assertEqual(result.largest_region_area_pixels, pixel_dict["largest_region_area_pixels"])
        self.assertEqual(result.smallest_region_area_pixels, pixel_dict["smallest_region_area_pixels"])
        self.assertEqual(result.mean_region_area_pixels, pixel_dict["mean_region_area_pixels"])
        self.assertIsNone(result.physical_area_m2)
        self.assertIsNone(result.physical_area_hectares)
        self.assertEqual(result.physical_area_status, "insufficient_data")


class TestPixelAreaMeasurement(unittest.TestCase):
    """
    Direct tests for pixel-based solar area measurements, connected-region metrics,
    edge-case handling, and physical area safety.
    """

    def test_total_and_solar_pixel_counting(self):
        """Verify solar pixel counting and total pixel calculation directly from mask."""
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[10:40, 10:50] = 1  # 30 * 40 = 1200 pixels
        mask[100:110, 100:110] = 1  # 10 * 10 = 100 pixels

        metrics = compute_pixel_area_metrics(mask, min_component_area=20)
        self.assertEqual(metrics["image_width"], 300)
        self.assertEqual(metrics["image_height"], 200)
        self.assertEqual(metrics["total_image_pixels"], 60000)
        self.assertEqual(metrics["solar_area_pixels"], 1300)
        self.assertAlmostEqual(metrics["solar_coverage_percent"], (1300 / 60000) * 100.0, places=4)

    def test_coverage_percentage_calculation(self):
        """Verify coverage percentage formula: solar_area_pixels / total_image_pixels * 100."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:25, :] = 1  # 2500 pixels out of 10000 = 25%

        metrics = compute_pixel_area_metrics(mask, min_component_area=20)
        self.assertEqual(metrics["solar_area_pixels"], 2500)
        self.assertEqual(metrics["total_image_pixels"], 10000)
        self.assertEqual(metrics["solar_coverage_percent"], 25.0)

    def test_connected_region_pixel_areas(self):
        """Verify region count, largest, smallest, and mean area calculations with filtering."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Region 1: 50x50 = 2500 px
        mask[10:60, 10:60] = 1
        # Region 2: 20x25 = 500 px
        mask[80:100, 80:105] = 1
        # Region 3 (noise): 3x3 = 9 px (< min_component_area 20)
        mask[150:153, 150:153] = 1

        metrics = compute_pixel_area_metrics(mask, min_component_area=20)
        # Total solar pixels includes noise
        self.assertEqual(metrics["solar_area_pixels"], 2500 + 500 + 9)
        # Filtered regions exclude noise
        self.assertEqual(metrics["detected_region_count"], 2)
        self.assertEqual(metrics["largest_region_area_pixels"], 2500)
        self.assertEqual(metrics["smallest_region_area_pixels"], 500)
        self.assertEqual(metrics["mean_region_area_pixels"], 1500.0)

    def test_empty_mask_handling(self):
        """Verify empty mask (all zeros) returns safe nulls and 0 counts without error."""
        mask = np.zeros((640, 640), dtype=np.uint8)
        metrics = compute_pixel_area_metrics(mask, min_component_area=20)

        self.assertEqual(metrics["image_width"], 640)
        self.assertEqual(metrics["image_height"], 640)
        self.assertEqual(metrics["total_image_pixels"], 409600)
        self.assertEqual(metrics["solar_area_pixels"], 0)
        self.assertEqual(metrics["solar_coverage_percent"], 0.0)
        self.assertEqual(metrics["detected_region_count"], 0)
        self.assertIsNone(metrics["largest_region_area_pixels"])
        self.assertIsNone(metrics["smallest_region_area_pixels"])
        self.assertIsNone(metrics["mean_region_area_pixels"])
        self.assertIsNone(metrics["physical_area_m2"])
        self.assertIsNone(metrics["physical_area_hectares"])
        self.assertEqual(metrics["physical_area_status"], "insufficient_data")

    def test_all_solar_mask_handling(self):
        """Verify all-solar mask (100% solar) returns 1 connected region of full mask area."""
        mask = np.ones((50, 50), dtype=np.uint8)
        metrics = compute_pixel_area_metrics(mask, min_component_area=20)

        self.assertEqual(metrics["image_width"], 50)
        self.assertEqual(metrics["image_height"], 50)
        self.assertEqual(metrics["total_image_pixels"], 2500)
        self.assertEqual(metrics["solar_area_pixels"], 2500)
        self.assertEqual(metrics["solar_coverage_percent"], 100.0)
        self.assertEqual(metrics["detected_region_count"], 1)
        self.assertEqual(metrics["largest_region_area_pixels"], 2500)
        self.assertEqual(metrics["smallest_region_area_pixels"], 2500)
        self.assertEqual(metrics["mean_region_area_pixels"], 2500.0)

    def test_single_pixel_mask_handling(self):
        """Verify single-pixel mask under default noise filter vs threshold=1."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1

        # Default filter (min_area = 20): single pixel is filtered out
        metrics_filtered = compute_pixel_area_metrics(mask, min_component_area=20)
        self.assertEqual(metrics_filtered["solar_area_pixels"], 1)
        self.assertEqual(metrics_filtered["detected_region_count"], 0)
        self.assertIsNone(metrics_filtered["largest_region_area_pixels"])
        self.assertIsNone(metrics_filtered["smallest_region_area_pixels"])
        self.assertIsNone(metrics_filtered["mean_region_area_pixels"])

        # Threshold = 1: single pixel is retained as a valid detected region
        metrics_retained = compute_pixel_area_metrics(mask, min_component_area=1)
        self.assertEqual(metrics_retained["solar_area_pixels"], 1)
        self.assertEqual(metrics_retained["detected_region_count"], 1)
        self.assertEqual(metrics_retained["largest_region_area_pixels"], 1)
        self.assertEqual(metrics_retained["smallest_region_area_pixels"], 1)
        self.assertEqual(metrics_retained["mean_region_area_pixels"], 1.0)

    def test_missing_physical_resolution_safety(self):
        """Verify that physical area fields are null and status is 'insufficient_data'."""
        mask = np.ones((100, 100), dtype=np.uint8)
        metrics = compute_pixel_area_metrics(mask)

        self.assertIsNone(metrics["physical_area_m2"])
        self.assertIsNone(metrics["physical_area_hectares"])
        self.assertEqual(metrics["physical_area_status"], "insufficient_data")

    def test_no_division_by_zero_empty_dimensions(self):
        """Verify that a zero-dimension mask handles calculations safely without ZeroDivisionError."""
        zero_mask = np.zeros((0, 0), dtype=np.uint8)
        metrics = compute_pixel_area_metrics(zero_mask)

        self.assertEqual(metrics["image_width"], 0)
        self.assertEqual(metrics["image_height"], 0)
        self.assertEqual(metrics["total_image_pixels"], 0)
        self.assertEqual(metrics["solar_area_pixels"], 0)
        self.assertEqual(metrics["solar_coverage_percent"], 0.0)
        self.assertEqual(metrics["detected_region_count"], 0)
        self.assertIsNone(metrics["largest_region_area_pixels"])

    def test_invalid_mask_raises(self):
        """Verify that invalid mask dimensions or values raise ValueError."""
        # 1D array
        with self.assertRaises(ValueError):
            compute_pixel_area_metrics(np.zeros((100,), dtype=np.uint8))

        # 3D array
        with self.assertRaises(ValueError):
            compute_pixel_area_metrics(np.zeros((100, 100, 3), dtype=np.uint8))

        # Non-binary array (e.g. contains 2)
        invalid = np.zeros((100, 100), dtype=np.uint8)
        invalid[10, 10] = 2
        with self.assertRaises(ValueError):
            compute_pixel_area_metrics(invalid)


if __name__ == "__main__":
    unittest.main()


