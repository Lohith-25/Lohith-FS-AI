"""
SolarMap-India — Phase 3 Prediction Confidence & Uncertainty Test Suite.

Automated tests for:
1. Probability range validation (in [0, 1], reject invalid bounds)
2. Probability sum validation (P(bg) + P(solar) ≈ 1)
3. Confidence calculation (max(bg, solar))
4. Uncertainty calculation (1 - confidence)
5. Confidence and uncertainty mathematical bounds
6. Histogram count consistency (sum of 10 bins == total pixels)
7. Solar-only probability statistics
8. Background-only probability statistics
9. Region confidence calculation
10. Region classification thresholds (HIGH, MEDIUM, LOW)
11. Phase 1 mask consistency (pixel-for-pixel match)
12. Phase 2 component consistency (count, IDs, areas)
13. Real image end-to-end confidence pipeline execution
"""

import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analytics.components import SolarRegion, analyze_components
from ml.confidence.probability_analysis import (
    compute_ambiguity_stats,
    compute_confidence_and_uncertainty,
    compute_confidence_histogram,
    compute_stats,
    extract_model_probabilities,
    run_confidence_pipeline,
    validate_probabilities,
)
from ml.confidence.region_confidence import (
    classify_region_confidence,
    compute_region_confidences,
)
from ml.inference.predict import DEFAULT_CHECKPOINT

REAL_IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "images"
    / "default"
    / "768.0_1.0.png"
)
PHASE1_MASK_PATH = PROJECT_ROOT / "outputs" / "inference" / "masks" / "768.0_1.0_mask.png"
PHASE2_REPORT_PATH = PROJECT_ROOT / "outputs" / "analytics" / "reports" / "768.0_1.0_analytics.json"


class TestConfidenceAndUncertainty(unittest.TestCase):

    def setUp(self):
        # Deterministic 10x10 synthetic test arrays
        self.p_solar = np.array([
            [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50],
            [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00],
        ] * 5, dtype=np.float32)  # shape (10, 10)
        self.p_bg = 1.0 - self.p_solar

    def test_01_probability_range_validation(self):
        """Test 1: Normal range passes; out-of-range values raise ValueError."""
        dev = validate_probabilities(self.p_bg, self.p_solar)
        self.assertLess(dev, 1e-5)

        # Negative value
        bad_solar = self.p_solar.copy()
        bad_solar[0, 0] = -0.1
        with self.assertRaises(ValueError):
            validate_probabilities(self.p_bg, bad_solar)

        # > 1.0 value
        bad_bg = self.p_bg.copy()
        bad_bg[0, 0] = 1.2
        with self.assertRaises(ValueError):
            validate_probabilities(bad_bg, self.p_solar)

    def test_02_probability_sum_validation(self):
        """Test 2: Sum must equal 1.0 within tolerance."""
        bad_sum = self.p_bg.copy()
        bad_sum[0, 0] = 0.5  # sum = 0.5 + 0.95 = 1.45
        with self.assertRaises(ValueError):
            validate_probabilities(bad_sum, self.p_solar)

    def test_03_confidence_calculation(self):
        """Test 3: Confidence == max(bg, solar)."""
        conf, _ = compute_confidence_and_uncertainty(self.p_bg, self.p_solar)
        expected = np.maximum(self.p_bg, self.p_solar)
        np.testing.assert_allclose(conf, expected, atol=1e-6)

    def test_04_uncertainty_calculation(self):
        """Test 4: Uncertainty == 1 - confidence."""
        conf, unc = compute_confidence_and_uncertainty(self.p_bg, self.p_solar)
        np.testing.assert_allclose(unc, 1.0 - conf, atol=1e-6)

    def test_05_confidence_uncertainty_bounds(self):
        """Test 5: Bounds for two-class: conf in [0.5, 1.0], unc in [0.0, 0.5]."""
        conf, unc = compute_confidence_and_uncertainty(self.p_bg, self.p_solar)
        self.assertGreaterEqual(conf.min(), 0.50 - 1e-6)
        self.assertLessEqual(conf.max(), 1.00 + 1e-6)
        self.assertGreaterEqual(unc.min(), 0.00 - 1e-6)
        self.assertLessEqual(unc.max(), 0.50 + 1e-6)

    def test_06_histogram_consistency(self):
        """Test 6: 10-bin histogram sum equals total elements."""
        conf, _ = compute_confidence_and_uncertainty(self.p_bg, self.p_solar)
        hist = compute_confidence_histogram(conf, num_bins=10)
        self.assertEqual(len(hist), 10)
        total_hist_pixels = sum(b["pixel_count"] for b in hist)
        self.assertEqual(total_hist_pixels, conf.size)

    def test_07_solar_probability_statistics(self):
        """Test 7: Solar-only statistics calculation."""
        solar_mask = (self.p_solar >= 0.5)
        stats = compute_stats(self.p_solar[solar_mask])
        self.assertIsNotNone(stats)
        self.assertGreaterEqual(stats.min, 0.5)
        self.assertLessEqual(stats.max, 1.0)
        self.assertAlmostEqual(stats.mean, float(np.mean(self.p_solar[solar_mask])), places=4)

    def test_08_background_probability_statistics(self):
        """Test 8: Background-only statistics calculation."""
        bg_mask = (self.p_solar < 0.5)
        stats = compute_stats(self.p_bg[bg_mask])
        self.assertIsNotNone(stats)
        self.assertGreaterEqual(stats.min, 0.5)
        self.assertLessEqual(stats.max, 1.0)
        self.assertAlmostEqual(stats.mean, float(np.mean(self.p_bg[bg_mask])), places=4)

    def test_09_region_confidence_calculation(self):
        """Test 9: Region confidence metrics on mock label map."""
        label_map = np.zeros((10, 10), dtype=np.int32)
        label_map[0:3, 0:3] = 1  # 9 pixels region
        reg = SolarRegion(
            id=1,
            area_pixels=9,
            x=0,
            y=0,
            width=3,
            height=3,
            centroid_x=1.0,
            centroid_y=1.0,
        )
        conf, unc = compute_confidence_and_uncertainty(self.p_bg, self.p_solar)
        records = compute_region_confidences(
            regions=[reg],
            label_map=label_map,
            solar_prob=self.p_solar,
            confidence_map=conf,
            uncertainty_map=unc,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, 1)
        self.assertEqual(records[0].area_pixels, 9)
        self.assertGreater(records[0].mean_solar_probability, 0.0)

    def test_10_region_classification_thresholds(self):
        """Test 10: Operational classifications (HIGH, MEDIUM, LOW)."""
        self.assertEqual(classify_region_confidence(0.95, high_threshold=0.90, medium_threshold=0.70), "HIGH")
        self.assertEqual(classify_region_confidence(0.90, high_threshold=0.90, medium_threshold=0.70), "HIGH")
        self.assertEqual(classify_region_confidence(0.85, high_threshold=0.90, medium_threshold=0.70), "MEDIUM")
        self.assertEqual(classify_region_confidence(0.70, high_threshold=0.90, medium_threshold=0.70), "MEDIUM")
        self.assertEqual(classify_region_confidence(0.65, high_threshold=0.90, medium_threshold=0.70), "LOW")

    def test_11_phase1_mask_pixel_consistency(self):
        """Test 11: Phase 3 generated mask is IDENTICAL pixel-for-pixel to Phase 1 mask."""
        self.assertTrue(REAL_IMAGE_PATH.is_file(), f"Image missing: {REAL_IMAGE_PATH}")
        self.assertTrue(PHASE1_MASK_PATH.is_file(), f"Phase 1 mask missing: {PHASE1_MASK_PATH}")

        # Extract probabilities and mask directly from model
        _, _, p3_mask, _ = extract_model_probabilities(
            image_path=REAL_IMAGE_PATH,
            checkpoint_path=DEFAULT_CHECKPOINT,
        )

        # Load Phase 1 saved mask
        p1_mask_raw = cv2.imread(str(PHASE1_MASK_PATH), cv2.IMREAD_GRAYSCALE)
        p1_mask = (p1_mask_raw > 0).astype(np.uint8)

        diff = int(np.count_nonzero(p1_mask != p3_mask))
        self.assertEqual(diff, 0, f"Discrepancy of {diff} pixels between Phase 1 and Phase 3 masks")

    def test_12_phase2_component_consistency(self):
        """Test 12: Phase 3 region analysis matches Phase 2 component counts, IDs, and areas."""
        self.assertTrue(PHASE2_REPORT_PATH.is_file(), f"Phase 2 report missing: {PHASE2_REPORT_PATH}")
        with open(PHASE2_REPORT_PATH, "r") as f:
            p2_data = json.load(f)

        # Run Phase 3 pipeline
        report, _ = run_confidence_pipeline(
            image_path=REAL_IMAGE_PATH,
            checkpoint_path=DEFAULT_CHECKPOINT,
        )

        p2_regions = p2_data["regions"]
        p3_regions = report["regions"]

        self.assertEqual(len(p2_regions), len(p3_regions), "Component counts differ between Phase 2 and 3")
        for r2, r3 in zip(p2_regions, p3_regions):
            self.assertEqual(r2["id"], r3["id"], f"ID mismatch: {r2['id']} vs {r3['id']}")
            self.assertEqual(r2["area_pixels"], r3["area_pixels"], f"Area mismatch for ID {r2['id']}")

    def test_13_real_image_end_to_end(self):
        """Test 13: End-to-end execution on real image generates all files."""
        report, artifacts = run_confidence_pipeline(
            image_path=REAL_IMAGE_PATH,
            checkpoint_path=DEFAULT_CHECKPOINT,
        )
        self.assertTrue(artifacts["json_report"].is_file())
        self.assertTrue(artifacts["region_csv"].is_file())
        self.assertTrue(artifacts["solar_probability_map"].is_file())
        self.assertTrue(artifacts["confidence_map"].is_file())
        self.assertTrue(artifacts["uncertainty_map"].is_file())

        self.assertAlmostEqual(report["probability_validation"]["max_probability_sum_deviation"], 0.0, places=4)
        self.assertGreater(report["probabilities"]["solar_pixels_probability"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
