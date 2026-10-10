"""
SolarMap-India — Phase 2 Segmentation Analytics Test Suite.

Automated tests for:
1. Binary mask validation (valid {0, 1}, {0, 255}, rejecting corrupt/non-binary values)
2. Pixel counting (solar_pixels, background_pixels, total_pixels)
3. Solar coverage calculation
4. Background calculation consistency
5. Connected component detection
6. Component filtering & threshold retention
7. Bounding box correctness & boundaries
8. Centroid correctness & boundaries
9. JSON report generation
10. Real Phase 1 mask integration (768.0_1.0_mask.png)
"""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analytics.components import (
    analyze_components,
    extract_connected_components,
)
from ml.analytics.mask_analysis import (
    compute_pixel_statistics,
    load_and_validate_mask,
    run_segmentation_analytics,
)
from ml.analytics.report import build_report_dictionary

REAL_MASK_PATH = PROJECT_ROOT / "outputs" / "inference" / "masks" / "768.0_1.0_mask.png"
REAL_IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "images"
    / "default"
    / "768.0_1.0.png"
)
TEST_ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics"


class TestSolarSegmentationAnalytics(unittest.TestCase):

    def setUp(self):
        # Deterministic synthetic test mask: 100x100
        # Region 1: 10x10 square at (10, 10), area = 100
        # Region 2: 3x3 square at (50, 50), area = 9 (noise)
        self.synthetic_mask = np.zeros((100, 100), dtype=np.uint8)
        self.synthetic_mask[10:20, 10:20] = 1  # 100 pixels
        self.synthetic_mask[50:53, 50:53] = 1  # 9 pixels

    def test_01_binary_mask_validation(self):
        """Test 1: Normalization of {0, 255} and rejection of invalid values."""
        # 1. Valid {0, 1}
        stats = compute_pixel_statistics(self.synthetic_mask)
        self.assertEqual(stats.solar_pixels, 109)
        self.assertEqual(stats.total_pixels, 10000)

        # 2. Invalid multi-class values {0, 1, 2}
        invalid_mask = self.synthetic_mask.copy()
        invalid_mask[0, 0] = 2
        with self.assertRaises(ValueError):
            extract_connected_components(invalid_mask)

    def test_02_pixel_counting(self):
        """Test 2: Solar and background pixel counts."""
        stats = compute_pixel_statistics(self.synthetic_mask)
        self.assertEqual(stats.solar_pixels, 109)
        self.assertEqual(stats.background_pixels, 10000 - 109)

    def test_03_solar_coverage_calculation(self):
        """Test 3: Solar pixel coverage percentage calculation."""
        stats = compute_pixel_statistics(self.synthetic_mask)
        expected_coverage = (109 / 10000) * 100.0  # 1.09%
        self.assertAlmostEqual(stats.solar_coverage_percent, expected_coverage, places=4)

    def test_04_background_calculation_consistency(self):
        """Test 4: solar_pixels + background_pixels == total_pixels."""
        stats = compute_pixel_statistics(self.synthetic_mask)
        self.assertEqual(
            stats.solar_pixels + stats.background_pixels,
            stats.total_pixels,
        )

    def test_05_connected_component_detection(self):
        """Test 5: Detection of individual connected regions."""
        regions, label_map = extract_connected_components(self.synthetic_mask)
        self.assertEqual(len(regions), 2)
        areas = sorted([r.area_pixels for r in regions])
        self.assertEqual(areas, [9, 100])

    def test_06_component_filtering(self):
        """Test 6: Configurable area threshold filtering."""
        # Threshold at 20: 9-pixel region dropped, 100-pixel region kept
        stats, _ = analyze_components(self.synthetic_mask, min_component_area=20)
        self.assertEqual(stats.raw_count, 2)
        self.assertEqual(stats.filtered_count, 1)
        self.assertEqual(stats.largest_area_pixels, 100)
        self.assertEqual(stats.smallest_area_pixels, 100)
        self.assertEqual(stats.filtered_solar_pixels, 100)
        self.assertAlmostEqual(stats.retained_percentage, (100 / 109) * 100.0, places=2)

    def test_07_bounding_box_correctness(self):
        """Test 7: Bounding box coordinates and dimensions."""
        regions, _ = extract_connected_components(self.synthetic_mask)
        # Find 100-pixel region
        reg100 = [r for r in regions if r.area_pixels == 100][0]
        self.assertEqual(reg100.x, 10)
        self.assertEqual(reg100.y, 10)
        self.assertEqual(reg100.width, 10)
        self.assertEqual(reg100.height, 10)

    def test_08_centroid_correctness(self):
        """Test 8: Centroid calculation within boundaries."""
        regions, _ = extract_connected_components(self.synthetic_mask)
        reg100 = [r for r in regions if r.area_pixels == 100][0]
        # Centroid of [10:20, 10:20] should be ~14.5
        self.assertAlmostEqual(reg100.centroid_x, 14.5, places=1)
        self.assertAlmostEqual(reg100.centroid_y, 14.5, places=1)

    def test_09_json_report_generation(self):
        """Test 9: Report dictionary schema and completeness."""
        stats, _ = analyze_components(self.synthetic_mask, min_component_area=20)
        report = build_report_dictionary(
            image_name="test_synth.png",
            width=100,
            height=100,
            solar_pixels=109,
            background_pixels=10000 - 109,
            solar_coverage_percent=1.09,
            component_stats=stats,
        )
        self.assertIn("image", report)
        self.assertIn("segmentation", report)
        self.assertIn("components", report)
        self.assertIn("regions", report)
        self.assertEqual(len(report["regions"]), 1)
        self.assertEqual(report["components"]["filtered_count"], 1)

    def test_10_real_phase1_mask_integration(self):
        """Test 10: Full integration test on actual Phase 1 output."""
        self.assertTrue(REAL_MASK_PATH.is_file(), f"Phase 1 mask missing: {REAL_MASK_PATH}")

        report, json_path, vis_path = run_segmentation_analytics(
            mask_path=REAL_MASK_PATH,
            original_image_path=REAL_IMAGE_PATH,
            min_component_area=20,
            output_dir=TEST_ANALYTICS_DIR,
        )

        self.assertTrue(json_path.is_file(), "Analytics JSON report was not created")
        self.assertTrue(vis_path.is_file(), "Visualization PNG was not created")

        # Independent cross-check against Phase 1 recorded measurements
        img_info = report["image"]
        seg_info = report["segmentation"]
        comp_info = report["components"]

        self.assertEqual(img_info["width"], 640)
        self.assertEqual(img_info["height"], 640)
        self.assertEqual(img_info["total_pixels"], 409600)
        self.assertEqual(seg_info["solar_pixels"], 68451)
        self.assertEqual(seg_info["background_pixels"], 409600 - 68451)
        self.assertAlmostEqual(seg_info["solar_pixel_coverage_percent"], 16.7117, places=2)

        # Component metrics
        self.assertEqual(comp_info["raw_count"], 43)
        self.assertEqual(comp_info["filtered_count"], 43)
        self.assertEqual(comp_info["largest_area_pixels"], 4750)
        self.assertEqual(comp_info["smallest_area_pixels"], 616)
        self.assertEqual(comp_info["filtered_solar_pixels"], 68451)
        self.assertEqual(comp_info["retained_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
