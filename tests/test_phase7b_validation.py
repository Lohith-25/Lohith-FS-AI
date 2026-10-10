"""
SolarMap-India — Phase 7B In-Domain Validation Test Suite.

Unit and integration tests for validation metrics, confusion calculations,
error overlays, and connected component diagnostics.
"""

from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from ml.validation.phase7b import (
    ConfusionStats,
    ImageMetrics,
    aggregate_sample_metrics,
    analyze_component_errors,
    compute_confusion_stats,
    compute_metrics,
    create_error_overlay,
    rasterize_polygons,
)


class TestPhase7BValidation(unittest.TestCase):
    """Test suite for Phase 7B validation algorithms and metrics."""

    def test_rasterize_polygons(self):
        """Validates polygon rasterization to binary mask."""
        # Square from (10, 10) to (50, 50)
        poly = [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]]
        mask = rasterize_polygons(poly, 100, 100)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(mask.sum(), 0)
        # Inside the square should be 1
        self.assertEqual(mask[25, 25], 1)
        # Outside should be 0
        self.assertEqual(mask[5, 5], 0)

    def test_rasterize_empty_or_degenerate_polygons(self):
        """Validates that empty or degenerate polygons return empty masks."""
        mask = rasterize_polygons([], 64, 64)
        self.assertEqual(mask.sum(), 0)

        # Degenerate (< 6 coordinates)
        mask_degen = rasterize_polygons([[10.0, 20.0]], 64, 64)
        self.assertEqual(mask_degen.sum(), 0)

    def test_compute_confusion_stats_perfect_match(self):
        """Validates confusion statistics on identical ground truth and prediction."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[10:30, 10:30] = 1  # 400 pixels
        pred = gt.copy()

        conf = compute_confusion_stats(gt, pred)
        self.assertEqual(conf.tp, 400)
        self.assertEqual(conf.tn, 9600)
        self.assertEqual(conf.fp, 0)
        self.assertEqual(conf.fn, 0)
        self.assertEqual(conf.total_pixels, 10000)
        self.assertEqual(conf.pixel_diff, 0)

    def test_compute_confusion_stats_errors(self):
        """Validates confusion statistics with known FP and FN."""
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0:2, 0:5] = 1  # 10 positive pixels

        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[0:2, 0:3] = 1  # 6 TP pixels, 4 FN pixels
        pred[5:7, 0:2] = 1  # 4 FP pixels

        conf = compute_confusion_stats(gt, pred)
        self.assertEqual(conf.tp, 6)
        self.assertEqual(conf.fn, 4)
        self.assertEqual(conf.fp, 4)
        self.assertEqual(conf.tn, 86)
        self.assertEqual(conf.gt_solar_pixels, 10)
        self.assertEqual(conf.pred_solar_pixels, 10)
        self.assertEqual(conf.pixel_diff, 0)

    def test_compute_metrics_perfect(self):
        """Validates metrics on perfect predictions."""
        conf = ConfusionStats(
            tp=500, tn=9500, fp=0, fn=0,
            total_pixels=10000, gt_solar_pixels=500,
            pred_solar_pixels=500, pixel_diff=0
        )
        fg_m, macro_m = compute_metrics(conf)

        self.assertAlmostEqual(fg_m["dice"], 1.0, places=4)
        self.assertAlmostEqual(fg_m["iou"], 1.0, places=4)
        self.assertAlmostEqual(fg_m["precision"], 1.0, places=4)
        self.assertAlmostEqual(fg_m["recall"], 1.0, places=4)
        self.assertAlmostEqual(fg_m["accuracy"], 1.0, places=4)
        self.assertAlmostEqual(fg_m["fpr"], 0.0, places=4)
        self.assertAlmostEqual(fg_m["fnr"], 0.0, places=4)
        self.assertAlmostEqual(macro_m["macro_dice"], 1.0, places=4)
        self.assertAlmostEqual(macro_m["macro_iou"], 1.0, places=4)

    def test_compute_metrics_zero_positives_robustness(self):
        """Validates division-by-zero handling when both GT and Pred have zero positives."""
        conf = ConfusionStats(
            tp=0, tn=10000, fp=0, fn=0,
            total_pixels=10000, gt_solar_pixels=0,
            pred_solar_pixels=0, pixel_diff=0
        )
        fg_m, macro_m = compute_metrics(conf)

        self.assertEqual(fg_m["dice"], 1.0)
        self.assertEqual(fg_m["iou"], 1.0)
        self.assertEqual(fg_m["precision"], 1.0)
        self.assertEqual(fg_m["recall"], 1.0)
        self.assertEqual(fg_m["accuracy"], 1.0)
        self.assertEqual(fg_m["fpr"], 0.0)
        self.assertEqual(fg_m["fnr"], 0.0)

    def test_compute_metrics_complete_miss(self):
        """Validates metrics when model predicts nothing (complete miss)."""
        conf = ConfusionStats(
            tp=0, tn=9000, fp=0, fn=1000,
            total_pixels=10000, gt_solar_pixels=1000,
            pred_solar_pixels=0, pixel_diff=-1000
        )
        fg_m, macro_m = compute_metrics(conf)

        self.assertAlmostEqual(fg_m["dice"], 0.0, places=4)
        self.assertAlmostEqual(fg_m["iou"], 0.0, places=4)
        self.assertAlmostEqual(fg_m["recall"], 0.0, places=4)
        self.assertAlmostEqual(fg_m["fnr"], 1.0, places=4)

    def test_analyze_component_errors(self):
        """Validates connected components error detection."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        # Create 2 GT components of size 20x20
        gt[10:30, 10:30] = 1
        gt[60:80, 60:80] = 1

        pred = np.zeros((100, 100), dtype=np.uint8)
        # Match only the first component
        pred[10:30, 10:30] = 1
        # Add 1 false positive component
        pred[60:80, 10:30] = 1

        c_gt, c_pred, ratio, missed, false_c = analyze_component_errors(gt, pred, min_area=20)
        self.assertEqual(c_gt, 2)
        self.assertEqual(c_pred, 2)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertEqual(missed, 1)  # The component at [60:80, 60:80] was missed
        self.assertEqual(false_c, 1)  # The component at [60:80, 10:30] is a false alarm

    def test_create_error_overlay(self):
        """Validates diagnostic error overlay creation."""
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        gt = np.zeros((64, 64), dtype=np.uint8)
        pred = np.zeros((64, 64), dtype=np.uint8)

        gt[10:20, 10:20] = 1  # TP
        pred[10:20, 10:20] = 1

        gt[30:40, 10:20] = 1  # FN
        pred[30:40, 10:20] = 0

        gt[10:20, 30:40] = 0  # FP
        pred[10:20, 30:40] = 1

        overlay = create_error_overlay(img, gt, pred, alpha=0.5)
        self.assertEqual(overlay.size, (64, 64))
        self.assertEqual(overlay.mode, "RGB")
        overlay_arr = np.array(overlay)
        # TP pixel should have boosted green
        self.assertGreater(overlay_arr[15, 15, 1], overlay_arr[15, 15, 0])
        # FP pixel should have boosted red
        self.assertGreater(overlay_arr[15, 35, 0], overlay_arr[15, 35, 1])
        # FN pixel should have boosted blue
        self.assertGreater(overlay_arr[35, 15, 2], overlay_arr[35, 15, 0])

    def test_aggregate_sample_metrics(self):
        """Validates aggregation calculation across mock samples."""
        conf1 = ConfusionStats(
            tp=100, tn=900, fp=20, fn=10,
            total_pixels=1030, gt_solar_pixels=110, pred_solar_pixels=120, pixel_diff=10
        )
        conf2 = ConfusionStats(
            tp=200, tn=800, fp=10, fn=20,
            total_pixels=1030, gt_solar_pixels=220, pred_solar_pixels=210, pixel_diff=-10
        )
        fg1, mac1 = compute_metrics(conf1)
        fg2, mac2 = compute_metrics(conf2)

        m1 = ImageMetrics(
            sample_id=1, coco_image_id=1, file_name="img1.png",
            density_tier="High", annotation_count=5, confusion=conf1,
            dice=fg1["dice"], iou=fg1["iou"], precision=fg1["precision"],
            recall=fg1["recall"], accuracy=fg1["accuracy"], fpr=fg1["fpr"],
            fnr=fg1["fnr"], macro_dice=mac1["macro_dice"], macro_iou=mac1["macro_iou"],
            macro_precision=mac1["macro_precision"], macro_recall=mac1["macro_recall"],
            gt_components_count=5, pred_components_count=5, component_ratio=1.0,
            missed_components=0, false_components=0, inference_time_ms=10.0
        )
        m2 = ImageMetrics(
            sample_id=2, coco_image_id=2, file_name="img2.png",
            density_tier="Low", annotation_count=2, confusion=conf2,
            dice=fg2["dice"], iou=fg2["iou"], precision=fg2["precision"],
            recall=fg2["recall"], accuracy=fg2["accuracy"], fpr=fg2["fpr"],
            fnr=fg2["fnr"], macro_dice=mac2["macro_dice"], macro_iou=mac2["macro_iou"],
            macro_precision=mac2["macro_precision"], macro_recall=mac2["macro_recall"],
            gt_components_count=2, pred_components_count=2, component_ratio=1.0,
            missed_components=0, false_components=0, inference_time_ms=10.0
        )

        aggs = aggregate_sample_metrics([m1, m2])
        self.assertEqual(aggs.num_samples, 2)
        self.assertAlmostEqual(aggs.mean_dice, (m1.dice + m2.dice) / 2.0)
        self.assertEqual(aggs.min_dice, min(m1.dice, m2.dice))
        self.assertEqual(aggs.max_dice, max(m1.dice, m2.dice))
        self.assertEqual(aggs.total_tp, 300)
        self.assertEqual(aggs.total_tn, 1700)


if __name__ == "__main__":
    unittest.main()
