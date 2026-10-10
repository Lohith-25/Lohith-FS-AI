"""
SolarMap-India — Phase 7B Half U-Net In-Domain Validation Suite.

Executes rigorous in-domain validation on strictly unseen test images from the
original SolarMap-India dataset to evaluate whether the production Half U-Net
generalizes correctly within its training domain vs out-of-distribution shift.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw
import torch

from ml.analytics.components import analyze_components
from ml.inference.model import HalfUNet, get_device, load_halfunet_model
from ml.inference.postprocessing import (
    create_overlay,
    mask_to_visual_image,
    process_logits,
    resize_mask_to_original,
)
from ml.inference.predict import InferenceResult, predict_image
from ml.inference.preprocessing import load_and_validate_image, preprocess_image

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
SPLIT_FILE = PROJECT_ROOT / "outputs" / "HalfUNet" / "metrics" / "split_image_ids.json"
ANNOTATIONS_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "annotations"
    / "merged_instances_default.json"
)
IMAGES_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "images"
    / "default"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation" / "phase7b"

# 10 Representative Unseen Test Images across Density Tiers
SELECTED_TEST_IMAGE_IDS = [
    1116,  # 863.0_1.0.png: 48 panels (Tier: Very High / Max test density)
    1732,  # 1732.0_1.0.png: 34 panels (Tier: High density / complex layout)
    1017,  # 764.0_1.0.png: 24 panels (Tier: High-medium density)
    102,   # 190.0_1.0.png: 17 panels (Tier: Medium density)
    181,   # 261.0_1.0.png: 10 panels (Tier: Median density)
    523,   # 57.0_1.0.png: 8 panels (Tier: Moderate-low density)
    422,   # 479.0_1.0.png: 4 panels (Tier: Low density)
    970,   # 717.0_1.0.png: 2 panels (Tier: Minimal density / challenge)
    1430,  # 1430.0_1.0.png: 1 panel (Tier: Minimal density / 1 panel)
    439,   # 494.0_1.0.png: 1 panel (Tier: Minimal density / 1 panel)
]

ZERO_CONTROL_IMAGE_ID = 526  # 572.0_1.0.png: 0 panels (In-domain negative control)


@dataclass
class ConfusionStats:
    tp: int
    tn: int
    fp: int
    fn: int
    total_pixels: int
    gt_solar_pixels: int
    pred_solar_pixels: int
    pixel_diff: int


@dataclass
class ImageMetrics:
    sample_id: int
    coco_image_id: int
    file_name: str
    density_tier: str
    annotation_count: int
    # Confusion matrix
    confusion: ConfusionStats
    # Foreground Solar Metrics
    dice: float
    iou: float
    precision: float
    recall: float
    accuracy: float
    fpr: float  # False Positive Rate = FP / (FP + TN)
    fnr: float  # False Negative Rate = FN / (FN + TP)
    # Macro Metrics (Background + Foreground)
    macro_dice: float
    macro_iou: float
    macro_precision: float
    macro_recall: float
    # Connected Components Error Analysis
    gt_components_count: int
    pred_components_count: int
    component_ratio: float  # pred / gt
    missed_components: int
    false_components: int
    inference_time_ms: float


@dataclass
class AggregateMetrics:
    num_samples: int
    mean_dice: float
    median_dice: float
    min_dice: float
    max_dice: float
    mean_iou: float
    median_iou: float
    min_iou: float
    max_iou: float
    mean_precision: float
    median_precision: float
    mean_recall: float
    median_recall: float
    mean_accuracy: float
    mean_fpr: float
    mean_fnr: float
    # Macro averages
    macro_mean_dice: float
    macro_mean_iou: float
    macro_mean_precision: float
    macro_mean_recall: float
    total_tp: int
    total_tn: int
    total_fp: int
    total_fn: int
    dataset_level_dice: float
    dataset_level_iou: float
    dataset_level_precision: float
    dataset_level_recall: float


def rasterize_polygons(
    segmentations: List[List[float]], height: int, width: int
) -> np.ndarray:
    """Rasterizes COCO polygon lists into a binary 2D mask uint8 {0, 1}."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in segmentations:
        if len(poly) < 6:
            continue
        xy = list(zip(poly[0::2], poly[1::2]))
        draw.polygon(xy, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def get_ground_truth_mask(
    entry: Dict[str, Any], annotations: List[Dict[str, Any]]
) -> np.ndarray:
    """Constructs ground truth semantic mask at image resolution."""
    h, w = entry["height"], entry["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        seg = ann.get("segmentation", [])
        if not seg or ann.get("iscrowd", 0) == 1:
            continue
        inst_mask = rasterize_polygons(seg, h, w)
        mask = np.maximum(mask, inst_mask)
    return mask


def compute_confusion_stats(
    gt_mask: np.ndarray, pred_mask: np.ndarray
) -> ConfusionStats:
    """Computes exact pixel counts for TP, TN, FP, FN."""
    tp = int(np.logical_and(gt_mask == 1, pred_mask == 1).sum())
    fp = int(np.logical_and(gt_mask == 0, pred_mask == 1).sum())
    fn = int(np.logical_and(gt_mask == 1, pred_mask == 0).sum())
    tn = int(np.logical_and(gt_mask == 0, pred_mask == 0).sum())

    total = int(gt_mask.size)
    gt_solar = tp + fn
    pred_solar = tp + fp
    diff = pred_solar - gt_solar

    return ConfusionStats(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total_pixels=total,
        gt_solar_pixels=gt_solar,
        pred_solar_pixels=pred_solar,
        pixel_diff=diff,
    )


def compute_metrics(
    confusion: ConfusionStats, eps: float = 1e-7
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Computes both foreground solar metrics and two-class macro metrics.
    Robust against zero-positive divisions.
    """
    tp, tn, fp, fn = confusion.tp, confusion.tn, confusion.fp, confusion.fn
    gt_pos = tp + fn
    pred_pos = tp + fp
    bg_actual = tn + fp

    # 1. Foreground Solar Class Metrics
    if gt_pos == 0 and pred_pos == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou = float(tp) / (tp + fp + fn + eps)
        precision = float(tp) / (tp + fp + eps) if pred_pos > 0 else 0.0
        recall = float(tp) / (tp + fn + eps) if gt_pos > 0 else 0.0

    accuracy = float(tp + tn) / float(confusion.total_pixels)
    fpr = float(fp) / float(bg_actual + eps) if bg_actual > 0 else 0.0
    fnr = float(fn) / float(gt_pos + eps) if gt_pos > 0 else 0.0

    fg_metrics = {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }

    # 2. Background Class Metrics
    tp_bg = tn
    fp_bg = fn
    fn_bg = fp
    prec_bg = float(tp_bg) / (tp_bg + fp_bg + eps)
    rec_bg = float(tp_bg) / (tp_bg + fn_bg + eps)
    iou_bg = float(tp_bg) / (tp_bg + fp_bg + fn_bg + eps)
    dice_bg = (2.0 * tp_bg) / (2.0 * tp_bg + fp_bg + fn_bg + eps)

    macro_metrics = {
        "macro_dice": float((dice + dice_bg) / 2.0),
        "macro_iou": float((iou + iou_bg) / 2.0),
        "macro_precision": float((precision + prec_bg) / 2.0),
        "macro_recall": float((recall + rec_bg) / 2.0),
    }

    return fg_metrics, macro_metrics


def analyze_component_errors(
    gt_mask: np.ndarray, pred_mask: np.ndarray, min_area: int = 15
) -> Tuple[int, int, float, int, int]:
    """
    Performs connected components analysis to identify fragmentation, merging,
    missed panels, and false alarm clusters.
    """
    # 8-connectivity labeling
    num_gt, labels_gt, stats_gt, _ = cv2.connectedComponentsWithStats(
        gt_mask.astype(np.uint8), connectivity=8
    )
    num_pred, labels_pred, stats_pred, _ = cv2.connectedComponentsWithStats(
        pred_mask.astype(np.uint8), connectivity=8
    )

    # Filter background component (index 0) and small noise (< min_area)
    gt_comp_indices = [
        i for i in range(1, num_gt) if stats_gt[i, cv2.CC_STAT_AREA] >= min_area
    ]
    pred_comp_indices = [
        i for i in range(1, num_pred) if stats_pred[i, cv2.CC_STAT_AREA] >= min_area
    ]

    count_gt = len(gt_comp_indices)
    count_pred = len(pred_comp_indices)
    ratio = (count_pred / count_gt) if count_gt > 0 else (1.0 if count_pred == 0 else 999.0)

    # Missed panels: GT components with zero overlap in pred_mask
    missed_panels = 0
    for idx in gt_comp_indices:
        comp_mask = labels_gt == idx
        if np.logical_and(comp_mask, pred_mask == 1).sum() == 0:
            missed_panels += 1

    # False positive components: Pred components with zero overlap in gt_mask
    false_components = 0
    for idx in pred_comp_indices:
        comp_mask = labels_pred == idx
        if np.logical_and(comp_mask, gt_mask == 1).sum() == 0:
            false_components += 1

    return count_gt, count_pred, ratio, missed_panels, false_components


def create_error_overlay(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Creates pixel-level diagnostic overlay:
    - True Positive (TP): GREEN [0, 230, 0]
    - False Positive (FP): RED [230, 30, 30]
    - False Negative (FN): BLUE / AMBER [0, 140, 255]
    """
    img_rgb = np.array(image.convert("RGB")).astype(np.float32)

    tp_mask = np.logical_and(gt_mask == 1, pred_mask == 1)
    fp_mask = np.logical_and(gt_mask == 0, pred_mask == 1)
    fn_mask = np.logical_and(gt_mask == 1, pred_mask == 0)

    overlay = img_rgb.copy()

    # Blend colors
    color_tp = np.array([0, 230, 0], dtype=np.float32)
    color_fp = np.array([240, 30, 30], dtype=np.float32)
    color_fn = np.array([0, 150, 255], dtype=np.float32)

    overlay[tp_mask] = (1.0 - alpha) * overlay[tp_mask] + alpha * color_tp
    overlay[fp_mask] = (1.0 - alpha) * overlay[fp_mask] + alpha * color_fp
    overlay[fn_mask] = (1.0 - alpha) * overlay[fn_mask] + alpha * color_fn

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def generate_comparison_dashboard(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    error_overlay: Image.Image,
    metrics: ImageMetrics,
    output_path: Path,
) -> None:
    """
    Generates a 4-panel comparison dashboard:
    1. Original Image (640x640)
    2. Ground Truth Mask
    3. Production Model Prediction
    4. Diagnostic Error Overlay with legend & metrics badge
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=150)
    fig.patch.set_facecolor("#18191A")

    # Panel 1: Original Image
    axes[0].imshow(image)
    axes[0].set_title("1. Original Image (640x640)", color="#FFFFFF", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Ground Truth
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"2. Ground Truth ({metrics.annotation_count} panels)",
        color="#FFFFFF",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    # Panel 3: Prediction
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"3. Half U-Net Prediction ({metrics.confusion.pred_solar_pixels:,} px)",
        color="#FFFFFF",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    # Panel 4: Error Overlay
    axes[3].imshow(error_overlay)
    axes[3].set_title("4. Diagnostic Error Overlay", color="#FFFFFF", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    # Legend for overlay
    patch_tp = mpatches.Patch(color=(0/255, 230/255, 0/255), label=f"TP: {metrics.confusion.tp:,}")
    patch_fp = mpatches.Patch(color=(240/255, 30/255, 30/255), label=f"FP: {metrics.confusion.fp:,}")
    patch_fn = mpatches.Patch(color=(0/255, 150/255, 255/255), label=f"FN: {metrics.confusion.fn:,}")
    leg = axes[3].legend(
        handles=[patch_tp, patch_fp, patch_fn],
        loc="lower right",
        facecolor="#242526",
        edgecolor="#4E4F50",
        fontsize=9,
        labelcolor="#E4E6EB",
    )

    # Metrics Summary Title Subtext
    summary_text = (
        f"ID: {metrics.coco_image_id} | File: {metrics.file_name} | Tier: {metrics.density_tier} | "
        f"Dice: {metrics.dice:.4f} | IoU: {metrics.iou:.4f} | Prec: {metrics.precision:.4f} | "
        f"Rec: {metrics.recall:.4f} | Acc: {metrics.accuracy:.4f}"
    )
    plt.suptitle(summary_text, color="#00D26A", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def aggregate_sample_metrics(metrics_list: List[ImageMetrics]) -> AggregateMetrics:
    """Computes sample-level and dataset-level aggregated statistics."""
    n = len(metrics_list)
    dices = [m.dice for m in metrics_list]
    ious = [m.iou for m in metrics_list]
    precisions = [m.precision for m in metrics_list]
    recalls = [m.recall for m in metrics_list]
    accuracies = [m.accuracy for m in metrics_list]
    fprs = [m.fpr for m in metrics_list]
    fnrs = [m.fnr for m in metrics_list]

    macro_dices = [m.macro_dice for m in metrics_list]
    macro_ious = [m.macro_iou for m in metrics_list]
    macro_precs = [m.macro_precision for m in metrics_list]
    macro_recs = [m.macro_recall for m in metrics_list]

    tot_tp = sum(m.confusion.tp for m in metrics_list)
    tot_tn = sum(m.confusion.tn for m in metrics_list)
    tot_fp = sum(m.confusion.fp for m in metrics_list)
    tot_fn = sum(m.confusion.fn for m in metrics_list)

    eps = 1e-7
    ds_dice = (2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + eps)
    ds_iou = float(tot_tp) / (tot_tp + tot_fp + tot_fn + eps)
    ds_prec = float(tot_tp) / (tot_tp + tot_fp + eps)
    ds_rec = float(tot_tp) / (tot_tp + tot_fn + eps)

    return AggregateMetrics(
        num_samples=n,
        mean_dice=float(np.mean(dices)),
        median_dice=float(np.median(dices)),
        min_dice=float(np.min(dices)),
        max_dice=float(np.max(dices)),
        mean_iou=float(np.mean(ious)),
        median_iou=float(np.median(ious)),
        min_iou=float(np.min(ious)),
        max_iou=float(np.max(ious)),
        mean_precision=float(np.mean(precisions)),
        median_precision=float(np.median(precisions)),
        mean_recall=float(np.mean(recalls)),
        median_recall=float(np.median(recalls)),
        mean_accuracy=float(np.mean(accuracies)),
        mean_fpr=float(np.mean(fprs)),
        mean_fnr=float(np.mean(fnrs)),
        macro_mean_dice=float(np.mean(macro_dices)),
        macro_mean_iou=float(np.mean(macro_ious)),
        macro_mean_precision=float(np.mean(macro_precs)),
        macro_mean_recall=float(np.mean(macro_recs)),
        total_tp=tot_tp,
        total_tn=tot_tn,
        total_fp=tot_fp,
        total_fn=tot_fn,
        dataset_level_dice=float(ds_dice),
        dataset_level_iou=float(ds_iou),
        dataset_level_precision=float(ds_prec),
        dataset_level_recall=float(ds_rec),
    )


def run_phase7b_validation(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: Optional[torch.device] = None,
) -> Tuple[List[ImageMetrics], AggregateMetrics, Optional[ImageMetrics], Dict[str, Any]]:
    """
    Executes the complete Phase 7B in-domain validation workflow.
    """
    out_dir = Path(output_dir)
    images_out = out_dir / "images"
    masks_out = out_dir / "masks"
    overlays_out = out_dir / "overlays"
    metrics_out = out_dir / "metrics"
    report_out = out_dir / "report"

    for d in [images_out, masks_out, overlays_out, metrics_out, report_out]:
        d.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device()

    print(f"[Phase 7B] Loading Half U-Net checkpoint from {checkpoint_path}...")
    model = load_halfunet_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    # Load COCO dataset annotations and split
    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)
    test_ids_set = set(split_data["test"])

    with open(ANNOTATIONS_FILE, "r") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    # Density tier metadata mapping
    tiers = {
        1116: "Very High (48 panels)",
        1732: "High (34 panels)",
        1017: "High-Medium (24 panels)",
        102: "Medium (17 panels)",
        181: "Median (10 panels)",
        523: "Moderate-Low (8 panels)",
        422: "Low (4 panels)",
        970: "Minimal (2 panels)",
        1430: "Minimal (1 panel)",
        439: "Minimal (1 panel)",
        526: "Zero-Solar Control (0 panels)",
    }

    # Evaluate the 10 Primary Selected Test Images
    metrics_list: List[ImageMetrics] = []

    print("[Phase 7B] Evaluating 10 representative unseen in-domain test images...")
    for idx, img_id in enumerate(SELECTED_TEST_IMAGE_IDS, 1):
        assert img_id in test_ids_set, f"Image ID {img_id} must be in test set!"
        entry = images_by_id[img_id]
        fn = entry["file_name"]
        anns = annotations_by_image.get(img_id, [])
        img_path = IMAGES_DIR / fn

        # 1. Load image and construct GT mask
        pil_img, (w, h) = load_and_validate_image(img_path)
        gt_mask = get_ground_truth_mask(entry, anns)

        # 2. Run existing production inference pipeline
        start_t = time.perf_counter()
        res: InferenceResult = predict_image(
            image_path=img_path,
            model=model,
            checkpoint_path=checkpoint_path,
            output_dir=out_dir / "temp_pred",
            device=device,
            threshold=0.5,
            min_component_area=20,
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        pred_mask = res.binary_mask

        # 3. Compute Confusion Stats & Metrics
        confusion = compute_confusion_stats(gt_mask, pred_mask)
        fg_m, macro_m = compute_metrics(confusion)
        c_gt, c_pred, c_ratio, missed_c, false_c = analyze_component_errors(
            gt_mask, pred_mask
        )

        stem = Path(fn).stem
        sid = entry["id"]

        image_metric = ImageMetrics(
            sample_id=sid,
            coco_image_id=img_id,
            file_name=fn,
            density_tier=tiers.get(img_id, "Unknown"),
            annotation_count=len(anns),
            confusion=confusion,
            dice=fg_m["dice"],
            iou=fg_m["iou"],
            precision=fg_m["precision"],
            recall=fg_m["recall"],
            accuracy=fg_m["accuracy"],
            fpr=fg_m["fpr"],
            fnr=fg_m["fnr"],
            macro_dice=macro_m["macro_dice"],
            macro_iou=macro_m["macro_iou"],
            macro_precision=macro_m["macro_precision"],
            macro_recall=macro_m["macro_recall"],
            gt_components_count=c_gt,
            pred_components_count=c_pred,
            component_ratio=c_ratio,
            missed_components=missed_c,
            false_components=false_c,
            inference_time_ms=elapsed_ms,
        )
        metrics_list.append(image_metric)

        # 4. Save Visual Artifacts
        # Save Original Image
        pil_img.save(images_out / f"{idx:02d}_{stem}_original.png", format="PNG")
        # Save GT Mask visual
        gt_vis = mask_to_visual_image(gt_mask)
        gt_vis.save(masks_out / f"{idx:02d}_{stem}_gt_mask.png", format="PNG")
        # Save Pred Mask visual
        pred_vis = mask_to_visual_image(pred_mask)
        pred_vis.save(masks_out / f"{idx:02d}_{stem}_pred_mask.png", format="PNG")
        # Save Diagnostic Error Overlay
        err_overlay = create_error_overlay(pil_img, gt_mask, pred_mask)
        err_overlay.save(overlays_out / f"{idx:02d}_{stem}_error_overlay.png", format="PNG")
        # Save 4-panel dashboard
        dashboard_path = overlays_out / f"{idx:02d}_{stem}_dashboard.png"
        generate_comparison_dashboard(
            pil_img, gt_mask, pred_mask, err_overlay, image_metric, dashboard_path
        )

        print(
            f"  [{idx:02d}/10] ID {img_id:4d} ({fn:14s}) | "
            f"Anns: {len(anns):2d} | Dice: {image_metric.dice:.4f} | "
            f"IoU: {image_metric.iou:.4f} | Prec: {image_metric.precision:.4f} | "
            f"Rec: {image_metric.recall:.4f} | Acc: {image_metric.accuracy:.4f}"
        )

    # Clean up temp_pred dir
    temp_dir = out_dir / "temp_pred"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 5. Evaluate In-Domain Zero-Solar Control Image (ID 526)
    control_metric: Optional[ImageMetrics] = None
    if ZERO_CONTROL_IMAGE_ID in images_by_id:
        ctrl_entry = images_by_id[ZERO_CONTROL_IMAGE_ID]
        ctrl_fn = ctrl_entry["file_name"]
        ctrl_path = IMAGES_DIR / ctrl_fn
        ctrl_img, _ = load_and_validate_image(ctrl_path)
        ctrl_gt = np.zeros((ctrl_entry["height"], ctrl_entry["width"]), dtype=np.uint8)

        start_t = time.perf_counter()
        ctrl_res = predict_image(
            image_path=ctrl_path,
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=0.5,
        )
        ctrl_time_ms = (time.perf_counter() - start_t) * 1000.0
        ctrl_pred = ctrl_res.binary_mask

        ctrl_conf = compute_confusion_stats(ctrl_gt, ctrl_pred)
        ctrl_fg, ctrl_macro = compute_metrics(ctrl_conf)
        c_gt, c_pred, c_ratio, missed_c, false_c = analyze_component_errors(
            ctrl_gt, ctrl_pred
        )

        control_metric = ImageMetrics(
            sample_id=ctrl_entry["id"],
            coco_image_id=ZERO_CONTROL_IMAGE_ID,
            file_name=ctrl_fn,
            density_tier=tiers.get(ZERO_CONTROL_IMAGE_ID, "Zero Control"),
            annotation_count=0,
            confusion=ctrl_conf,
            dice=ctrl_fg["dice"],
            iou=ctrl_fg["iou"],
            precision=ctrl_fg["precision"],
            recall=ctrl_fg["recall"],
            accuracy=ctrl_fg["accuracy"],
            fpr=ctrl_fg["fpr"],
            fnr=ctrl_fg["fnr"],
            macro_dice=ctrl_macro["macro_dice"],
            macro_iou=ctrl_macro["macro_iou"],
            macro_precision=ctrl_macro["macro_precision"],
            macro_recall=ctrl_macro["macro_recall"],
            gt_components_count=c_gt,
            pred_components_count=c_pred,
            component_ratio=c_ratio,
            missed_components=missed_c,
            false_components=false_c,
            inference_time_ms=ctrl_time_ms,
        )

        # Save control visualizations
        ctrl_stem = Path(ctrl_fn).stem
        ctrl_img.save(images_out / f"control_{ctrl_stem}_original.png", format="PNG")
        mask_to_visual_image(ctrl_gt).save(masks_out / f"control_{ctrl_stem}_gt_mask.png", format="PNG")
        mask_to_visual_image(ctrl_pred).save(masks_out / f"control_{ctrl_stem}_pred_mask.png", format="PNG")
        ctrl_overlay = create_error_overlay(ctrl_img, ctrl_gt, ctrl_pred)
        ctrl_overlay.save(overlays_out / f"control_{ctrl_stem}_error_overlay.png", format="PNG")
        generate_comparison_dashboard(
            ctrl_img,
            ctrl_gt,
            ctrl_pred,
            ctrl_overlay,
            control_metric,
            overlays_out / f"control_{ctrl_stem}_dashboard.png",
        )
        print(
            f"  [Control] ID {ZERO_CONTROL_IMAGE_ID} ({ctrl_fn}) | "
            f"GT Solar: 0 px | Pred Solar: {ctrl_conf.pred_solar_pixels:,} px | "
            f"FPR: {control_metric.fpr:.6f} | Accuracy: {control_metric.accuracy:.4f}"
        )

    # 6. Evaluate External / Out-of-Distribution Image (testing.gif / testimg.gif)
    external_stats: Dict[str, Any] = {}
    external_img_path = PROJECT_ROOT / "dataset" / "testimg.gif"
    if external_img_path.exists():
        ext_pil, (ext_w, ext_h) = load_and_validate_image(external_img_path)
        ext_res = predict_image(
            image_path=external_img_path,
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=0.5,
        )
        ext_pred = ext_res.binary_mask

        # Save external artifacts
        ext_out = out_dir / "external"
        ext_out.mkdir(parents=True, exist_ok=True)
        ext_pil.save(ext_out / "external_testing_gif_original.png", format="PNG")
        mask_to_visual_image(ext_pred).save(
            ext_out / "external_testing_gif_pred_mask.png", format="PNG"
        )
        create_overlay(ext_pil, ext_pred).save(
            ext_out / "external_testing_gif_overlay.png", format="PNG"
        )

        num_ext_comp, _, stats_ext, _ = cv2.connectedComponentsWithStats(
            ext_pred.astype(np.uint8), connectivity=8
        )
        valid_ext_comps = [
            i for i in range(1, num_ext_comp) if stats_ext[i, cv2.CC_STAT_AREA] >= 20
        ]

        external_stats = {
            "image_path": str(external_img_path),
            "dimensions": [ext_w, ext_h],
            "total_pixels": ext_w * ext_h,
            "predicted_solar_pixels": ext_res.solar_pixels,
            "predicted_solar_coverage_percent": ext_res.solar_coverage_percent,
            "detected_components_count": len(valid_ext_comps),
            "inference_time_ms": ext_res.inference_time_ms,
            "notes": (
                "External internet image tested qualitatively without ground truth. "
                "Represents potential out-of-distribution domain shift."
            ),
        }
        print(
            f"  [External] {external_img_path.name} ({ext_w}x{ext_h}) | "
            f"Pred Solar: {ext_res.solar_pixels:,} px ({ext_res.solar_coverage_percent:.2f}%) | "
            f"Components: {len(valid_ext_comps)}"
        )

    # 7. Aggregate Metrics across the 10 Primary Test Images
    aggregates = aggregate_sample_metrics(metrics_list)

    # 8. Save Metrics CSV & JSON
    metrics_summary_data = {
        "model_architecture": "HalfUNet (2-class: background, hassolar)",
        "checkpoint": str(checkpoint_path),
        "validation_dataset": "SolarMap-India (merged_instances_default.json)",
        "test_split_source": str(SPLIT_FILE),
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_metrics": asdict(aggregates),
        "per_image_metrics": [asdict(m) for m in metrics_list],
        "zero_control_metric": asdict(control_metric) if control_metric else None,
        "external_ood_test": external_stats,
    }

    with open(metrics_out / "phase7b_metrics.json", "w") as f:
        json.dump(metrics_summary_data, f, indent=2)

    # Also save CSV for easy tabular inspection
    import pandas as pd
    rows = []
    for m in metrics_list:
        rows.append({
            "sample_id": m.sample_id,
            "coco_image_id": m.coco_image_id,
            "file_name": m.file_name,
            "density_tier": m.density_tier,
            "annotation_count": m.annotation_count,
            "gt_solar_pixels": m.confusion.gt_solar_pixels,
            "pred_solar_pixels": m.confusion.pred_solar_pixels,
            "pixel_diff": m.confusion.pixel_diff,
            "tp": m.confusion.tp,
            "tn": m.confusion.tn,
            "fp": m.confusion.fp,
            "fn": m.confusion.fn,
            "dice": m.dice,
            "iou": m.iou,
            "precision": m.precision,
            "recall": m.recall,
            "accuracy": m.accuracy,
            "fpr": m.fpr,
            "fnr": m.fnr,
            "macro_dice": m.macro_dice,
            "macro_iou": m.macro_iou,
            "macro_precision": m.macro_precision,
            "macro_recall": m.macro_recall,
            "gt_components": m.gt_components_count,
            "pred_components": m.pred_components_count,
            "missed_components": m.missed_components,
            "false_components": m.false_components,
        })
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(metrics_out / "per_image_metrics.csv", index=False)

    return metrics_list, aggregates, control_metric, external_stats
