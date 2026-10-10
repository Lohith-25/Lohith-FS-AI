"""
SolarMap-India — PV01 Zero-Shot Evaluation.

Evaluates the existing Half U-Net checkpoint on all 645 images from the external PV01 dataset
without modifying any model weights, checkpoints, or existing pipeline files.
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.model import HalfUNet, get_device, load_halfunet_model

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
PV01_DATASET_DIR = PROJECT_ROOT / "external_datasets" / "PV01" / "PV01"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "pv01_zero_shot"

MODEL_INPUT_SIZE = (256, 256)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
])


def evaluate_pv01(
    dataset_dir: Path = PV01_DATASET_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = OUTPUT_DIR,
    threshold: float = 0.5,
    device: torch.device = None,
) -> Dict[str, Any]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"PV01 dataset directory not found: {dataset_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    print(f"Loading Half U-Net checkpoint: {checkpoint_path}")
    model = load_halfunet_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    categories = [
        ("Brick", "PV01_Rooftop_Brick"),
        ("FlatConcrete", "PV01_Rooftop_FlatConcrete"),
        ("SteelTile", "PV01_Rooftop_SteelTile"),
    ]

    records: List[Dict[str, Any]] = []
    category_summary: Dict[str, Dict[str, Any]] = {}

    start_eval_time = time.time()
    total_processed = 0

    print("Beginning zero-shot evaluation across 645 PV01 images...")

    with torch.no_grad():
        for cat_name, cat_folder in categories:
            folder_path = dataset_dir / cat_folder
            if not folder_path.is_dir():
                print(f"Warning: Folder not found: {folder_path}")
                continue

            # Find all image-mask pairs
            all_bmps = sorted(list(folder_path.glob("*.bmp")))
            images = [f for f in all_bmps if not f.stem.endswith("_label")]

            cat_tp = 0
            cat_tn = 0
            cat_fp = 0
            cat_fn = 0
            cat_dices = []
            cat_ious = []
            cat_precisions = []
            cat_recalls = []
            cat_accuracies = []

            for img_file in images:
                lbl_file = folder_path / f"{img_file.stem}_label.bmp"
                if not lbl_file.is_file():
                    raise FileNotFoundError(f"Mask missing for image: {img_file}")

                # 1. Load image
                pil_img = Image.open(img_file).convert("RGB")
                orig_w, orig_h = pil_img.size

                # 2. Preprocess
                input_tensor = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(device)

                # 3. Model forward pass
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                solar_prob = probs[0, 1].cpu().numpy()
                pred_binary = (solar_prob >= threshold).astype(np.uint8)

                # 4. Load ground truth mask & convert > 0 to 1
                pil_lbl = Image.open(lbl_file)
                gt_raw = np.array(pil_lbl)
                gt_binary = (gt_raw > 0).astype(np.uint8)

                # Resize if necessary (PV01 is already 256x256)
                if gt_binary.shape != (256, 256):
                    gt_img = Image.fromarray(gt_binary * 255).resize((256, 256), resample=Image.NEAREST)
                    gt_binary = (np.array(gt_img) > 127).astype(np.uint8)

                # 5. Compute confusion statistics
                tp = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
                tn = int(np.sum((pred_binary == 0) & (gt_binary == 0)))
                fp = int(np.sum((pred_binary == 1) & (gt_binary == 0)))
                fn = int(np.sum((pred_binary == 0) & (gt_binary == 1)))
                total_px = int(gt_binary.size)

                gt_solar_px = tp + fn
                pred_solar_px = tp + fp

                eps = 1e-7
                if gt_solar_px == 0 and pred_solar_px == 0:
                    dice = 1.0
                    iou = 1.0
                    precision = 1.0
                    recall = 1.0
                else:
                    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
                    iou = float(tp / (tp + fp + fn + eps))
                    precision = float(tp / (tp + fp + eps)) if pred_solar_px > 0 else 0.0
                    recall = float(tp / (tp + fn + eps)) if gt_solar_px > 0 else 0.0

                accuracy = float((tp + tn) / total_px)

                rec = {
                    "filename": img_file.name,
                    "category": cat_name,
                    "width": orig_w,
                    "height": orig_h,
                    "gt_solar_pixels": gt_solar_px,
                    "pred_solar_pixels": pred_solar_px,
                    "pixel_diff": pred_solar_px - gt_solar_px,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "accuracy": round(accuracy, 4),
                }
                records.append(rec)

                # Accumulate for category
                cat_tp += tp
                cat_tn += tn
                cat_fp += fp
                cat_fn += fn
                cat_dices.append(dice)
                cat_ious.append(iou)
                cat_precisions.append(precision)
                cat_recalls.append(recall)
                cat_accuracies.append(accuracy)

                total_processed += 1
                if total_processed % 100 == 0 or total_processed == 645:
                    print(f"Processed {total_processed} / 645 images...")

            # Category summary
            cat_dataset_dice = (2.0 * cat_tp) / (2.0 * cat_tp + cat_fp + cat_fn + eps)
            cat_dataset_iou = float(cat_tp) / (cat_tp + cat_fp + cat_fn + eps)
            cat_dataset_prec = float(cat_tp) / (cat_tp + cat_fp + eps) if (cat_tp + cat_fp) > 0 else 0.0
            cat_dataset_rec = float(cat_tp) / (cat_tp + cat_fn + eps) if (cat_tp + cat_fn) > 0 else 0.0

            category_summary[cat_name] = {
                "num_samples": len(images),
                "total_tp": cat_tp,
                "total_tn": cat_tn,
                "total_fp": cat_fp,
                "total_fn": cat_fn,
                "total_gt_solar_pixels": cat_tp + cat_fn,
                "total_pred_solar_pixels": cat_tp + cat_fp,
                "sample_mean_dice": round(float(np.mean(cat_dices)), 4),
                "sample_median_dice": round(float(np.median(cat_dices)), 4),
                "sample_std_dice": round(float(np.std(cat_dices)), 4),
                "sample_mean_iou": round(float(np.mean(cat_ious)), 4),
                "sample_median_iou": round(float(np.median(cat_ious)), 4),
                "sample_mean_precision": round(float(np.mean(cat_precisions)), 4),
                "sample_median_precision": round(float(np.median(cat_precisions)), 4),
                "sample_mean_recall": round(float(np.mean(cat_recalls)), 4),
                "sample_median_recall": round(float(np.median(cat_recalls)), 4),
                "sample_mean_accuracy": round(float(np.mean(cat_accuracies)), 4),
                "dataset_level_dice": round(float(cat_dataset_dice), 4),
                "dataset_level_iou": round(float(cat_dataset_iou), 4),
                "dataset_level_precision": round(float(cat_dataset_prec), 4),
                "dataset_level_recall": round(float(cat_dataset_rec), 4),
            }

    elapsed_sec = time.time() - start_eval_time

    # Overall dataset metrics
    df = pd.DataFrame(records)
    all_dices = df["dice"].tolist()
    all_ious = df["iou"].tolist()
    all_precs = df["precision"].tolist()
    all_recs = df["recall"].tolist()
    all_accs = df["accuracy"].tolist()

    tot_tp = int(df["tp"].sum())
    tot_tn = int(df["tn"].sum())
    tot_fp = int(df["fp"].sum())
    tot_fn = int(df["fn"].sum())
    tot_gt_px = tot_tp + tot_fn
    tot_pred_px = tot_tp + tot_fp

    tot_dataset_dice = (2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + eps)
    tot_dataset_iou = float(tot_tp) / (tot_tp + tot_fp + tot_fn + eps)
    tot_dataset_prec = float(tot_tp) / (tot_tp + tot_fp + eps) if tot_pred_px > 0 else 0.0
    tot_dataset_rec = float(tot_tp) / (tot_tp + tot_fn + eps) if tot_gt_px > 0 else 0.0

    overall_metrics = {
        "num_evaluated_images": len(records),
        "total_evaluation_time_sec": round(elapsed_sec, 2),
        "total_tp_pixels": tot_tp,
        "total_tn_pixels": tot_tn,
        "total_fp_pixels": tot_fp,
        "total_fn_pixels": tot_fn,
        "total_gt_solar_pixels": tot_gt_px,
        "total_pred_solar_pixels": tot_pred_px,
        "sample_mean_dice": round(float(np.mean(all_dices)), 4),
        "sample_median_dice": round(float(np.median(all_dices)), 4),
        "sample_std_dice": round(float(np.std(all_dices)), 4),
        "sample_min_dice": round(float(np.min(all_dices)), 4),
        "sample_max_dice": round(float(np.max(all_dices)), 4),
        "sample_mean_iou": round(float(np.mean(all_ious)), 4),
        "sample_median_iou": round(float(np.median(all_ious)), 4),
        "sample_mean_precision": round(float(np.mean(all_precs)), 4),
        "sample_median_precision": round(float(np.median(all_precs)), 4),
        "sample_mean_recall": round(float(np.mean(all_recs)), 4),
        "sample_median_recall": round(float(np.median(all_recs)), 4),
        "sample_mean_accuracy": round(float(np.mean(all_accs)), 4),
        "dataset_level_dice": round(float(tot_dataset_dice), 4),
        "dataset_level_iou": round(float(tot_dataset_iou), 4),
        "dataset_level_precision": round(float(tot_dataset_prec), 4),
        "dataset_level_recall": round(float(tot_dataset_rec), 4),
    }

    # Save per-image CSV
    csv_path = output_dir / "pv01_zero_shot_per_image_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-image metrics to: {csv_path}")

    # Save comprehensive JSON report
    report_data = {
        "task": "PV01 Zero-Shot Cross-Domain Evaluation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint_path),
        "model_architecture": "Half U-Net (2-class)",
        "dataset_evaluated": str(dataset_dir),
        "total_images": len(records),
        "overall_metrics": overall_metrics,
        "category_metrics": category_summary,
    }

    json_path = output_dir / "pv01_zero_shot_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved JSON report to: {json_path}")

    return report_data


if __name__ == "__main__":
    report = evaluate_pv01()
    print("\n" + "=" * 60)
    print("PV01 ZERO-SHOT EVALUATION COMPLETED")
    print("=" * 60)
    ov = report["overall_metrics"]
    print(f"Total Evaluated: {ov['num_evaluated_images']}")
    print(f"Sample Mean Dice / F1:     {ov['sample_mean_dice']:.4f}")
    print(f"Sample Median Dice / F1:   {ov['sample_median_dice']:.4f}")
    print(f"Sample Mean IoU:           {ov['sample_mean_iou']:.4f}")
    print(f"Sample Mean Precision:     {ov['sample_mean_precision']:.4f}")
    print(f"Sample Mean Recall:        {ov['sample_mean_recall']:.4f}")
    print(f"Dataset-level Dice / F1:   {ov['dataset_level_dice']:.4f}")
    print(f"Dataset-level IoU:         {ov['dataset_level_iou']:.4f}")
    print(f"Dataset-level Precision:   {ov['dataset_level_precision']:.4f}")
    print(f"Dataset-level Recall:      {ov['dataset_level_recall']:.4f}")
    print(f"Total FP Pixels:           {ov['total_fp_pixels']:,}")
    print(f"Total FN Pixels:           {ov['total_fn_pixels']:,}")
    print("-" * 60)
    for cat, c_met in report["category_metrics"].items():
        print(f"Category: {cat:<14} (N={c_met['num_samples']}) | Mean Dice: {c_met['sample_mean_dice']:.4f} | Dataset Dice: {c_met['dataset_level_dice']:.4f} | Prec: {c_met['dataset_level_precision']:.4f} | Rec: {c_met['dataset_level_recall']:.4f}")
    print("=" * 60)
