"""
SolarMap-India — Phase 2: Group A (Training) vs Group B (Held-Out Test) Evaluation.
Evaluates the production Half U-Net checkpoint on both splits using identical conventions:
- Target resolution 256x256, PyTorch evaluation mode.
- Threshold tau = 0.50.
- Bilinear probability restoration to original 640x640.
- Micro and Macro Dice, IoU, Precision, Recall.
- Per-image distributions (min, 25%, median, mean, 75%, max, std).
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path("d:/SolarMap-India")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

from ml.inference.model import get_device, load_halfunet_model

SPLIT_FILE = PROJECT_ROOT / "outputs/HalfUNet/metrics/split_image_ids.json"
ANNOTATIONS_FILE = PROJECT_ROOT / "dataset/Solar Images/Solar Images/annotations/merged_instances_default.json"
IMAGES_DIR = PROJECT_ROOT / "dataset/Solar Images/Solar Images/images/default"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/HalfUNet/models/halfunet_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs/leakage_and_split_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rasterize_polygons(segmentations: List[Any], height: int, width: int) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    if not segmentations:
        return np.array(mask_img, dtype=np.uint8)

    polys = []
    if isinstance(segmentations, list) and len(segmentations) > 0 and isinstance(segmentations[0], (int, float)):
        polys.append(segmentations)
    elif isinstance(segmentations, list):
        for item in segmentations:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], (int, float)):
                polys.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, list) and len(sub) > 0 and isinstance(sub[0], (int, float)):
                        polys.append(sub)

    for poly in polys:
        if len(poly) < 6:
            continue
        xy = list(zip(poly[0::2], poly[1::2]))
        draw.polygon(xy, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)


def evaluate_split(split_name: str, image_ids: List[int], model, device, coco_images, coco_anns):
    print(f"\n--- Evaluating {split_name.upper()} Split ({len(image_ids)} images) ---")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0
    per_image_results = []
    eps = 1e-7

    t0 = time.time()
    for idx, tid in enumerate(image_ids):
        info = coco_images[tid]
        fn = os.path.basename(info["file_name"])
        img_path = IMAGES_DIR / fn

        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # GT mask
        anns = coco_anns.get(tid, [])
        all_segs = [a["segmentation"] for a in anns]
        gt_mask = rasterize_polygons(all_segs, orig_h, orig_w)

        # Forward pass
        input_t = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_t)
            prob_256 = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)

        prob_orig = cv2.resize(prob_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pred_mask = (prob_orig >= 0.50).astype(np.uint8)

        tp = int(np.sum((pred_mask == 1) & (gt_mask == 1)))
        tn = int(np.sum((pred_mask == 0) & (gt_mask == 0)))
        fp = int(np.sum((pred_mask == 1) & (gt_mask == 0)))
        fn_val = int(np.sum((pred_mask == 0) & (gt_mask == 1)))

        tot_tp += tp
        tot_tn += tn
        tot_fp += fp
        tot_fn += fn_val

        gt_solar = tp + fn_val
        pred_solar = tp + fp

        if gt_solar == 0 and pred_solar == 0:
            dice, iou, prec, rec = 1.0, 1.0, 1.0, 1.0
        elif gt_solar == 0 and pred_solar > 0:
            dice, iou, prec, rec = 0.0, 0.0, 0.0, 1.0
        elif gt_solar > 0 and pred_solar == 0:
            dice, iou, prec, rec = 0.0, 0.0, 0.0, 0.0
        else:
            dice = float((2.0 * tp) / (2.0 * tp + fp + fn_val + eps))
            iou = float(tp / (tp + fp + fn_val + eps))
            prec = float(tp / (tp + fp + eps))
            rec = float(tp / (tp + fn_val + eps))

        per_image_results.append({
            "image_id": tid,
            "file_name": fn,
            "gt_solar_pixels": gt_solar,
            "pred_solar_pixels": pred_solar,
            "tp": tp,
            "fp": fp,
            "fn": fn_val,
            "tn": tn,
            "dice": dice,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "has_gt_solar": bool(gt_solar > 0),
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(image_ids):
            print(f"  Processed {idx + 1}/{len(image_ids)} images ({time.time() - t0:.1f}s)...")

    # Aggregate metrics
    micro_dice = float((2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + eps))
    micro_iou = float(tot_tp / (tot_tp + tot_fp + tot_fn + eps))
    micro_prec = float(tot_tp / (tot_tp + tot_fp + eps))
    micro_rec = float(tot_tp / (tot_tp + tot_fn + eps))
    micro_acc = float((tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn))

    # Positive-only images for macro
    pos_dices = [r["dice"] for r in per_image_results if r["has_gt_solar"]]
    pos_ious = [r["iou"] for r in per_image_results if r["has_gt_solar"]]
    pos_precs = [r["precision"] for r in per_image_results if r["has_gt_solar"]]
    pos_recs = [r["recall"] for r in per_image_results if r["has_gt_solar"]]

    dist = {
        "dice": {
            "mean": round(float(np.mean(pos_dices)), 4),
            "std": round(float(np.std(pos_dices)), 4),
            "min": round(float(np.min(pos_dices)), 4),
            "p25": round(float(np.percentile(pos_dices, 25)), 4),
            "median": round(float(np.median(pos_dices)), 4),
            "p75": round(float(np.percentile(pos_dices, 75)), 4),
            "max": round(float(np.max(pos_dices)), 4),
        },
        "iou": {
            "mean": round(float(np.mean(pos_ious)), 4),
            "std": round(float(np.std(pos_ious)), 4),
            "min": round(float(np.min(pos_ious)), 4),
            "p25": round(float(np.percentile(pos_ious, 25)), 4),
            "median": round(float(np.median(pos_ious)), 4),
            "p75": round(float(np.percentile(pos_ious, 75)), 4),
            "max": round(float(np.max(pos_ious)), 4),
        },
        "precision": {
            "mean": round(float(np.mean(pos_precs)), 4),
            "std": round(float(np.std(pos_precs)), 4),
            "median": round(float(np.median(pos_precs)), 4),
        },
        "recall": {
            "mean": round(float(np.mean(pos_recs)), 4),
            "std": round(float(np.std(pos_recs)), 4),
            "median": round(float(np.median(pos_recs)), 4),
        },
    }

    summary = {
        "split_name": split_name,
        "image_count": len(image_ids),
        "positive_images_count": len(pos_dices),
        "zero_annotation_images_count": len(image_ids) - len(pos_dices),
        "elapsed_seconds": round(time.time() - t0, 2),
        "micro_pixel_metrics": {
            "dice": round(micro_dice, 4),
            "iou": round(micro_iou, 4),
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "accuracy": round(micro_acc, 4),
            "total_tp": tot_tp,
            "total_fp": tot_fp,
            "total_fn": tot_fn,
            "total_tn": tot_tn,
        },
        "macro_distributions_positive_only": dist,
    }

    # Save per-image CSV
    csv_path = OUTPUT_DIR / f"{split_name}_per_image_metrics.csv"
    pd.DataFrame(per_image_results).to_csv(csv_path, index=False)
    print(f"Saved {split_name} CSV: {csv_path}")

    return summary


def main():
    device = get_device()
    print(f"Device: {device}")
    model = load_halfunet_model(CHECKPOINT_PATH, device=device).eval()

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    train_ids = splits["train"]
    test_ids = splits["test"]

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)
    coco_images = {x["id"]: x for x in coco["images"]}
    coco_anns = {}
    for a in coco["annotations"]:
        coco_anns.setdefault(a["image_id"], []).append(a)

    train_summary = evaluate_split("group_a_train", train_ids, model, device, coco_images, coco_anns)
    test_summary = evaluate_split("group_b_test", test_ids, model, device, coco_images, coco_anns)

    comparison = {
        "production_checkpoint": str(CHECKPOINT_PATH),
        "group_a_train": train_summary,
        "group_b_heldout_test": test_summary,
        "generalization_gap": {
            "micro_dice_gap": round(train_summary["micro_pixel_metrics"]["dice"] - test_summary["micro_pixel_metrics"]["dice"], 4),
            "micro_iou_gap": round(train_summary["micro_pixel_metrics"]["iou"] - test_summary["micro_pixel_metrics"]["iou"], 4),
            "micro_precision_gap": round(train_summary["micro_pixel_metrics"]["precision"] - test_summary["micro_pixel_metrics"]["precision"], 4),
            "micro_recall_gap": round(train_summary["micro_pixel_metrics"]["recall"] - test_summary["micro_pixel_metrics"]["recall"], 4),
            "macro_dice_gap": round(train_summary["macro_distributions_positive_only"]["dice"]["mean"] - test_summary["macro_distributions_positive_only"]["dice"]["mean"], 4),
        }
    }

    out_json = OUTPUT_DIR / "group_a_vs_group_b_comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n==================================================================")
    print("PHASE 2 SUMMARY: GROUP A (TRAIN) vs GROUP B (TEST)")
    print("==================================================================")
    print(f"Group A (Train, N={len(train_ids)}):")
    print(f"  Micro Dice: {train_summary['micro_pixel_metrics']['dice']:.4f} | IoU: {train_summary['micro_pixel_metrics']['iou']:.4f} | Prec: {train_summary['micro_pixel_metrics']['precision']:.4f} | Rec: {train_summary['micro_pixel_metrics']['recall']:.4f}")
    print(f"  Macro Dice: {train_summary['macro_distributions_positive_only']['dice']['mean']:.4f} (std={train_summary['macro_distributions_positive_only']['dice']['std']:.4f}, median={train_summary['macro_distributions_positive_only']['dice']['median']:.4f})")
    print(f"\nGroup B (Test, N={len(test_ids)}):")
    print(f"  Micro Dice: {test_summary['micro_pixel_metrics']['dice']:.4f} | IoU: {test_summary['micro_pixel_metrics']['iou']:.4f} | Prec: {test_summary['micro_pixel_metrics']['precision']:.4f} | Rec: {test_summary['micro_pixel_metrics']['recall']:.4f}")
    print(f"  Macro Dice: {test_summary['macro_distributions_positive_only']['dice']['mean']:.4f} (std={test_summary['macro_distributions_positive_only']['dice']['std']:.4f}, median={test_summary['macro_distributions_positive_only']['dice']['median']:.4f})")
    print(f"\nGeneralization Gap:")
    print(f"  Micro Dice Gap: {comparison['generalization_gap']['micro_dice_gap']:.4f} (Train {train_summary['micro_pixel_metrics']['dice']:.4f} -> Test {test_summary['micro_pixel_metrics']['dice']:.4f})")
    print(f"  Micro IoU Gap:  {comparison['generalization_gap']['micro_iou_gap']:.4f}")
    print(f"  Macro Dice Gap: {comparison['generalization_gap']['macro_dice_gap']:.4f}")


if __name__ == "__main__":
    main()
