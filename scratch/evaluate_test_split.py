"""
SolarMap-India — Full Evaluation of Production Half U-Net on Held-Out Test Split.
Evaluates the locked 375-image held-out test split from split_image_ids.json.
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

from ml.inference.model import HalfUNet, get_device, load_halfunet_model

PROJECT_ROOT = Path("d:/SolarMap-India")
SPLIT_FILE = PROJECT_ROOT / "outputs" / "HalfUNet" / "metrics" / "split_image_ids.json"
ANNOTATIONS_FILE = PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "annotations" / "merged_instances_default.json"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images" / "default"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"

OUT_CSV = PROJECT_ROOT / "scratch" / "test_split_per_image_metrics.csv"
OUT_JSON = PROJECT_ROOT / "scratch" / "test_split_evaluation_summary.json"


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


def evaluate():
    device = get_device()
    print(f"Using device: {device}")
    model = load_halfunet_model(CHECKPOINT_PATH, device=device).eval()

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    test_ids = splits["test"]
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])

    # Overlap assertions
    assert len(set(test_ids) & train_ids) == 0, "Test set overlaps with train set!"
    assert len(set(test_ids) & val_ids) == 0, "Test set overlaps with val set!"

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    coco_images = {x["id"]: x for x in coco["images"]}
    coco_anns = {}
    for a in coco["annotations"]:
        coco_anns.setdefault(a["image_id"], []).append(a)

    print(f"Evaluating {len(test_ids)} held-out test images...")

    results = []
    tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    t0 = time.time()
    for idx, tid in enumerate(test_ids):
        img_info = coco_images.get(tid)
        if img_info is None:
            raise ValueError(f"COCO metadata missing for image ID {tid}")

        fn = os.path.basename(img_info["file_name"])
        img_path = IMAGES_DIR / fn
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Rasterize GT mask
        anns = coco_anns.get(tid, [])
        all_segs = [a["segmentation"] for a in anns]
        gt_mask = rasterize_polygons(all_segs, orig_h, orig_w)

        # Run inference via 256x256 tensor
        input_t = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_t)
            probs = torch.softmax(logits, dim=1)
            pred_256 = (probs[0, 1] >= 0.5).cpu().numpy().astype(np.uint8)

        # Restore prediction mask to original 640x640 using nearest neighbor
        pred_mask = cv2.resize(pred_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Pixel metrics
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
        total_px = orig_w * orig_h
        eps = 1e-7

        if gt_solar == 0 and pred_solar == 0:
            dice = 1.0
            iou = 1.0
            prec = 1.0
            rec = 1.0
        elif gt_solar == 0 and pred_solar > 0:
            dice = 0.0
            iou = 0.0
            prec = 0.0
            rec = 1.0  # or 0.0 depending on convention
        elif gt_solar > 0 and pred_solar == 0:
            dice = 0.0
            iou = 0.0
            prec = 0.0
            rec = 0.0
        else:
            dice = float((2.0 * tp) / (2.0 * tp + fp + fn_val + eps))
            iou = float(tp / (tp + fp + fn_val + eps))
            prec = float(tp / (tp + fp + eps))
            rec = float(tp / (tp + fn_val + eps))

        acc = float((tp + tn) / total_px)
        gt_cov = (gt_solar / total_px) * 100.0
        pred_cov = (pred_solar / total_px) * 100.0

        results.append({
            "test_index": idx + 1,
            "sample_id": tid,
            "file_name": fn,
            "width": orig_w,
            "height": orig_h,
            "annotation_count": len(anns),
            "gt_solar_pixels": gt_solar,
            "pred_solar_pixels": pred_solar,
            "gt_coverage_percent": round(gt_cov, 4),
            "pred_coverage_percent": round(pred_cov, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn_val,
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "accuracy": round(acc, 4),
        })

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.2f}s ({elapsed/len(test_ids):.4f}s/image)")

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"Per-image metrics saved to: {OUT_CSV}")

    # Micro metrics (pooled pixel confusion matrix across all test images)
    eps = 1e-7
    micro_dice = float((2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + eps))
    micro_iou = float(tot_tp / (tot_tp + tot_fp + tot_fn + eps))
    micro_prec = float(tot_tp / (tot_tp + tot_fp + eps))
    micro_rec = float(tot_tp / (tot_tp + tot_fn + eps))
    micro_acc = float((tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn))

    # Macro metrics across all 375 images
    macro_dice_all = float(df["dice"].mean())
    macro_iou_all = float(df["iou"].mean())
    macro_prec_all = float(df["precision"].mean())
    macro_rec_all = float(df["recall"].mean())

    median_dice_all = float(df["dice"].median())
    median_iou_all = float(df["iou"].median())

    # Macro metrics on 373 positive images (GT > 0)
    pos_df = df[df["gt_solar_pixels"] > 0]
    macro_dice_pos = float(pos_df["dice"].mean())
    macro_iou_pos = float(pos_df["iou"].mean())
    macro_prec_pos = float(pos_df["precision"].mean())
    macro_rec_pos = float(pos_df["recall"].mean())

    median_dice_pos = float(pos_df["dice"].median())
    median_iou_pos = float(pos_df["iou"].median())

    # Two-class Macro Dice (averaging Background Dice and Solar Dice as in original notebook)
    # Background Dice per image: 2*TN / (2*TN + FP + FN)
    bg_dices = []
    for r in results:
        bg_d = (2.0 * r["tn"]) / (2.0 * r["tn"] + r["fp"] + r["fn"] + eps)
        bg_dices.append(bg_d)
    two_class_macro_dice = float(np.mean([(d + b) / 2.0 for d, b in zip(df["dice"], bg_dices)]))

    summary = {
        "dataset": "SolarMap-India",
        "checkpoint": str(CHECKPOINT_PATH),
        "test_split_size": len(test_ids),
        "positive_test_images": len(pos_df),
        "zero_solar_control_images": len(test_ids) - len(pos_df),
        "elapsed_seconds": round(elapsed, 2),
        "micro_pixel_metrics": {
            "total_tp": tot_tp,
            "total_tn": tot_tn,
            "total_fp": tot_fp,
            "total_fn": tot_fn,
            "dice": round(micro_dice, 4),
            "iou": round(micro_iou, 4),
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "accuracy": round(micro_acc, 4),
            "total_gt_solar_pixels": tot_tp + tot_fn,
            "total_pred_solar_pixels": tot_tp + tot_fp,
            "pooled_gt_coverage_percent": round(((tot_tp + tot_fn) / (tot_tp + tot_tn + tot_fp + tot_fn)) * 100.0, 4),
            "pooled_pred_coverage_percent": round(((tot_tp + tot_fp) / (tot_tp + tot_tn + tot_fp + tot_fn)) * 100.0, 4),
        },
        "macro_sample_metrics_positive_only": {
            "mean_foreground_dice": round(macro_dice_pos, 4),
            "median_foreground_dice": round(median_dice_pos, 4),
            "mean_foreground_iou": round(macro_iou_pos, 4),
            "median_foreground_iou": round(median_iou_pos, 4),
            "mean_precision": round(macro_prec_pos, 4),
            "mean_recall": round(macro_rec_pos, 4),
        },
        "macro_sample_metrics_all_375": {
            "mean_foreground_dice": round(macro_dice_all, 4),
            "median_foreground_dice": round(median_dice_all, 4),
            "mean_foreground_iou": round(macro_iou_all, 4),
            "median_foreground_iou": round(median_iou_all, 4),
            "mean_precision": round(macro_prec_all, 4),
            "mean_recall": round(macro_rec_all, 4),
        },
        "notebook_two_class_macro_dice": round(two_class_macro_dice, 4),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Test Images:                   {len(test_ids)}")
    print(f"Positive Test Images:                {len(pos_df)}")
    print(f"Zero-Solar Control Images:           {len(test_ids) - len(pos_df)}")
    print(f"Micro Foreground Dice:               {micro_dice:.4f}")
    print(f"Micro Foreground IoU:                {micro_iou:.4f}")
    print(f"Micro Precision:                     {micro_prec:.4f}")
    print(f"Micro Recall:                        {micro_rec:.4f}")
    print(f"Total True Positive Pixels:          {tot_tp:,}")
    print(f"Total False Positive Pixels:         {tot_fp:,}")
    print(f"Total False Negative Pixels:         {tot_fn:,}")
    print(f"Macro Positive Mean Foreground Dice: {macro_dice_pos:.4f}")
    print(f"Macro Positive Median FG Dice:       {median_dice_pos:.4f}")
    print(f"Macro Positive Mean Foreground IoU:  {macro_iou_pos:.4f}")
    print(f"Macro Positive Median FG IoU:        {median_iou_pos:.4f}")
    print(f"Notebook Two-Class Macro Dice:       {two_class_macro_dice:.4f}")
    print("====================================================")


if __name__ == "__main__":
    evaluate()
