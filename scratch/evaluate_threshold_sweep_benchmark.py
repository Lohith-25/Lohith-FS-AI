"""
SolarMap-India — Held-Out Test Split Threshold Sweep Benchmark.
Evaluates the locked 375-image held-out test split from split_image_ids.json
across thresholds 0.20, 0.30, 0.40, 0.50, 0.60.
Measures true Precision, Recall, IoU, and Dice without fabricating any data.
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
OUT_JSON = PROJECT_ROOT / "outputs/probability_investigation/benchmark_threshold_sweep.json"


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


def main():
    device = get_device()
    print(f"Loading model on device: {device}")
    model = load_halfunet_model(CHECKPOINT_PATH, device=device).eval()

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    test_ids = splits["test"]
    print(f"Loaded {len(test_ids)} test image IDs.")

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)
    coco_images = {x["id"]: x for x in coco["images"]}
    coco_anns = {}
    for a in coco["annotations"]:
        coco_anns.setdefault(a["image_id"], []).append(a)

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Pre-load or compute batch logits/probabilities to evaluate all thresholds efficiently
    # For 375 images, running inference once and caching probabilities in memory is very fast.
    print("Running forward passes on 375 test images...")
    t0 = time.time()
    
    cached_probs = []
    cached_gts = []

    for idx, tid in enumerate(test_ids):
        img_info = coco_images[tid]
        fn = os.path.basename(img_info["file_name"])
        img_path = IMAGES_DIR / fn
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # GT mask rasterization
        anns = coco_anns.get(tid, [])
        all_segs = [a["segmentation"] for a in anns]
        gt_mask = rasterize_polygons(all_segs, orig_h, orig_w)
        cached_gts.append(gt_mask)

        # Forward pass at 256x256
        input_t = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_t)
            probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
        
        # Resize probability map back to original image size (640x640) using bilinear interpolation
        prob_orig = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        cached_probs.append(prob_orig)

    elapsed_forward = time.time() - t0
    print(f"Forward pass completed in {elapsed_forward:.2f}s for 375 images.")

    # Evaluate across thresholds
    benchmark_results = {}
    eps = 1e-7

    for thr in thresholds:
        print(f"\nEvaluating threshold {thr:.2f}...")
        tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0
        sample_dices = []
        sample_ious = []
        sample_precs = []
        sample_recs = []

        for gt_mask, prob_orig in zip(cached_gts, cached_probs):
            pred_mask = (prob_orig >= thr).astype(np.uint8)

            tp = int(np.sum((pred_mask == 1) & (gt_mask == 1)))
            tn = int(np.sum((pred_mask == 0) & (gt_mask == 0)))
            fp = int(np.sum((pred_mask == 1) & (gt_mask == 0)))
            fn = int(np.sum((pred_mask == 0) & (gt_mask == 1)))

            tot_tp += tp
            tot_tn += tn
            tot_fp += fp
            tot_fn += fn

            gt_solar = tp + fn
            pred_solar = tp + fp

            if gt_solar == 0 and pred_solar == 0:
                dice = 1.0
                iou = 1.0
                prec = 1.0
                rec = 1.0
            elif gt_solar == 0 and pred_solar > 0:
                dice = 0.0
                iou = 0.0
                prec = 0.0
                rec = 1.0
            elif gt_solar > 0 and pred_solar == 0:
                dice = 0.0
                iou = 0.0
                prec = 0.0
                rec = 0.0
            else:
                dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
                iou = float(tp / (tp + fp + fn + eps))
                prec = float(tp / (tp + fp + eps))
                rec = float(tp / (tp + fn + eps))

            # Only track positive samples for macro foreground
            if gt_solar > 0:
                sample_dices.append(dice)
                sample_ious.append(iou)
                sample_precs.append(prec)
                sample_recs.append(rec)

        micro_dice = float((2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + eps))
        micro_iou = float(tot_tp / (tot_tp + tot_fp + tot_fn + eps))
        micro_prec = float(tot_tp / (tot_tp + tot_fp + eps))
        micro_rec = float(tot_tp / (tot_tp + tot_fn + eps))
        micro_acc = float((tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn))

        benchmark_results[f"{thr:.2f}"] = {
            "threshold": thr,
            "micro": {
                "dice": round(micro_dice, 4),
                "iou": round(micro_iou, 4),
                "precision": round(micro_prec, 4),
                "recall": round(micro_rec, 4),
                "accuracy": round(micro_acc, 4),
                "tp": tot_tp,
                "fp": tot_fp,
                "fn": tot_fn,
                "tn": tot_tn,
            },
            "macro_positive_images": {
                "mean_dice": round(float(np.mean(sample_dices)), 4),
                "median_dice": round(float(np.median(sample_dices)), 4),
                "mean_iou": round(float(np.mean(sample_ious)), 4),
                "median_iou": round(float(np.median(sample_ious)), 4),
                "mean_precision": round(float(np.mean(sample_precs)), 4),
                "mean_recall": round(float(np.mean(sample_recs)), 4),
            }
        }

        print(
            f"  Threshold {thr:.2f}: "
            f"Micro Dice={micro_dice:.4f}, IoU={micro_iou:.4f}, Prec={micro_prec:.4f}, Rec={micro_rec:.4f} | "
            f"Macro Dice={np.mean(sample_dices):.4f}, IoU={np.mean(sample_ious):.4f}"
        )

    with open(OUT_JSON, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved benchmark results to {OUT_JSON}")


if __name__ == "__main__":
    main()
