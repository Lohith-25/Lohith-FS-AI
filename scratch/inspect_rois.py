"""
SolarMap-India — ROI Deep-Dive on 637x463 Screenshot.
Extracts cropped comparisons for:
1. Actual visible solar panel arrays (missed or weak).
2. False positive detections (roof tiles, chimneys, vents, ground/roads).
3. Shadow and vegetation regions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path("d:/SolarMap-India")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ml.inference.postprocessing import create_overlay

INPUT_IMAGE_PATH = PROJECT_ROOT / "backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs/probability_investigation/rois"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original image and raw tiled prob map
original_img = Image.open(INPUT_IMAGE_PATH).convert("RGB")
img_np = np.array(original_img)
h, w, _ = img_np.shape

tiled_prob = np.load(PROJECT_ROOT / "outputs/probability_investigation/tiled_prob_map.npz")["prob"]
lb_prob = np.load(PROJECT_ROOT / "outputs/probability_investigation/letterbox_prob_map.npz")["prob"]

# Define 6 Key ROIs to inspect:
# ROI 1: Real Solar Array (Large vertical array on House 3: x=225..275, y=95..185)
# ROI 2: Real Solar Array (Horizontal array on House 4: x=300..350, y=300..350)
# ROI 3: Real Solar Array (East-facing array on House 1: x=30..65, y=60..105)
# ROI 4: False Positive (Reddish brown roof section on House 5: x=465..535, y=220..265)
# ROI 5: False Positive (Roof peak / skylight / vent on House 2: x=140..180, y=295..325)
# ROI 6: False Positive / Vegetation / Road (Street & trees: x=300..380, y=0..60)

rois = [
    {
        "id": "roi1_real_solar_house3",
        "name": "Actual Solar Array (House 3, Large Vertical PV)",
        "type": "real_solar_missed",
        "box": (220, 95, 280, 185), # (x1, y1, x2, y2)
    },
    {
        "id": "roi2_real_solar_house4",
        "name": "Actual Solar Array (House 4, Horizontal PV)",
        "type": "real_solar_missed",
        "box": (295, 295, 355, 355),
    },
    {
        "id": "roi3_real_solar_house1",
        "name": "Actual Solar Array (House 1, East-facing PV)",
        "type": "real_solar_partially_detected",
        "box": (25, 60, 75, 110),
    },
    {
        "id": "roi4_fp_red_roof_house5",
        "name": "False Positive (House 5, Red/Brown Shingles)",
        "type": "false_positive",
        "box": (465, 220, 535, 265),
    },
    {
        "id": "roi5_fp_vent_dormer_house2",
        "name": "False Positive (House 2, Dormer / Flat Shingle Patch)",
        "type": "false_positive",
        "box": (135, 295, 185, 330),
    },
    {
        "id": "roi6_road_and_vegetation",
        "name": "Road, Trees & Curbs",
        "type": "background_control",
        "box": (300, 5, 380, 65),
    },
]

print("Analyzing ROIs on 637x463 Satellite Screenshot...")

roi_summary = []

for r in rois:
    x1, y1, x2, y2 = r["box"]
    rgb_crop = img_np[y1:y2, x1:x2]
    tiled_p_crop = tiled_prob[y1:y2, x1:x2]
    lb_p_crop = lb_prob[y1:y2, x1:x2]
    
    tiled_mask_05_crop = (tiled_p_crop >= 0.50).astype(np.uint8)
    lb_mask_05_crop = (lb_p_crop >= 0.50).astype(np.uint8)
    
    mean_p_tiled = float(tiled_p_crop.mean())
    max_p_tiled = float(tiled_p_crop.max())
    min_p_tiled = float(tiled_p_crop.min())
    p_tiled_over_05 = int(np.sum(tiled_mask_05_crop))
    total_roi_px = int((x2 - x1) * (y2 - y1))
    
    mean_p_lb = float(lb_p_crop.mean())
    max_p_lb = float(lb_p_crop.max())
    p_lb_over_05 = int(np.sum(lb_mask_05_crop))

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=150)
    axes[0].imshow(rgb_crop)
    axes[0].set_title(f"RGB ({r['name']})", fontsize=9, fontweight="bold")
    axes[0].axis("off")
    
    im1 = axes[1].imshow(tiled_p_crop, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Tiled Prob (Mean: {mean_p_tiled:.2f}, Max: {max_p_tiled:.2f})", fontsize=9)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(lb_p_crop, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Letterbox Prob (Mean: {mean_p_lb:.2f}, Max: {max_p_lb:.2f})", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Overlay with Tiled Mask
    roi_overlay = rgb_crop.copy()
    roi_overlay[tiled_mask_05_crop == 1] = (
        roi_overlay[tiled_mask_05_crop == 1] * 0.5 + np.array([255, 60, 60]) * 0.5
    ).astype(np.uint8)
    axes[3].imshow(roi_overlay)
    axes[3].set_title(f"Tiled Mask (Thr=0.50): {p_tiled_over_05}/{total_roi_px} px", fontsize=9)
    axes[3].axis("off")

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{r['id']}.png"
    fig.savefig(save_path)
    plt.close(fig)

    entry = {
        "id": r["id"],
        "name": r["name"],
        "type": r["type"],
        "box": r["box"],
        "total_pixels": total_roi_px,
        "tiled": {
            "mean_prob": round(mean_p_tiled, 4),
            "max_prob": round(max_p_tiled, 4),
            "min_prob": round(min_p_tiled, 4),
            "positive_pixels_05": p_tiled_over_05,
            "coverage_percent": round(100.0 * p_tiled_over_05 / total_roi_px, 2),
        },
        "letterbox": {
            "mean_prob": round(mean_p_lb, 4),
            "max_prob": round(max_p_lb, 4),
            "positive_pixels_05": p_lb_over_05,
            "coverage_percent": round(100.0 * p_lb_over_05 / total_roi_px, 2),
        },
        "artifact": str(save_path.relative_to(PROJECT_ROOT)),
    }
    roi_summary.append(entry)
    print(f"[{r['type']}] {r['name']}:")
    print(f"  Tiled: mean={mean_p_tiled:.3f}, max={max_p_tiled:.3f}, pos={p_tiled_over_05}/{total_roi_px}")
    print(f"  Letterbox: mean={mean_p_lb:.3f}, max={max_p_lb:.3f}, pos={p_lb_over_05}/{total_roi_px}")

import json
with open(OUTPUT_DIR / "roi_summary.json", "w") as f:
    json.dump(roi_summary, f, indent=2)
print("ROI Analysis complete!")
