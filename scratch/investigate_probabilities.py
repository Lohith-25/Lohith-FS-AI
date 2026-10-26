"""
SolarMap-India — Deep Probability Map and Error Investigation on 637x463 Screenshot.
Tasks 1, 2, and 3: Probability analysis, heatmap generation, threshold sweep, and error diagnosis.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path("d:/SolarMap-India")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from ml.analytics.components import analyze_components, compute_pixel_area_metrics
from ml.inference.model import get_device, load_halfunet_model
from ml.inference.postprocessing import (
    create_overlay,
    mask_to_visual_image,
    process_logits,
    unpad_and_restore_mask,
    unpad_and_restore_probability_map,
)
from ml.inference.preprocessing import (
    MODEL_INPUT_SIZE,
    load_and_validate_image,
    preprocess_image_letterbox,
)
from ml.inference.tiling import TilingConfig, predict_tiled

INPUT_IMAGE_PATH = PROJECT_ROOT / "backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs/probability_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/HalfUNet/models/halfunet_best.pth"


def main():
    print("==================================================================")
    print("TASK 1 & 2: PROBABILITY MAP & THRESHOLD SENSITIVITY INVESTIGATION")
    print("==================================================================")

    device = get_device()
    print(f"Device: {device}")
    model = load_halfunet_model(checkpoint_path=CHECKPOINT_PATH, device=device)
    model.eval()

    # 1. Load image
    original_img, (orig_w, orig_h) = load_and_validate_image(INPUT_IMAGE_PATH)
    print(f"Loaded original image: {orig_w}x{orig_h} (width x height)")
    img_rgb = np.array(original_img)

    # -------------------------------------------------------------
    # 2. TILED INFERENCE PROBABILITY MAP
    # -------------------------------------------------------------
    print("\n--- Running Tiled Inference ---")
    tile_cfg = TilingConfig(threshold=0.50, tile_size=(256, 256), overlap_ratio=0.25)
    tiled_mask_05, tiled_prob, num_tiles = predict_tiled(
        image=original_img,
        model=model,
        device=device,
        config=tile_cfg,
    )
    print(f"Tiled inference evaluated {num_tiles} tiles.")
    print(f"Tiled prob shape: {tiled_prob.shape}, dtype={tiled_prob.dtype}")
    print(f"Tiled prob range: min={tiled_prob.min():.6f}, max={tiled_prob.max():.6f}, mean={tiled_prob.mean():.6f}")

    # Coordinate mapping verification
    assert tiled_prob.shape == (orig_h, orig_w), f"Tiled prob shape mismatch: {tiled_prob.shape} vs ({orig_h}, {orig_w})"
    assert tiled_mask_05.shape == (orig_h, orig_w), f"Tiled mask shape mismatch: {tiled_mask_05.shape} vs ({orig_h}, {orig_w})"
    print("Coordinate mapping verified: Probability map shape strictly matches image height x width (463 x 637).")

    # Save raw probability maps
    np.savez_compressed(OUTPUT_DIR / "tiled_prob_map.npz", prob=tiled_prob)

    # -------------------------------------------------------------
    # 3. LETTERBOX INFERENCE PROBABILITY MAP
    # -------------------------------------------------------------
    print("\n--- Running Letterbox Inference ---")
    input_tensor, lb_params = preprocess_image_letterbox(
        original_img,
        device=device,
        target_size=MODEL_INPUT_SIZE,
    )
    with torch.no_grad():
        lb_logits = model(input_tensor)
    lb_mask_256, lb_prob_256 = process_logits(lb_logits, threshold=0.50)
    lb_mask_05 = unpad_and_restore_mask(lb_mask_256, lb_params)
    lb_prob = unpad_and_restore_probability_map(lb_prob_256, lb_params)
    print(f"Letterbox prob shape: {lb_prob.shape}, dtype={lb_prob.dtype}")
    print(f"Letterbox prob range: min={lb_prob.min():.6f}, max={lb_prob.max():.6f}, mean={lb_prob.mean():.6f}")
    assert lb_prob.shape == (orig_h, orig_w), f"LB prob shape mismatch: {lb_prob.shape} vs ({orig_h}, {orig_w})"

    np.savez_compressed(OUTPUT_DIR / "letterbox_prob_map.npz", prob=lb_prob)

    # -------------------------------------------------------------
    # 4. PRODUCE CLEARLY LABELED HEATMAPS AND TRIPTYCHS
    # -------------------------------------------------------------
    print("\n--- Generating Labeled Heatmaps & Triptychs ---")

    for strat_name, prob_map, mask_05 in [("tiled", tiled_prob, tiled_mask_05), ("letterbox", lb_prob, lb_mask_05)]:
        # Standalone Heatmap with Colorbar
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        im = ax.imshow(prob_map, cmap="turbo", vmin=0.0, vmax=1.0)
        ax.set_title(f"Solar Panel Probability Map ({strat_name.capitalize()} Strategy)\nResolution: {orig_w}x{orig_h}", fontsize=13, fontweight="bold")
        ax.set_xlabel("X (Width pixels: 0 to 636)", fontsize=10)
        ax.set_ylabel("Y (Height pixels: 0 to 462)", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Foreground Solar Probability P(solar)", fontsize=10)
        heatmap_path = OUTPUT_DIR / f"{strat_name}_heatmap_labeled.png"
        fig.tight_layout()
        fig.savefig(heatmap_path)
        plt.close(fig)
        print(f"Saved: {heatmap_path}")

        # Triptych: Original Image | Heatmap | Binary Mask (thr=0.50) | Overlay
        overlay_05 = create_overlay(original_img, mask_05)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title(f"Original Satellite Screenshot ({orig_w}x{orig_h})", fontsize=11, fontweight="bold")
        axes[0, 0].axis("on")

        im1 = axes[0, 1].imshow(prob_map, cmap="turbo", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title(f"Probability Heatmap ({strat_name.capitalize()})", fontsize=11, fontweight="bold")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="P(solar)")

        axes[1, 0].imshow(mask_05, cmap="gray")
        axes[1, 0].set_title(f"Binary Mask (Threshold = 0.50)\nSolar Pixels: {int(np.sum(mask_05)):,}", fontsize=11, fontweight="bold")

        axes[1, 1].imshow(overlay_05)
        axes[1, 1].set_title(f"Overlay on Original Image\nCoverage: {100.0 * np.sum(mask_05) / (orig_w * orig_h):.2f}%", fontsize=11, fontweight="bold")

        fig.tight_layout()
        composite_path = OUTPUT_DIR / f"{strat_name}_quad_comparison.png"
        fig.savefig(composite_path)
        plt.close(fig)
        print(f"Saved: {composite_path}")

    # -------------------------------------------------------------
    # 5. THRESHOLD COMPARISON (0.20, 0.30, 0.40, 0.50, 0.60)
    # -------------------------------------------------------------
    print("\n==================================================================")
    print("TASK 2: THRESHOLD COMPARISON (0.20, 0.30, 0.40, 0.50, 0.60)")
    print("==================================================================")

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    total_pixels = orig_w * orig_h
    sweep_summary = {}

    for strat_name, prob_map in [("tiled", tiled_prob), ("letterbox", lb_prob)]:
        sweep_summary[strat_name] = []
        print(f"\n--- Strategy: {strat_name.upper()} ---")

        for thr in thresholds:
            binary_mask = (prob_map >= thr).astype(np.uint8)
            solar_px = int(np.sum(binary_mask == 1))
            coverage_pct = (solar_px / total_pixels) * 100.0

            # Component analysis at min_area=10 and min_area=20
            comp_10, _ = analyze_components(binary_mask, min_component_area=10, connectivity=8)
            comp_20, _ = analyze_components(binary_mask, min_component_area=20, connectivity=8)

            mask_path = OUTPUT_DIR / f"{strat_name}_thresh_{thr:.2f}_mask.png"
            overlay_path = OUTPUT_DIR / f"{strat_name}_thresh_{thr:.2f}_overlay.png"

            mask_img = mask_to_visual_image(binary_mask)
            mask_img.save(mask_path)

            overlay_img = create_overlay(original_img, binary_mask)
            overlay_img.save(overlay_path)

            entry = {
                "threshold": thr,
                "solar_pixels": solar_px,
                "solar_coverage_percent": round(coverage_pct, 4),
                "raw_connected_components": comp_10.raw_count,
                "filtered_components_min10": comp_10.filtered_count,
                "filtered_components_min20": comp_20.filtered_count,
                "largest_component_pixels": comp_20.largest_area_pixels,
                "mean_component_pixels": comp_20.mean_area_pixels,
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
            }
            sweep_summary[strat_name].append(entry)

            print(
                f"  Thr {thr:.2f}: {solar_px:6d} px ({coverage_pct:5.2f}%) | "
                f"Raw comps: {comp_10.raw_count:3d} | Filtered(>=20): {comp_20.filtered_count:2d} | "
                f"Largest: {comp_20.largest_area_pixels or 0:4d} px"
            )

    sweep_json_path = OUTPUT_DIR / "threshold_comparison_summary.json"
    with open(sweep_json_path, "w") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"\nSaved threshold summary: {sweep_json_path}")

    # -------------------------------------------------------------
    # 6. TASK 3: INSPECTION OF ROIs (FALSE POSITIVES & MISSED DETECTIONS)
    # -------------------------------------------------------------
    print("\n==================================================================")
    print("TASK 3: ROI REGION-LEVEL INSPECTION (PROBABILITIES IN DETAIL)")
    print("==================================================================")

    # Let's inspect probability percentiles and histograms
    print("\nTiled Probability Percentiles:")
    for p in [50, 75, 80, 85, 90, 95, 97, 98, 99, 99.5, 99.9, 100]:
        v = np.percentile(tiled_prob, p)
        print(f"  P{p:4.1f}: {v:.6f}")

    print("\nLetterbox Probability Percentiles:")
    for p in [50, 75, 80, 85, 90, 95, 97, 98, 99, 99.5, 99.9, 100]:
        v = np.percentile(lb_prob, p)
        print(f"  P{p:4.1f}: {v:.6f}")

    # Pixel count in various probability brackets for Tiled
    bins = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.001)]
    print("\nTiled Pixel Distribution across probability bins:")
    for b_low, b_high in bins:
        count = int(np.sum((tiled_prob >= b_low) & (tiled_prob < b_high)))
        pct = 100.0 * count / total_pixels
        print(f"  [{b_low:.2f}, {b_high:.2f}): {count:6d} px ({pct:5.2f}%)")

    # Extract connected components from Tiled mask at 0.50 to identify specific ROIs
    comp_stats_tiled, _ = analyze_components((tiled_prob >= 0.50).astype(np.uint8), min_component_area=20)
    print(f"\nTiled (thr=0.50, min_area=20) identified {len(comp_stats_tiled.regions)} regions:")
    for r in sorted(comp_stats_tiled.regions, key=lambda x: x.area_pixels, reverse=True)[:15]:
        # Sample probability statistics within this component bounding box
        roi_prob = tiled_prob[r.y : r.y + r.height, r.x : r.x + r.width]
        print(
            f"  Region #{r.id:2d}: Area={r.area_pixels:4d} px, BBox=({r.x}, {r.y}, {r.width}, {r.height}), "
            f"Prob: mean={roi_prob.mean():.3f}, max={roi_prob.max():.3f}"
        )


if __name__ == "__main__":
    main()
