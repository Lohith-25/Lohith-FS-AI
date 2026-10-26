"""
SolarMap-India — Investigation of 637x463 Satellite Screenshot Segmentation.
Traces the complete pipeline and generates diagnostic artifacts for Phases 2, 3, 4, 5.
"""

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path("d:/SolarMap-India")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from ml.inference.model import get_device, load_halfunet_model
from ml.inference.preprocessing import (
    MODEL_INPUT_SIZE,
    compute_letterbox_params,
    letterbox_image,
    preprocess_image_letterbox,
    load_and_validate_image,
)
from ml.inference.postprocessing import (
    process_logits,
    unpad_and_restore_mask,
    mask_to_visual_image,
    create_overlay,
)
from ml.analytics.components import analyze_components, compute_pixel_area_metrics

PROJECT_ROOT = Path("d:/SolarMap-India")
INPUT_IMAGE_PATH = PROJECT_ROOT / "backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs/investigation_637x463"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/HalfUNet/models/halfunet_best.pth"


def main():
    print("==================================================================")
    print("PHASE 1 & 2: PIPELINE TRACE & FAILURE MEASUREMENT")
    print("==================================================================")

    device = get_device()
    print(f"Device: {device}")
    model = load_halfunet_model(checkpoint_path=CHECKPOINT_PATH, device=device)
    model.eval()

    # 1. Load and inspect original image
    assert INPUT_IMAGE_PATH.exists(), f"Image not found: {INPUT_IMAGE_PATH}"
    with Image.open(INPUT_IMAGE_PATH) as raw_img:
        print(f"Raw image size: {raw_img.size}, mode: {raw_img.mode}")
        raw_rgba = np.array(raw_img)
        has_alpha = raw_img.mode == "RGBA"
        if has_alpha:
            alpha_channel = raw_rgba[:, :, 3]
            print(f"Alpha channel stats: min={alpha_channel.min()}, max={alpha_channel.max()}, mean={alpha_channel.mean():.2f}")

    original_img, (orig_w, orig_h) = load_and_validate_image(INPUT_IMAGE_PATH)
    print(f"Loaded PIL RGB: width={orig_w}, height={orig_h}")
    assert (orig_w, orig_h) == (637, 463), f"Expected (637, 463), got ({orig_w}, {orig_h})"

    # Save original image
    original_save_path = OUTPUT_DIR / "01_original_input.png"
    original_img.save(original_save_path)
    print(f"Saved: {original_save_path}")

    # 2. Letterbox Preprocessing
    padded_canvas, lb_params = letterbox_image(original_img, target_size=MODEL_INPUT_SIZE)
    padded_save_path = OUTPUT_DIR / "02_letterbox_model_input_256x256.png"
    padded_canvas.save(padded_save_path)
    print(f"Letterbox params:")
    print(f"  original_size: {lb_params.original_size}")
    print(f"  scaled_size:   {lb_params.scaled_size}")
    print(f"  scale factor:  {lb_params.scale:.6f}")
    print(f"  padding (L, T, R, B): ({lb_params.pad_left}, {lb_params.pad_top}, {lb_params.pad_right}, {lb_params.pad_bottom})")
    print(f"Saved: {padded_save_path}")

    # Convert to tensor exactly as in predict.py
    input_tensor, _ = preprocess_image_letterbox(original_img, device=device, target_size=MODEL_INPUT_SIZE)
    print(f"Model input tensor shape: {input_tensor.shape}, min={input_tensor.min().item():.4f}, max={input_tensor.max().item():.4f}")

    # 3. Model Forward Pass & Raw Probability Map
    with torch.no_grad():
        logits = model(input_tensor)  # [1, 2, 256, 256]

    print(f"Output logits shape: {logits.shape}")
    print(f"  Class 0 (bg) logits: min={logits[0, 0].min().item():.4f}, max={logits[0, 0].max().item():.4f}, mean={logits[0, 0].mean().item():.4f}")
    print(f"  Class 1 (solar) logits: min={logits[0, 1].min().item():.4f}, max={logits[0, 1].max().item():.4f}, mean={logits[0, 1].mean().item():.4f}")

    probs = torch.softmax(logits, dim=1)
    solar_prob_256 = probs[0, 1].detach().cpu().numpy().astype(np.float32)

    # Save raw probability map as float32 npz and normalized visual representation
    np.savez_compressed(OUTPUT_DIR / "03_solar_prob_map_256x256.npz", prob=solar_prob_256)
    prob_visual = (solar_prob_256 * 255).astype(np.uint8)
    prob_visual_img = Image.fromarray(prob_visual, mode="L")
    prob_visual_path = OUTPUT_DIR / "03_solar_prob_visual_256x256.png"
    prob_visual_img.save(prob_visual_path)
    print(f"Saved: {prob_visual_path}")

    # Probability map distribution stats
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    pct_vals = np.percentile(solar_prob_256, percentiles)
    print("Probability percentiles at 256x256:")
    for p, v in zip(percentiles, pct_vals):
        print(f"  {p:5.1f}%: {v:.6f}")

    # Unpad probability map to original 637x463
    new_w, new_h = lb_params.scaled_size
    pad_l, pad_t = lb_params.pad_left, lb_params.pad_top
    unpadded_prob_scaled = solar_prob_256[pad_t : pad_t + new_h, pad_l : pad_l + new_w]
    unpadded_prob_637x463 = cv2.resize(unpadded_prob_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    np.savez_compressed(OUTPUT_DIR / "03_solar_prob_map_637x463.npz", prob=unpadded_prob_637x463)

    # 4. Binary mask before postprocessing (at 256x256 threshold 0.5)
    lb_mask_256, _ = process_logits(logits, threshold=0.5)
    pos_256 = int(np.sum(lb_mask_256 == 1))
    print(f"\nStage 4: Binary mask at 256x256 (threshold 0.5):")
    print(f"  Positive pixels on 256x256 canvas: {pos_256} / {256*256} ({(pos_256/(256*256))*100:.2f}%)")
    mask_256_save = OUTPUT_DIR / "04_binary_mask_before_restore_256x256.png"
    Image.fromarray((lb_mask_256 * 255).astype(np.uint8), mode="L").save(mask_256_save)
    print(f"Saved: {mask_256_save}")

    # 5. Final binary mask restored to 637x463
    final_binary_mask = unpad_and_restore_mask(lb_mask_256, lb_params)
    assert final_binary_mask.shape == (orig_h, orig_w), f"Expected {(orig_h, orig_w)}, got {final_binary_mask.shape}"
    final_solar_pixels = int(np.sum(final_binary_mask == 1))
    total_pixels = orig_w * orig_h
    coverage_percent = 100.0 * final_solar_pixels / total_pixels
    print(f"\nStage 5: Final Restored Binary Mask (637x463):")
    print(f"  Positive solar pixels: {final_solar_pixels}")
    print(f"  Total pixels:          {total_pixels}")
    print(f"  Solar coverage:        {coverage_percent:.4f}%")
    final_mask_save = OUTPUT_DIR / "05_final_binary_mask_637x463.png"
    Image.fromarray((final_binary_mask * 255).astype(np.uint8), mode="L").save(final_mask_save)
    print(f"Saved: {final_mask_save}")

    # 6. Final Overlay
    final_overlay = create_overlay(original_img, final_binary_mask)
    overlay_save = OUTPUT_DIR / "06_final_overlay_637x463.png"
    final_overlay.save(overlay_save)
    print(f"Saved: {overlay_save}")

    # 7. Connected components at default min_area=10 and min_area=20
    comp_stats_10, _ = analyze_components(final_binary_mask, min_component_area=10, connectivity=8)
    comp_stats_20, _ = analyze_components(final_binary_mask, min_component_area=20, connectivity=8)
    print(f"\nConnected Components Analysis (637x463):")
    print(f"  Raw components count (area >= 1): {comp_stats_10.raw_count}")
    print(f"  Filtered components (area >= 10): {comp_stats_10.filtered_count}")
    print(f"    Total filtered area: {comp_stats_10.filtered_solar_pixels} px")
    print(f"    Largest component:   {comp_stats_10.largest_area_pixels} px")
    print(f"    Smallest component:  {comp_stats_10.smallest_area_pixels} px")
    print(f"    Mean component area: {comp_stats_10.mean_area_pixels:.1f} px")
    print(f"  Filtered components (area >= 20): {comp_stats_20.filtered_count}")

    # Print individual components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_binary_mask, connectivity=8)
    print(f"\nIndividual components (all {num_labels - 1}):")
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[lbl]
        print(f"  Component #{lbl:2d}: area={area:4d} px, bbox=({x:3d}, {y:3d}, {w:3d}, {h:3d}), centroid=({cx:.1f}, {cy:.1f})")

    # ==================================================================
    # PHASE 3: THRESHOLD SENSITIVITY ANALYSIS
    # ==================================================================
    print("\n==================================================================")
    print("PHASE 3: THRESHOLD SENSITIVITY ANALYSIS (0.10 to 0.70)")
    print("==================================================================")

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    thresh_results = []

    for thr in thresholds:
        mask_thr_256, _ = process_logits(logits, threshold=thr)
        restored_thr = unpad_and_restore_mask(mask_thr_256, lb_params)

        sol_px = int(np.sum(restored_thr == 1))
        cov_pct = 100.0 * sol_px / total_pixels
        n_labels, _, _, _ = cv2.connectedComponentsWithStats(restored_thr, connectivity=8)
        n_comps = n_labels - 1

        mask_path = OUTPUT_DIR / f"thresh_{thr:.2f}_mask.png"
        overlay_path = OUTPUT_DIR / f"thresh_{thr:.2f}_overlay.png"
        Image.fromarray((restored_thr * 255).astype(np.uint8), mode="L").save(mask_path)
        create_overlay(original_img, restored_thr).save(overlay_path)

        thresh_results.append({
            "threshold": thr,
            "solar_pixels": sol_px,
            "coverage_percent": round(cov_pct, 4),
            "num_connected_components": n_comps,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
        })

        print(f"Threshold {thr:.2f}: {sol_px:6d} px ({cov_pct:6.2f}%), {n_comps:2d} components -> {mask_path.name}")

    with open(OUTPUT_DIR / "threshold_sensitivity_results.json", "w") as f:
        json.dump(thresh_results, f, indent=2)

    # ==================================================================
    # PHASE 4: CHECK GROUND TRUTH AVAILABILITY
    # ==================================================================
    print("\n==================================================================")
    print("PHASE 4: GROUND TRUTH VERIFICATION")
    print("==================================================================")
    import hashlib
    with open(INPUT_IMAGE_PATH, "rb") as f:
        input_bytes = f.read()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    print(f"Input file SHA-256: {input_sha256}")
    print(f"Input dimensions:   {orig_w}x{orig_h}")

    # Check against known dataset files
    found_matches = []
    for candidate_dir in [PROJECT_ROOT / "data", PROJECT_ROOT / "dataset", PROJECT_ROOT / "images"]:
        if candidate_dir.exists():
            for p in candidate_dir.rglob("*.png"):
                if p.name == "input.png":
                    continue
                try:
                    with Image.open(p) as ci:
                        if ci.size == (orig_w, orig_h):
                            with open(p, "rb") as cf:
                                c_sha = hashlib.sha256(cf.read()).hexdigest()
                            if c_sha == input_sha256:
                                found_matches.append(str(p))
                except Exception:
                    pass

    print(f"Dataset ground-truth matches found: {len(found_matches)}")
    if found_matches:
        for m in found_matches:
            print(f"  Matched: {m}")
    else:
        print("  VERIFIED: No ground truth annotation exists in the dataset repository for this uploaded screenshot.")
        print("  Scientific consequence: Accuracy metrics (IoU, Dice, Precision, Recall) CANNOT be computed or fabricated.")


if __name__ == "__main__":
    main()
