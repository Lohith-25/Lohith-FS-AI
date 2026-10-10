"""
SolarMap-India — Half U-Net Production Inference Engine CLI & Orchestrator.

Usage:
    python ml/inference/predict.py --image "path/to/image.png"
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
import torch

# Ensure local ml directory is importable
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analytics.components import (
    ComponentStatistics,
    analyze_components,
    compute_pixel_area_metrics,
)
from ml.inference.model import HalfUNet, get_device, load_halfunet_model
from ml.inference.postprocessing import (
    create_overlay,
    mask_to_visual_image,
    process_logits,
    resize_mask_to_original,
)
from ml.inference.preprocessing import (
    MODEL_INPUT_SIZE,
    load_and_validate_image,
    preprocess_image,
)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"


@dataclass
class InferenceResult:
    """Stores full inference result artifacts and derived pixel metrics."""
    image_path: Path
    original_size: tuple[int, int]
    model_name: str
    device: str
    input_size: tuple[int, int]
    solar_pixels: int
    total_pixels: int
    solar_coverage_percent: float
    inference_time_ms: float
    mask_path: Path
    overlay_path: Path
    binary_mask: np.ndarray
    solar_probabilities: np.ndarray
    component_stats: Optional[ComponentStatistics] = None
    pixel_area_metrics: Optional[Dict[str, Any]] = None

    @property
    def image_width(self) -> int:
        return self.original_size[0]

    @property
    def image_height(self) -> int:
        return self.original_size[1]

    @property
    def total_image_pixels(self) -> int:
        return self.total_pixels

    @property
    def solar_area_pixels(self) -> int:
        return self.solar_pixels

    @property
    def detected_region_count(self) -> int:
        if self.pixel_area_metrics:
            return self.pixel_area_metrics["detected_region_count"]
        if self.component_stats:
            return self.component_stats.filtered_count
        return 0

    @property
    def largest_region_area_pixels(self) -> Optional[int]:
        if self.pixel_area_metrics:
            return self.pixel_area_metrics["largest_region_area_pixels"]
        if self.component_stats:
            return self.component_stats.largest_area_pixels
        return None

    @property
    def smallest_region_area_pixels(self) -> Optional[int]:
        if self.pixel_area_metrics:
            return self.pixel_area_metrics["smallest_region_area_pixels"]
        if self.component_stats:
            return self.component_stats.smallest_area_pixels
        return None

    @property
    def mean_region_area_pixels(self) -> Optional[float]:
        if self.pixel_area_metrics:
            return self.pixel_area_metrics["mean_region_area_pixels"]
        if self.component_stats:
            return self.component_stats.mean_area_pixels
        return None

    @property
    def physical_area_m2(self) -> None:
        return None

    @property
    def physical_area_hectares(self) -> None:
        return None

    @property
    def physical_area_status(self) -> str:
        return "insufficient_data"

    def to_pixel_area_dict(self) -> Dict[str, Any]:
        """Returns the standardized Phase 1 pixel-area measurement dictionary."""
        if self.pixel_area_metrics is not None:
            return dict(self.pixel_area_metrics)
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "total_image_pixels": self.total_image_pixels,
            "solar_area_pixels": self.solar_area_pixels,
            "solar_coverage_percent": round(self.solar_coverage_percent, 4),
            "detected_region_count": self.detected_region_count,
            "largest_region_area_pixels": self.largest_region_area_pixels,
            "smallest_region_area_pixels": self.smallest_region_area_pixels,
            "mean_region_area_pixels": self.mean_region_area_pixels,
            "physical_area_m2": self.physical_area_m2,
            "physical_area_hectares": self.physical_area_hectares,
            "physical_area_status": self.physical_area_status,
        }


def predict_image(
    image_path: Union[str, Path],
    model: Optional[HalfUNet] = None,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    min_component_area: int = 20,
) -> InferenceResult:
    """
    Executes the end-to-end Half U-Net segmentation pipeline on an input image.

    Args:
        image_path: Path to target aerial/satellite image.
        model: Pre-loaded HalfUNet instance (optional; loaded from checkpoint if None).
        checkpoint_path: Path to model weights if model is not pre-loaded.
        output_dir: Root directory for inference outputs (masks/ and overlays/).
        device: Execution device (CUDA/CPU).
        threshold: Foreground class probability threshold (default 0.5).
        min_component_area: Minimum pixel area threshold for connected solar regions (default 20).

    Returns:
        InferenceResult containing metrics, output paths, and predictions.
    """
    img_path = Path(image_path)
    out_dir = Path(output_dir)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"

    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device and Model Initialization
    if device is None:
        device = get_device()

    if model is None:
        model = load_halfunet_model(checkpoint_path=checkpoint_path, device=device)
    else:
        model = model.to(device)
        model.eval()

    # 2. Image Loading and Validation
    original_image, (orig_w, orig_h) = load_and_validate_image(img_path)

    # 3. Preprocessing (Resize to 256x256 -> ToTensor -> Batch dimension)
    input_tensor = preprocess_image(original_image, device=device)

    # 4. Model Forward Pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 5. Postprocessing (Softmax -> Probabilities -> Binary Mask -> Resize to Original)
    mask_256, solar_prob = process_logits(logits, threshold=threshold)
    final_binary_mask = resize_mask_to_original(mask_256, (orig_w, orig_h))

    # 6. Direct Derived Metrics (Strictly Pixel-Based)
    solar_pixels = int(np.count_nonzero(final_binary_mask == 1))
    total_pixels = int(orig_w * orig_h)
    coverage_percent = (solar_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # 7. Standardized Pixel-Based Solar Area Measurement & Connected Region Analysis
    pixel_area_metrics = compute_pixel_area_metrics(
        final_binary_mask,
        min_component_area=min_component_area,
        connectivity=8,
    )
    component_stats, _ = analyze_components(
        final_binary_mask,
        min_component_area=min_component_area,
        connectivity=8,
    )

    # 8. Generate and Save Visual Artifacts
    stem = img_path.stem
    mask_file = masks_dir / f"{stem}_mask.png"
    overlay_file = overlays_dir / f"{stem}_overlay.png"

    visual_mask_img = mask_to_visual_image(final_binary_mask)
    visual_mask_img.save(mask_file, format="PNG")

    overlay_img = create_overlay(original_image, final_binary_mask)
    overlay_img.save(overlay_file, format="PNG")

    return InferenceResult(
        image_path=img_path,
        original_size=(orig_w, orig_h),
        model_name="Half U-Net",
        device=str(device),
        input_size=MODEL_INPUT_SIZE,
        solar_pixels=solar_pixels,
        total_pixels=total_pixels,
        solar_coverage_percent=coverage_percent,
        inference_time_ms=elapsed_ms,
        mask_path=mask_file,
        overlay_path=overlay_file,
        binary_mask=final_binary_mask,
        solar_probabilities=solar_prob,
        component_stats=component_stats,
        pixel_area_metrics=pixel_area_metrics,
    )


def print_summary(result: InferenceResult) -> None:
    """Formats and prints inference execution summary."""
    orig_w, orig_h = result.original_size
    in_w, in_h = result.input_size
    print("-" * 50)
    print("SolarMap-India Inference")
    print("-" * 50)
    print(f"Image:            {result.image_path.name}")
    print(f"Original Size:    {orig_w}x{orig_h}")
    print(f"Model:            {result.model_name}")
    print(f"Device:           {result.device}")
    print(f"Input Size:       {in_w}x{in_h}")
    print(f"Solar Pixels:     {result.solar_pixels:,} / {result.total_pixels:,}")
    print(f"Solar Coverage:   {result.solar_coverage_percent:.4f}% (pixel coverage)")
    if result.component_stats is not None:
        stats = result.component_stats
        print(f"Solar Regions:    {stats.filtered_count} (connected solar regions, min area >= {stats.min_component_area_pixels} px)")
        if stats.filtered_count > 0:
            print(f"Largest Region:   {stats.largest_area_pixels:,} px")
            print(f"Smallest Region:  {stats.smallest_area_pixels:,} px")
            print(f"Mean Region Area: {stats.mean_area_pixels:.2f} px")
    print(f"Physical Area:    status={result.physical_area_status} (unverified spatial scale)")
    print(f"Inference Time:   {result.inference_time_ms:.2f} ms")
    print(f"Mask:             {result.mask_path.resolve()}")
    print(f"Overlay:          {result.overlay_path.resolve()}")
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolarMap-India — Half U-Net Production Inference Engine"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to Half U-Net model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output masks and overlays (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference: 'cuda', 'cpu', or None for auto-detect",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Foreground solar class probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=20,
        help="Minimum pixel area threshold for connected solar regions (default: 20)",
    )

    args = parser.parse_args()

    try:
        result = predict_image(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device=get_device(args.device) if args.device else None,
            threshold=args.threshold,
            min_component_area=args.min_component_area,
        )
        print_summary(result)
    except Exception as exc:
        print(f"\n[ERROR] Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

