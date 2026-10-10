"""
SolarMap-India — Solar Segmentation Analytics Engine & CLI.

Validates binary segmentation masks, computes pixel-level statistics,
extracts and filters connected components, and generates reports and visualizations.

Usage:
    python ml/analytics/mask_analysis.py --mask "outputs/inference/masks/768.0_1.0_mask.png"
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analytics.components import ComponentStatistics, analyze_components
from ml.analytics.report import build_report_dictionary, save_components_csv, save_json_report
from ml.analytics.visualization import create_colored_components_map, draw_bounding_boxes

DEFAULT_ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics"


@dataclass
class MaskPixelStats:
    """Stores verified pixel-level segmentation measurements."""
    width: int
    height: int
    total_pixels: int
    solar_pixels: int
    background_pixels: int
    solar_coverage_percent: float


def load_and_validate_mask(mask_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Loads and validates a binary segmentation mask.
    Normalizes {0, 255} to {0, 1}.

    Args:
        mask_path: Path to the mask image file.

    Returns:
        Tuple of (binary_mask array with values {0, 1}, (width, height)).

    Raises:
        FileNotFoundError: If mask file does not exist.
        ValueError: If mask is empty, corrupted, or contains invalid non-binary pixel values.
    """
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path.resolve()}")

    # Read as single-channel grayscale
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to decode mask image: {path.name}")

    h, w = mask.shape
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid mask dimensions: {w}x{h}")

    unique_vals = set(np.unique(mask))

    # Accept {0, 1} or {0, 255} or single-class {0} / {1} / {255}
    if unique_vals.issubset({0, 1}):
        binary_mask = mask.astype(np.uint8)
    elif unique_vals.issubset({0, 255}):
        # Normalize 255 -> 1
        binary_mask = (mask > 0).astype(np.uint8)
    else:
        raise ValueError(
            f"Segmentation mask contains invalid non-binary pixel values: {unique_vals}. "
            "Expected {0, 1} or {0, 255}."
        )

    return binary_mask, (w, h)


def compute_pixel_statistics(binary_mask: np.ndarray) -> MaskPixelStats:
    """
    Computes directly derived pixel measurements from the binary mask.

    Validations:
        - solar_pixels + background_pixels == total_pixels
        - 0.0 <= solar_coverage_percent <= 100.0
    """
    h, w = binary_mask.shape
    total_pixels = int(w * h)
    solar_pixels = int(np.count_nonzero(binary_mask == 1))
    background_pixels = total_pixels - solar_pixels

    if solar_pixels + background_pixels != total_pixels:
        raise ValueError(
            f"Pixel sum mismatch: solar ({solar_pixels}) + bg ({background_pixels}) != total ({total_pixels})"
        )

    coverage_percent = (solar_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    if not (0.0 <= coverage_percent <= 100.0):
        raise ValueError(f"Solar coverage out of valid range [0, 100]: {coverage_percent}")

    return MaskPixelStats(
        width=w,
        height=h,
        total_pixels=total_pixels,
        solar_pixels=solar_pixels,
        background_pixels=background_pixels,
        solar_coverage_percent=coverage_percent,
    )


def run_segmentation_analytics(
    mask_path: Union[str, Path],
    original_image_path: Optional[Union[str, Path]] = None,
    min_component_area: int = 20,
    output_dir: Union[str, Path] = DEFAULT_ANALYTICS_DIR,
) -> Tuple[dict, Path, Path]:
    """
    Executes the complete segmentation analytics workflow.

    Args:
        mask_path: Path to the binary segmentation mask.
        original_image_path: Optional path to corresponding raw RGB image for overlaying bounding boxes.
        min_component_area: Threshold for filtering tiny disconnected components.
        output_dir: Root directory for analytics outputs.

    Returns:
        Tuple of (report_dict, json_report_path, visualization_path).
    """
    m_path = Path(mask_path)
    out_dir = Path(output_dir)
    reports_dir = out_dir / "reports"
    components_dir = out_dir / "components"
    visualizations_dir = out_dir / "visualizations"

    for d in [reports_dir, components_dir, visualizations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate mask
    binary_mask, (width, height) = load_and_validate_mask(m_path)

    # 2. Pixel-level statistics
    pixel_stats = compute_pixel_statistics(binary_mask)

    # 3. Connected component analysis & filtering
    component_stats, label_map = analyze_components(
        binary_mask=binary_mask,
        min_component_area=min_component_area,
    )

    # 4. Resolve image for visualization
    # If original_image_path not explicitly provided, attempt to locate by stem in default dataset
    stem = m_path.stem.replace("_mask", "")
    vis_base_image: Optional[Image.Image] = None

    if original_image_path and Path(original_image_path).is_file():
        vis_base_image = Image.open(original_image_path).convert("RGB")
    else:
        # Search common dataset paths
        possible_images = [
            PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images" / "default" / f"{stem}.png",
            PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images" / "default" / f"{stem}.jpg",
            PROJECT_ROOT / "test_images" / f"{stem}.png",
        ]
        for candidate in possible_images:
            if candidate.is_file():
                vis_base_image = Image.open(candidate).convert("RGB")
                break

    # If raw image still unavailable, draw over color-coded components
    if vis_base_image is None:
        retained_ids = {r.id for r in component_stats.regions}
        vis_base_image = create_colored_components_map(label_map, retained_ids=retained_ids)

    # 5. Generate Visualizations
    annotated_image = draw_bounding_boxes(
        image=vis_base_image,
        regions=component_stats.regions,
        draw_labels=True,
    )
    vis_output_path = visualizations_dir / f"{stem}_components.png"
    annotated_image.save(vis_output_path, format="PNG")

    # 6. Generate Machine-Readable Reports
    report_dict = build_report_dictionary(
        image_name=f"{stem}.png",
        width=width,
        height=height,
        solar_pixels=pixel_stats.solar_pixels,
        background_pixels=pixel_stats.background_pixels,
        solar_coverage_percent=pixel_stats.solar_coverage_percent,
        component_stats=component_stats,
    )

    json_report_path = reports_dir / f"{stem}_analytics.json"
    save_json_report(report_dict, json_report_path)

    csv_path = components_dir / f"{stem}_components.csv"
    save_components_csv(component_stats.regions, csv_path)

    return report_dict, json_report_path, vis_output_path


def print_analytics_summary(report: dict, json_path: Path, vis_path: Path) -> None:
    """Formats and prints terminal summary."""
    img = report["image"]
    seg = report["segmentation"]
    comp = report["components"]

    print("-" * 52)
    print("SolarMap-India Segmentation Analytics")
    print("-" * 52)
    print(f"Image:                        {img['name']}")
    print(f"Dimensions:                   {img['width']}x{img['height']}")
    print(f"Total Pixels:                 {img['total_pixels']:,}")
    print(f"Solar Pixels:                 {seg['solar_pixels']:,}")
    print(f"Background Pixels:            {seg['background_pixels']:,}")
    print(f"Solar Pixel Coverage:         {seg['solar_pixel_coverage_percent']:.2f}% (pixel coverage)")
    print(f"Raw Components:               {comp['raw_count']}")
    print(f"Filtered Components:          {comp['filtered_count']}")
    print(f"Largest Region:               {comp['largest_area_pixels']:,} pixels")
    print(f"Smallest Region:              {comp['smallest_area_pixels']:,} pixels")
    print(f"Mean Region Area:             {comp['mean_area_pixels']:.2f} pixels")
    print(f"Median Region Area:           {comp['median_area_pixels']:.2f} pixels")
    print(f"Minimum Component Threshold:  {comp['min_component_area_pixels']} pixels")
    print(f"Retained Solar Pixels:        {comp['filtered_solar_pixels']:,} ({comp['retained_percentage']:.2f}%)")
    print(f"JSON Report:                  {json_path.resolve()}")
    print(f"Visualization:                {vis_path.resolve()}")
    print("-" * 52)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolarMap-India — Solar Segmentation Analytics Engine"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to the binary segmentation mask (PNG)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the original RGB image (optional, for bounding box overlay)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum connected component pixel area threshold (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_ANALYTICS_DIR),
        help=f"Directory to store analytics reports and visualizations (default: {DEFAULT_ANALYTICS_DIR})",
    )

    args = parser.parse_args()

    try:
        report, json_path, vis_path = run_segmentation_analytics(
            mask_path=args.mask,
            original_image_path=args.image,
            min_component_area=args.min_area,
            output_dir=args.output_dir,
        )
        print_analytics_summary(report, json_path, vis_path)
    except Exception as exc:
        print(f"\n[ERROR] Analytics processing failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
