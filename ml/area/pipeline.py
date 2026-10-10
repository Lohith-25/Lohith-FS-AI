"""
SolarMap-India — GSD-Based Physical Area Pipeline & CLI.

Connects:
    Solar Mask (Phase 1 / Phase 2)
          ↓
    Solar Pixel Count (Phase 2)
          ↓
    Validated GSD (GSDProvider)
          ↓
    Physical Solar Area (m²)
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predict import predict_image
from ml.area.gsd_provider import GSDProvider, GSDResult
from ml.area.calculator import PhysicalAreaResult, calculate_physical_area


def compute_image_solar_area(
    image_path: Union[str, Path],
    custom_gsd: Optional[float] = None,
    provenance_citation: Optional[str] = None,
) -> PhysicalAreaResult:
    """
    Executes the full GSD-based area calculation pipeline for an image.
    Reuses existing Phase 1/Phase 2 inference to obtain exact solar pixel count.
    """
    img_file = Path(image_path)
    if not img_file.is_file():
        # Check standard default image folder if relative filename provided
        default_candidate = (
            PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images" / "default" / img_file.name
        )
        if default_candidate.is_file():
            img_file = default_candidate
        else:
            raise FileNotFoundError(f"Image file not found: {image_path}")

    # 1. Reuse existing Phase 1/Phase 2 segmentation pipeline to get solar pixel count
    inf_res = predict_image(image_path=img_file)
    solar_pixels = inf_res.solar_pixels
    total_pixels = inf_res.total_pixels

    # 2. Query GSDProvider
    gsd_result = GSDProvider.get_gsd(
        image_path=img_file,
        custom_gsd=custom_gsd,
        provenance_citation=provenance_citation,
    )

    # 3. Calculate physical area
    area_result = calculate_physical_area(
        solar_pixels=solar_pixels,
        gsd_result=gsd_result,
        total_pixels=total_pixels,
    )

    return area_result


def print_area_summary(result: PhysicalAreaResult, image_name: str) -> None:
    """Prints formatted summary of the physical area assessment."""
    print("-" * 50)
    print("SolarMap-India GSD-Based Physical Area Analysis")
    print("-" * 50)
    print(f"Image:                 {image_name}")
    print(f"Solar Pixels:          {result.solar_pixels:,} / {result.total_pixels:,}")
    if result.pixel_coverage_percent is not None:
        print(f"Pixel Coverage:        {result.pixel_coverage_percent:.2f}%")
    print()
    print(f"GSD Status:            {result.gsd_status}")
    if result.gsd_m_per_pixel is not None:
        print(f"GSD (m/pixel):         {result.gsd_m_per_pixel:.4f} m/pixel")
        print(f"Pixel Area (m^2):      {result.pixel_area_m2:.6f} m^2")
    else:
        print(f"GSD (m/pixel):         unavailable")
        print(f"Pixel Area (m^2):      unavailable")
    print()
    print(f"Area Status:           {result.area_status}")
    if result.solar_area_m2 is not None:
        print(f"Physical Solar Area:   {result.solar_area_m2:,.2f} m^2")
    else:
        print(f"Physical Solar Area:   insufficient_data")
    print(f"Source:                {result.source}")
    print(f"Reason:                {result.status_reason}")
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolarMap-India — Phase 7A GSD-Based Physical Area Calculation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="768.0_1.0.png",
        help="Path or name of target image (default: 768.0_1.0.png)",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Optional authenticated candidate GSD in meters/pixel",
    )
    parser.add_argument(
        "--citation",
        type=str,
        default=None,
        help="Authoritative citation or certificate backing candidate GSD",
    )

    args = parser.parse_args()

    result = compute_image_solar_area(
        image_path=args.image,
        custom_gsd=args.gsd,
        provenance_citation=args.citation,
    )
    print_area_summary(result, Path(args.image).name)


if __name__ == "__main__":
    main()
