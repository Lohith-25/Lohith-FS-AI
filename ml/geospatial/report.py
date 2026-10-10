"""
SolarMap-India — Geospatial Foundation & Scale Status Reporter.

Assembles comprehensive dataset and image-level geospatial assessment,
including georeferencing classification (Level A/B/C/D) and energy readiness.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.geospatial.area_calculator import calculate_physical_area
from ml.geospatial.coordinate_mapping import GeospatialCatalog
from ml.geospatial.metadata_inspector import inspect_dataset_directory, inspect_image_metadata
from ml.geospatial.spatial_scale import get_dataset_spatial_scale

DEFAULT_GEOSPATIAL_DIR = PROJECT_ROOT / "outputs" / "geospatial"


def generate_geospatial_status_report(
    image_name: str = "768.0_1.0.png",
    solar_pixels: Optional[int] = 68451,
    output_dir: Union[str, Path] = DEFAULT_GEOSPATIAL_DIR,
) -> Tuple[Dict[str, Any], Path]:
    """
    Generates the canonical dataset and image geospatial assessment report.
    """
    out_dir = Path(output_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    catalog = GeospatialCatalog()
    img_dir = PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images" / "default"

    # 1. Dataset-level audit
    dataset_audit = inspect_dataset_directory(img_dir, sample_limit=25)
    catalog_audit = catalog.audit_bounds()

    # Match image counts
    all_disk_images = sorted([p.name for p in img_dir.glob("*.png")])
    matched_count = sum(1 for f in all_disk_images if catalog.lookup(f) is not None)
    unmatched_count = len(all_disk_images) - matched_count

    # 2. Determine georeferencing level
    # Level B: Per-image coordinates available from CSV; physical scale is unavailable
    georef_level = "LEVEL_B"
    energy_readiness = "BLOCKED"  # Blocked because physical panel area in m^2 is unavailable

    # 3. Target image evaluation
    target_rec = catalog.lookup(image_name)
    spatial_scale = get_dataset_spatial_scale()

    if solar_pixels is not None:
        area_eval = calculate_physical_area(
            pixel_count=solar_pixels,
            meters_per_pixel_x=spatial_scale.meters_per_pixel_x,
            meters_per_pixel_y=spatial_scale.meters_per_pixel_y,
        )
    else:
        area_eval = {"status": "insufficient_data", "area_m2": None}

    report = {
        "dataset": {
            "image_count": len(all_disk_images),
            "metadata_rows": catalog.total_records,
            "matched_images": matched_count,
            "unmatched_images": unmatched_count,
            "min_latitude": catalog_audit["min_latitude"],
            "max_latitude": catalog_audit["max_latitude"],
            "min_longitude": catalog_audit["min_longitude"],
            "max_longitude": catalog_audit["max_longitude"],
            "anomalous_out_of_bounds_count": catalog_audit["anomalous_out_of_bounds_count"],
        },
        "georeferencing": {
            "level": georef_level,
            "description": "Per-image geographic coordinates available; physical pixel scale unavailable.",
            "image_coordinates_available": True,
            "component_coordinates_available": False,
            "physical_pixel_scale_available": False,
            "crs_available": False,
            "transform_available": False,
            "energy_estimation_readiness": energy_readiness,
        },
        "critical_decisions": {
            "image_level_coordinates_supported": "SUPPORTED",
            "component_level_coordinates_supported": "NOT SUPPORTED (no affine transform matrix)",
            "physical_area_m2_supported": "NOT SUPPORTED (no GSD/meters-per-pixel metadata)",
            "energy_estimation_supported": "NOT SUPPORTED (blocked by physical area)",
        },
        "image": {
            "filename": image_name,
            "sampleid": target_rec.sampleid if target_rec else None,
            "latitude": target_rec.latitude if target_rec else None,
            "longitude": target_rec.longitude if target_rec else None,
            "is_within_india_bounds": target_rec.is_within_india_bounds if target_rec else None,
            "meters_per_pixel_x": None,
            "meters_per_pixel_y": None,
            "solar_pixels": solar_pixels,
            "physical_area_status": area_eval["status"],
            "physical_area_m2": area_eval["area_m2"],
        },
        "scientific_disclaimer": (
            "Image-level geographic coordinates indicate the approximate ground location of the scene. "
            "Because calibrated Ground Sampling Distance (GSD) is absent from the raster metadata, "
            "physical solar panel area (m^2) cannot be legitimately computed and is reported as insufficient_data."
        ),
    }

    report_path = reports_dir / "geospatial_status.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report, report_path


def print_summary(report: dict, report_path: Path) -> None:
    """Formats and prints summary to terminal."""
    ds = report["dataset"]
    geo = report["georeferencing"]
    dec = report["critical_decisions"]
    img = report["image"]

    print("-" * 56)
    print("SolarMap-India Geospatial & Physical Area Assessment")
    print("-" * 56)
    print(f"Dataset Images:              {ds['image_count']}")
    print(f"Metadata Rows:               {ds['metadata_rows']}")
    print(f"Matched Images:              {ds['matched_images']} / {ds['image_count']}")
    print(f"Unmatched Images:            {ds['unmatched_images']}")
    print(f"Georeferencing Level:        {geo['level']} ({geo['description']})")
    print(f"Image Coordinates:           {dec['image_level_coordinates_supported']}")
    print(f"Component Coordinates:       {dec['component_level_coordinates_supported']}")
    print(f"Physical Pixel Scale:        {'AVAILABLE' if geo['physical_pixel_scale_available'] else 'UNAVAILABLE'}")
    print(f"Physical Area in m^2:        {dec['physical_area_m2_supported']}")
    print(f"Energy Readiness:            {geo['energy_estimation_readiness']}")
    print("-" * 56)
    print("Sample Image Details:")
    print(f"  Filename:                  {img['filename']}")
    print(f"  Sample ID:                 {img['sampleid']}")
    print(f"  Coordinates:               ({img['latitude']}, {img['longitude']})")
    print(f"  Physical Area Status:      {img['physical_area_status']}")
    print(f"  Physical Area (m^2):       {img['physical_area_m2']}")
    print(f"Status Report File:          {report_path.resolve()}")
    print("-" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolarMap-India — Geospatial Foundation Assessment"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="768.0_1.0.png",
        help="Target image filename for evaluation (default: 768.0_1.0.png)",
    )
    parser.add_argument(
        "--solar-pixels",
        type=int,
        default=68451,
        help="Solar pixel count from segmentation (default: 68451)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_GEOSPATIAL_DIR),
        help=f"Output directory (default: {DEFAULT_GEOSPATIAL_DIR})",
    )

    args = parser.parse_args()

    try:
        report, report_path = generate_geospatial_status_report(
            image_name=args.image,
            solar_pixels=args.solar_pixels,
            output_dir=args.output_dir,
        )
        print_summary(report, report_path)
    except Exception as exc:
        print(f"\n[ERROR] Geospatial assessment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
