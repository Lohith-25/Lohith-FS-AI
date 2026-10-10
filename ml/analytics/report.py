"""
SolarMap-India — Analytics Report Generation (JSON & CSV).

Exports machine-readable reports containing:
- Image pixel dimensions
- Pixel segmentation statistics
- Connected component summary metrics
- Per-region bounding box and centroid properties
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from ml.analytics.components import ComponentStatistics, SolarRegion


def build_report_dictionary(
    image_name: str,
    width: int,
    height: int,
    solar_pixels: int,
    background_pixels: int,
    solar_coverage_percent: float,
    component_stats: ComponentStatistics,
) -> Dict[str, Any]:
    """
    Constructs the canonical analytics report dictionary.
    """
    total_pixels = width * height
    return {
        "image": {
            "name": image_name,
            "width": int(width),
            "height": int(height),
            "total_pixels": int(total_pixels),
        },
        "segmentation": {
            "solar_pixels": int(solar_pixels),
            "background_pixels": int(background_pixels),
            "solar_pixel_coverage_percent": round(float(solar_coverage_percent), 4),
        },
        "components": component_stats.to_dict(),
        "regions": [region.to_dict() for region in component_stats.regions],
    }


def save_json_report(
    report_dict: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Writes formatted JSON report to disk.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    return out_file


def save_components_csv(
    regions: List[SolarRegion],
    output_path: Union[str, Path],
) -> Path:
    """
    Writes individual solar region records to a tabular CSV file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "area_pixels",
        "x",
        "y",
        "width",
        "height",
        "centroid_x",
        "centroid_y",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in regions:
            writer.writerow(r.to_dict())

    return out_file
