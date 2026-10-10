"""
SolarMap-India — Confidence & Uncertainty Report Generator (JSON & CSV).

Exports machine-readable summaries containing:
- Model probability validation results
- Global confidence and uncertainty statistics
- 10-bin confidence distribution histogram
- Ambiguity and uncertainty metrics
- Region-level confidence records and operational categorizations
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ml.confidence.region_confidence import RegionConfidenceRecord


def build_confidence_report(
    image_name: str,
    width: int,
    height: int,
    max_sum_dev: float,
    solar_stats: Optional[dict],
    bg_stats: Optional[dict],
    global_conf_stats: dict,
    global_unc_stats: dict,
    histogram: List[dict],
    ambiguity_stats: dict,
    region_summary: dict,
    region_records: List[RegionConfidenceRecord],
) -> Dict[str, Any]:
    """
    Constructs the canonical JSON report dictionary.
    """
    return {
        "image": {
            "name": image_name,
            "width": int(width),
            "height": int(height),
            "total_pixels": int(width * height),
        },
        "probability_validation": {
            "is_valid": True,
            "max_probability_sum_deviation": round(float(max_sum_dev), 6),
            "numerical_tolerance": 1e-4,
        },
        "probabilities": {
            "solar_pixels_probability": solar_stats if solar_stats else None,
            "background_pixels_probability": bg_stats if bg_stats else None,
            "global_prediction_confidence": global_conf_stats,
            "global_prediction_uncertainty": global_unc_stats,
        },
        "confidence_distribution_histogram": histogram,
        "ambiguity": ambiguity_stats,
        "region_confidence_summary": region_summary,
        "regions": [r.to_dict() for r in region_records],
        "scientific_disclaimer": (
            "Confidence metrics represent model output probabilities directly from the softmax layer. "
            "They are operational prediction confidence measures and do NOT represent calibrated Bayesian "
            "probabilities or guarantees of empirical accuracy."
        ),
    }


def save_confidence_json(
    report_dict: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Writes formatted JSON report to disk."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    return out_file


def save_region_confidence_csv(
    records: List[RegionConfidenceRecord],
    output_path: Union[str, Path],
) -> Path:
    """Exports per-region confidence table to CSV."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "area_pixels",
        "mean_solar_probability",
        "median_solar_probability",
        "min_solar_probability",
        "max_solar_probability",
        "std_solar_probability",
        "mean_confidence",
        "mean_uncertainty",
        "confidence_category",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())

    return out_file
