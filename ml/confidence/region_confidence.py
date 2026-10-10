"""
SolarMap-India — Region-Level Confidence Analysis.

Integrates with Phase 2 connected components to calculate
localized probability, confidence, and uncertainty metrics per solar region.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ml.analytics.components import SolarRegion


@dataclass
class RegionConfidenceRecord:
    """Confidence summary for a specific connected solar region."""
    id: int
    area_pixels: int
    mean_solar_probability: float
    median_solar_probability: float
    min_solar_probability: float
    max_solar_probability: float
    std_solar_probability: float
    mean_confidence: float
    mean_uncertainty: float
    confidence_category: str

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "area_pixels": int(self.area_pixels),
            "mean_solar_probability": round(float(self.mean_solar_probability), 4),
            "median_solar_probability": round(float(self.median_solar_probability), 4),
            "min_solar_probability": round(float(self.min_solar_probability), 4),
            "max_solar_probability": round(float(self.max_solar_probability), 4),
            "std_solar_probability": round(float(self.std_solar_probability), 4),
            "mean_confidence": round(float(self.mean_confidence), 4),
            "mean_uncertainty": round(float(self.mean_uncertainty), 4),
            "confidence_category": self.confidence_category,
        }


def classify_region_confidence(
    mean_solar_prob: float,
    high_threshold: float = 0.90,
    medium_threshold: float = 0.70,
) -> str:
    """
    Assigns an operational descriptive category based on mean solar probability.
    NOTE: Operational categories only; not calibrated Bayesian confidence.
    """
    if mean_solar_prob >= high_threshold:
        return "HIGH"
    elif mean_solar_prob >= medium_threshold:
        return "MEDIUM"
    else:
        return "LOW"


def compute_region_confidences(
    regions: List[SolarRegion],
    label_map: np.ndarray,
    solar_prob: np.ndarray,
    confidence_map: np.ndarray,
    uncertainty_map: np.ndarray,
    high_threshold: float = 0.90,
    medium_threshold: float = 0.70,
) -> List[RegionConfidenceRecord]:
    """
    Computes statistical confidence measures for every connected solar region.

    Args:
        regions: Filtered SolarRegion list from Phase 2.
        label_map: 2D label map where label matches region.id.
        solar_prob: 2D float array of solar probabilities.
        confidence_map: 2D float array of prediction confidence.
        uncertainty_map: 2D float array of prediction uncertainty.
        high_threshold: Threshold for 'HIGH' categorization (default 0.90).
        medium_threshold: Threshold for 'MEDIUM' categorization (default 0.70).

    Returns:
        List of RegionConfidenceRecord instances.
    """
    records: List[RegionConfidenceRecord] = []

    for reg in regions:
        reg_mask = (label_map == reg.id)
        if not np.any(reg_mask):
            continue

        p_solar = solar_prob[reg_mask]
        p_conf = confidence_map[reg_mask]
        p_unc = uncertainty_map[reg_mask]

        mean_sp = float(np.mean(p_solar))
        median_sp = float(np.median(p_solar))
        min_sp = float(np.min(p_solar))
        max_sp = float(np.max(p_solar))
        std_sp = float(np.std(p_solar))

        mean_conf = float(np.mean(p_conf))
        mean_unc = float(np.mean(p_unc))

        category = classify_region_confidence(
            mean_solar_prob=mean_sp,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
        )

        records.append(
            RegionConfidenceRecord(
                id=reg.id,
                area_pixels=reg.area_pixels,
                mean_solar_probability=mean_sp,
                median_solar_probability=median_sp,
                min_solar_probability=min_sp,
                max_solar_probability=max_sp,
                std_solar_probability=std_sp,
                mean_confidence=mean_conf,
                mean_uncertainty=mean_unc,
                confidence_category=category,
            )
        )

    return records


def summarize_region_confidences(records: List[RegionConfidenceRecord]) -> dict:
    """Returns counts and breakdown of region confidence categories."""
    high_count = sum(1 for r in records if r.confidence_category == "HIGH")
    med_count = sum(1 for r in records if r.confidence_category == "MEDIUM")
    low_count = sum(1 for r in records if r.confidence_category == "LOW")
    total = len(records)

    return {
        "total_regions": total,
        "high_confidence_count": high_count,
        "high_confidence_percentage": round((high_count / total) * 100.0, 2) if total > 0 else 0.0,
        "medium_confidence_count": med_count,
        "medium_confidence_percentage": round((med_count / total) * 100.0, 2) if total > 0 else 0.0,
        "low_confidence_count": low_count,
        "low_confidence_percentage": round((low_count / total) * 100.0, 2) if total > 0 else 0.0,
    }
