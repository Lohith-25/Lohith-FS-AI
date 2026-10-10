"""
SolarMap-India — Prediction Confidence & Uncertainty Package.
"""

from ml.confidence.confidence_maps import (
    generate_all_confidence_maps,
    save_heatmap_visualization,
)
from ml.confidence.probability_analysis import (
    ProbabilityStats,
    compute_ambiguity_stats,
    compute_confidence_and_uncertainty,
    compute_confidence_histogram,
    compute_stats,
    extract_model_probabilities,
    run_confidence_pipeline,
    validate_probabilities,
)
from ml.confidence.region_confidence import (
    RegionConfidenceRecord,
    classify_region_confidence,
    compute_region_confidences,
    summarize_region_confidences,
)
from ml.confidence.report import (
    build_confidence_report,
    save_confidence_json,
    save_region_confidence_csv,
)

__all__ = [
    "ProbabilityStats",
    "validate_probabilities",
    "compute_confidence_and_uncertainty",
    "compute_stats",
    "compute_confidence_histogram",
    "compute_ambiguity_stats",
    "extract_model_probabilities",
    "run_confidence_pipeline",
    "save_heatmap_visualization",
    "generate_all_confidence_maps",
    "RegionConfidenceRecord",
    "classify_region_confidence",
    "compute_region_confidences",
    "summarize_region_confidences",
    "build_confidence_report",
    "save_confidence_json",
    "save_region_confidence_csv",
]
