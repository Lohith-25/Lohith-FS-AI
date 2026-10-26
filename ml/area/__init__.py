"""
SolarMap-India — Physical Area & GSD Provider Package.

Phase 7A components:
- GSDProvider & GSDResult
- calculate_physical_area & PhysicalAreaResult
- compute_image_solar_area pipeline
"""

from ml.area.gsd_provider import GSDProvider, GSDResult, PROHIBITED_GENERIC_GSDS
from ml.area.calculator import calculate_physical_area, PhysicalAreaResult
from ml.area.pipeline import compute_image_solar_area, print_area_summary
from ml.area.calibration import (
    ImageCalibration,
    validate_user_gsd,
    resolve_image_calibration,
    calculate_calibrated_area,
    MIN_PLAUSIBLE_GSD_METERS,
    MAX_PLAUSIBLE_GSD_METERS,
)

__all__ = [
    "GSDProvider",
    "GSDResult",
    "PROHIBITED_GENERIC_GSDS",
    "calculate_physical_area",
    "PhysicalAreaResult",
    "compute_image_solar_area",
    "print_area_summary",
    "ImageCalibration",
    "validate_user_gsd",
    "resolve_image_calibration",
    "calculate_calibrated_area",
    "MIN_PLAUSIBLE_GSD_METERS",
    "MAX_PLAUSIBLE_GSD_METERS",
]

