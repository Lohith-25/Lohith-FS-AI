"""
SolarMap-India — Phase 7: CO2 & Environmental Analysis Package.

Provides scientifically validated estimation of greenhouse gas (CO2) emission
avoidance from photovoltaic energy potential, based on Indian electricity grid
emission factors (Central Electricity Authority).
"""

from ml.environmental.assumptions import (
    EnvironmentalAssumptions,
    STANDARD_CEA_GRID_EMISSION_FACTOR,
    CEA_THERMAL_GRID_EMISSION_FACTOR,
    MIN_PLAUSIBLE_EMISSION_FACTOR,
    MAX_PLAUSIBLE_EMISSION_FACTOR,
)
from ml.environmental.emission_factors import (
    GridEmissionConfig,
    EmissionFactorProvenance,
)
from ml.environmental.validation import EnvironmentalValidator
from ml.environmental.co2_model import (
    CO2AvoidanceModel,
    CO2CalculationResult,
)
from ml.environmental.report import (
    run_environmental_analysis_pipeline,
    build_and_save_phase7_reports,
)

__all__ = [
    "EnvironmentalAssumptions",
    "STANDARD_CEA_GRID_EMISSION_FACTOR",
    "CEA_THERMAL_GRID_EMISSION_FACTOR",
    "MIN_PLAUSIBLE_EMISSION_FACTOR",
    "MAX_PLAUSIBLE_EMISSION_FACTOR",
    "GridEmissionConfig",
    "EmissionFactorProvenance",
    "EnvironmentalValidator",
    "CO2AvoidanceModel",
    "CO2CalculationResult",
    "run_environmental_analysis_pipeline",
    "build_and_save_phase7_reports",
]
