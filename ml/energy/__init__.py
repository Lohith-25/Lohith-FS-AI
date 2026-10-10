"""
SolarMap-India — Physical Area & Estimated Solar Energy Potential Package.

Phase 6 foundation providing:
- Physical solar area estimation based on validated GSD.
- Rigorous handling of missing GSD ("insufficient_data").
- Transparent solar energy potential modeling (E = A * GHI * eta * PR).
- Configurable scientific assumptions for module efficiency and performance ratio.
- End-to-end provenance and limitation tracking.
"""

from ml.energy.assumptions import (
    ScientificAssumptions,
    EnergyModelConfig,
    STANDARD_EFFICIENCY_BENCHMARK,
    STANDARD_PERFORMANCE_RATIO_BENCHMARK,
)
from ml.energy.validation import EnergyValidator, EnergyValidationResult
from ml.energy.area import PhysicalAreaCalculator, AreaCalculationResult
from ml.energy.energy_model import SolarEnergyModel, EnergyCalculationResult

__all__ = [
    "ScientificAssumptions",
    "EnergyModelConfig",
    "STANDARD_EFFICIENCY_BENCHMARK",
    "STANDARD_PERFORMANCE_RATIO_BENCHMARK",
    "EnergyValidator",
    "EnergyValidationResult",
    "PhysicalAreaCalculator",
    "AreaCalculationResult",
    "SolarEnergyModel",
    "EnergyCalculationResult",
]
