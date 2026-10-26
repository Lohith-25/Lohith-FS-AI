"""
SolarMap-India — Environmental Input & Parameter Validation (Phase 7).

Provides rigorous scientific verification for annual energy yield inputs
and electricity grid carbon emission factors.
"""

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Optional, Tuple

from ml.environmental.assumptions import (
    MIN_PLAUSIBLE_EMISSION_FACTOR,
    MAX_PLAUSIBLE_EMISSION_FACTOR,
)

# Plausible upper bound for single rooftop or solar installation annual generation (kWh/year)
# 100 GWh/year (100,000,000 kWh/year) is far above single-image rooftop scale
MAX_PLAUSIBLE_ANNUAL_ENERGY_KWH = 100_000_000.0


@dataclass
class EnvironmentalValidationResult:
    """Outcome of environmental parameter validation checks."""
    is_valid: bool
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnvironmentalValidator:
    """Validates inputs to the CO2 and environmental avoidance modeling layer."""

    @staticmethod
    def validate_energy_generation(annual_energy_kwh: Any) -> Tuple[bool, Optional[str]]:
        """
        Validates annual photovoltaic energy generation input.
        """
        if annual_energy_kwh is None:
            return False, "Annual solar energy generation is missing or null."

        if isinstance(annual_energy_kwh, bool) or not isinstance(annual_energy_kwh, (int, float)):
            return False, f"Annual energy generation must be a numeric value, got {type(annual_energy_kwh).__name__}."

        if math.isnan(annual_energy_kwh) or math.isinf(annual_energy_kwh):
            return False, f"Annual energy generation must be a finite number, got {annual_energy_kwh}."

        if annual_energy_kwh < 0.0:
            return False, f"Annual solar energy generation cannot be negative: {annual_energy_kwh} kWh/year."

        if annual_energy_kwh > MAX_PLAUSIBLE_ANNUAL_ENERGY_KWH:
            return False, (
                f"Annual energy generation of {annual_energy_kwh:,.1f} kWh/year exceeds plausible "
                f"single-installation ceiling ({MAX_PLAUSIBLE_ANNUAL_ENERGY_KWH:,.0f} kWh/year)."
            )

        return True, None

    @staticmethod
    def validate_emission_factor(emission_factor_kg_kwh: Any) -> Tuple[bool, Optional[str]]:
        """
        Validates electricity grid emission factor (kg CO2/kWh).
        """
        if emission_factor_kg_kwh is None:
            return False, "Grid emission factor is missing or null."

        if isinstance(emission_factor_kg_kwh, bool) or not isinstance(emission_factor_kg_kwh, (int, float)):
            return False, f"Grid emission factor must be numeric, got {type(emission_factor_kg_kwh).__name__}."

        if math.isnan(emission_factor_kg_kwh) or math.isinf(emission_factor_kg_kwh):
            return False, f"Grid emission factor must be a finite number, got {emission_factor_kg_kwh}."

        if emission_factor_kg_kwh <= 0.0:
            return False, f"Grid emission factor must be strictly positive (> 0), got {emission_factor_kg_kwh}."

        if not (MIN_PLAUSIBLE_EMISSION_FACTOR <= emission_factor_kg_kwh <= MAX_PLAUSIBLE_EMISSION_FACTOR):
            return False, (
                f"Grid emission factor {emission_factor_kg_kwh} kg CO2/kWh is outside physically "
                f"plausible terrestrial grid range [{MIN_PLAUSIBLE_EMISSION_FACTOR}, {MAX_PLAUSIBLE_EMISSION_FACTOR}]."
            )

        return True, None
