"""
SolarMap-India — Energy Input & Parameter Validation.

Provides rigorous scientific verification for physical scale, photovoltaic efficiency,
performance ratio, and solar irradiation inputs.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from ml.energy.assumptions import (
    MIN_PLAUSIBLE_EFFICIENCY,
    MAX_PLAUSIBLE_EFFICIENCY,
    MIN_PLAUSIBLE_PR,
    MAX_PLAUSIBLE_PR,
)

MIN_PLAUSIBLE_ANNUAL_GHI = 500.0   # Arctic minimum
MAX_PLAUSIBLE_ANNUAL_GHI = 3500.0  # High-altitude Atacama Desert maximum


@dataclass
class EnergyValidationResult:
    """Outcome of parameter validation checks."""
    is_valid: bool
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnergyValidator:
    """Validates inputs to the physical area and energy calculation layer."""

    @staticmethod
    def validate_gsd(gsd_meters: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Validates Ground Sampling Distance."""
        if gsd_meters is None:
            return False, "Ground Sampling Distance (GSD) is missing or None."
        if not isinstance(gsd_meters, (int, float)):
            return False, f"GSD must be a numeric value, got {type(gsd_meters).__name__}."
        if gsd_meters <= 0.0:
            return False, f"GSD must be strictly positive, got {gsd_meters}."
        if gsd_meters > 500.0:
            return False, f"GSD of {gsd_meters} m/px exceeds plausible satellite resolution."
        return True, None

    @staticmethod
    def validate_efficiency(efficiency: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Validates PV module efficiency."""
        if efficiency is None:
            return False, "Module efficiency is missing or null."
        if not isinstance(efficiency, (int, float)):
            return False, f"Efficiency must be numeric, got {type(efficiency).__name__}."
        if not (MIN_PLAUSIBLE_EFFICIENCY <= efficiency <= MAX_PLAUSIBLE_EFFICIENCY):
            return False, (
                f"Module efficiency {efficiency} ({efficiency*100:.1f}%) is outside "
                f"physically plausible range [{MIN_PLAUSIBLE_EFFICIENCY}, {MAX_PLAUSIBLE_EFFICIENCY}]."
            )
        return True, None

    @staticmethod
    def validate_performance_ratio(pr: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Validates system performance ratio."""
        if pr is None:
            return False, "Performance ratio (PR) is missing or null."
        if not isinstance(pr, (int, float)):
            return False, f"Performance ratio must be numeric, got {type(pr).__name__}."
        if not (MIN_PLAUSIBLE_PR <= pr <= MAX_PLAUSIBLE_PR):
            return False, (
                f"Performance ratio {pr} is outside physically plausible range "
                f"[{MIN_PLAUSIBLE_PR}, {MAX_PLAUSIBLE_PR}]."
            )
        return True, None

    @staticmethod
    def validate_annual_ghi(annual_ghi: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Validates annual solar irradiation (kWh/m^2/year)."""
        if annual_ghi is None:
            return False, "Annual GHI is missing or null."
        if not isinstance(annual_ghi, (int, float)):
            return False, f"Annual GHI must be numeric, got {type(annual_ghi).__name__}."
        if not (MIN_PLAUSIBLE_ANNUAL_GHI <= annual_ghi <= MAX_PLAUSIBLE_ANNUAL_GHI):
            return False, (
                f"Annual GHI {annual_ghi} kWh/m^2/year is outside terrestrial surface range "
                f"[{MIN_PLAUSIBLE_ANNUAL_GHI}, {MAX_PLAUSIBLE_ANNUAL_GHI}]."
            )
        return True, None

    @staticmethod
    def validate_solar_pixels(solar_pixels: int, total_pixels: int) -> Tuple[bool, Optional[str]]:
        """Validates pixel counts."""
        if not isinstance(solar_pixels, int) or not isinstance(total_pixels, int):
            return False, "Pixel counts must be integers."
        if solar_pixels < 0:
            return False, f"Solar pixel count cannot be negative: {solar_pixels}."
        if total_pixels <= 0:
            return False, f"Total image pixel count must be positive: {total_pixels}."
        if solar_pixels > total_pixels:
            return False, f"Solar pixels ({solar_pixels}) exceed total pixels ({total_pixels})."
        return True, None
