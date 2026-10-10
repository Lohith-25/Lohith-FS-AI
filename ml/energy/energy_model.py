"""
SolarMap-India — Solar Energy Potential Model.

Implements the first-order photovoltaic yield formula:
    E = A * GHI * eta * PR

Where:
    E   = Estimated Solar Energy Potential (kWh/year)
    A   = Detected Solar Panel Area (m²)
    GHI = Annual Global Horizontal Irradiation (kWh/m²/year)
    eta = Photovoltaic Module Efficiency (fraction, e.g. 0.20)
    PR  = Performance Ratio (fraction, e.g. 0.75)

If any required input (A, GHI, eta, PR) is missing or unverified,
the model strictly returns energy_status = 'insufficient_data' and estimated_energy_kwh_year = null.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ml.energy.area import AreaCalculationResult
from ml.energy.assumptions import EnergyModelConfig, ScientificAssumptions
from ml.energy.validation import EnergyValidator


@dataclass
class SolarIrradiationRecord:
    """Stores source daily irradiance and derived annual irradiation with explicit conversion."""
    source_daily_value: Optional[float]
    source_unit: str
    derived_annual_value: Optional[float]
    derived_unit: str
    conversion: str
    variable_name: str
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnergyCalculationResult:
    """Comprehensive output of the solar energy potential calculation."""
    energy_label: str
    energy_status: str                         # "estimated" | "insufficient_data"
    solar_area_m2: Optional[float]
    annual_ghi_kwh_m2_year: Optional[float]
    module_efficiency: Optional[float]
    performance_ratio: Optional[float]
    estimated_energy_kwh_year: Optional[float]
    formula: str
    unavailability_reasons: List[str] = field(default_factory=list)
    area_details: Optional[Dict[str, Any]] = None
    solar_resource_details: Optional[Dict[str, Any]] = None
    assumptions_and_limitations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SolarEnergyModel:
    """
    Executes first-order photovoltaic energy potential modeling with rigorous input verification.
    """

    @classmethod
    def estimate_energy(
        cls,
        area_result: AreaCalculationResult,
        annual_ghi_kwh_m2_year: Optional[float] = None,
        config: Optional[EnergyModelConfig] = None,
        source_daily_ghi: Optional[float] = None,
        source_unit: str = "kW-hr/m^2/day",
        solar_resource_provenance: Optional[Dict[str, Any]] = None,
    ) -> EnergyCalculationResult:
        """
        Estimates annual photovoltaic potential (kWh/year).
        """
        cfg = config or EnergyModelConfig.create_default()
        assumptions = ScientificAssumptions()
        reasons: List[str] = []

        # 1. Check physical area availability
        solar_area = area_result.solar_area_m2
        if solar_area is None or area_result.physical_area_status != "calculated":
            reasons.append(
                f"Physical solar area unavailable (status: {area_result.physical_area_status}). "
                f"Reason: {area_result.status_reason}"
            )

        # 2. Check and derive solar resource (annual GHI)
        effective_annual_ghi = annual_ghi_kwh_m2_year
        conversion_note = "direct input"

        if effective_annual_ghi is None and source_daily_ghi is not None:
            # Check compatible daily units
            if "day" in source_unit.lower() and ("kwh" in source_unit.lower() or "kw-hr" in source_unit.lower()):
                effective_annual_ghi = round(float(source_daily_ghi) * 365.25, 2)
                conversion_note = "daily × 365.25"
            else:
                reasons.append(f"Incompatible source irradiance unit: {source_unit}")

        if effective_annual_ghi is not None:
            ok_ghi, err_ghi = EnergyValidator.validate_annual_ghi(effective_annual_ghi)
            if not ok_ghi:
                reasons.append(f"Invalid annual GHI: {err_ghi}")
                effective_annual_ghi = None
        else:
            reasons.append("Solar resource (annual GHI) is missing or unavailable.")

        # Build structured irradiation record
        irrad_record = SolarIrradiationRecord(
            source_daily_value=source_daily_ghi,
            source_unit=source_unit,
            derived_annual_value=effective_annual_ghi,
            derived_unit="kWh/m2/year",
            conversion=conversion_note,
            variable_name="ALLSKY_SFC_SW_DWN",
            provenance=solar_resource_provenance,
        )

        # 3. Check module efficiency
        eff = cfg.module_efficiency
        if eff is not None:
            ok_eff, err_eff = EnergyValidator.validate_efficiency(eff)
            if not ok_eff:
                reasons.append(f"Invalid module efficiency: {err_eff}")
                eff = None
        else:
            reasons.append("Module efficiency (eta) is unconfigured (null). Explicit parameter required.")

        # 4. Check performance ratio
        pr = cfg.performance_ratio
        if pr is not None:
            ok_pr, err_pr = EnergyValidator.validate_performance_ratio(pr)
            if not ok_pr:
                reasons.append(f"Invalid performance ratio: {err_pr}")
                pr = None
        else:
            reasons.append("Performance ratio (PR) is unconfigured (null). Explicit parameter required.")

        # 5. Determine if all required inputs are valid
        can_calculate = (
            solar_area is not None
            and effective_annual_ghi is not None
            and eff is not None
            and pr is not None
            and len(reasons) == 0
        )

        if can_calculate:
            # E = A * GHI * eta * PR
            estimated_kwh = round(float(solar_area * effective_annual_ghi * eff * pr), 2)
            energy_status = "estimated"
        else:
            estimated_kwh = None
            energy_status = "insufficient_data"

        return EnergyCalculationResult(
            energy_label=assumptions.energy_label,
            energy_status=energy_status,
            solar_area_m2=solar_area,
            annual_ghi_kwh_m2_year=effective_annual_ghi,
            module_efficiency=eff,
            performance_ratio=pr,
            estimated_energy_kwh_year=estimated_kwh,
            formula=assumptions.formula,
            unavailability_reasons=reasons,
            area_details=area_result.to_dict(),
            solar_resource_details=irrad_record.to_dict(),
            assumptions_and_limitations=assumptions.to_dict(),
        )
