"""
SolarMap-India — Photovoltaic CO2 Avoidance Model (Phase 7).

Estimates annual avoided grid greenhouse gas (CO2) emissions:
    CO2 avoided (kg/year) = Annual solar electricity generation (kWh/year) * Grid emission factor (kg CO2/kWh)
    CO2 avoided (tonnes/year) = CO2 avoided (kg/year) / 1000.0

Strictly returns co2_status = 'insufficient_data' and co2_avoided_kg_year = null
whenever physical area, solar resource, or energy generation is unavailable or unverified.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from ml.energy.energy_model import EnergyCalculationResult
from ml.environmental.assumptions import EnvironmentalAssumptions
from ml.environmental.emission_factors import GridEmissionConfig
from ml.environmental.validation import EnvironmentalValidator


@dataclass
class CO2CalculationResult:
    """Comprehensive output of the avoided CO2 emissions calculation."""
    co2_label: str
    co2_status: str                              # "estimated" | "insufficient_data"
    annual_energy_kwh_year: Optional[float]
    emission_factor_kg_kwh: Optional[float]
    co2_avoided_kg_year: Optional[float]
    co2_avoided_tonnes_year: Optional[float]
    formula: str
    units: Dict[str, str] = field(default_factory=lambda: {
        "energy": "kWh/year",
        "emission_factor": "kg CO2/kWh",
        "co2_avoided_kg": "kg CO2/year",
        "co2_avoided_tonnes": "metric tonnes CO2/year",
    })
    unavailability_reasons: List[str] = field(default_factory=list)
    energy_details: Optional[Dict[str, Any]] = None
    emission_factor_details: Optional[Dict[str, Any]] = None
    assumptions_and_limitations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CO2AvoidanceModel:
    """
    Calculates estimated annual avoided grid CO2 emissions with rigorous input verification.
    """

    @classmethod
    def estimate_avoided_emissions(
        cls,
        energy_input: Union[EnergyCalculationResult, Dict[str, Any], float, int, None],
        config: Optional[GridEmissionConfig] = None,
        emission_factor_kg_kwh: Optional[float] = None,
    ) -> CO2CalculationResult:
        """
        Estimates annual avoided CO2 emissions (kg/year and metric tonnes/year).

        Args:
            energy_input: EnergyCalculationResult, dict output from energy pipeline,
                          or numeric annual generation in kWh/year.
            config: Optional GridEmissionConfig. If omitted, defaults to unconfigured default.
            emission_factor_kg_kwh: Optional direct emission factor override (kg CO2/kWh).

        Returns:
            CO2CalculationResult with calculated figures or explicit insufficient_data status.
        """
        assumptions = EnvironmentalAssumptions()
        cfg = config or GridEmissionConfig.create_default()
        reasons: List[str] = []

        # 1. Resolve annual energy generation and upstream status
        energy_details_dict: Optional[Dict[str, Any]] = None
        annual_energy: Optional[float] = None

        if isinstance(energy_input, EnergyCalculationResult):
            energy_details_dict = energy_input.to_dict()
            if energy_input.energy_status != "estimated" or energy_input.estimated_energy_kwh_year is None:
                reasons.append(
                    f"Annual solar energy potential unavailable (status: {energy_input.energy_status})."
                )
                if energy_input.unavailability_reasons:
                    reasons.extend(energy_input.unavailability_reasons)
            else:
                annual_energy = energy_input.estimated_energy_kwh_year

        elif isinstance(energy_input, dict):
            energy_details_dict = energy_input
            status = energy_input.get("energy_status")
            val = energy_input.get("estimated_energy_kwh_year")
            if status != "estimated" or val is None:
                reasons.append(
                    f"Annual solar energy potential unavailable (status: {status or 'unknown'})."
                )
                if "unavailability_reasons" in energy_input and isinstance(energy_input["unavailability_reasons"], list):
                    reasons.extend(energy_input["unavailability_reasons"])
            else:
                annual_energy = float(val)

        elif energy_input is None:
            reasons.append("Annual solar energy generation input is missing (null).")

        else:
            # Direct numeric energy input - strictly validate
            ok_eng, err_eng = EnvironmentalValidator.validate_energy_generation(energy_input)
            if not ok_eng:
                raise ValueError(f"Invalid annual energy input: {err_eng}")
            annual_energy = float(energy_input)

        # If annual_energy is extracted from a result/dict, validate its numeric plausibility
        if annual_energy is not None:
            ok_eng, err_eng = EnvironmentalValidator.validate_energy_generation(annual_energy)
            if not ok_eng:
                reasons.append(f"Invalid annual energy value: {err_eng}")
                annual_energy = None

        # 2. Resolve and validate grid emission factor
        effective_factor: Optional[float] = None
        if emission_factor_kg_kwh is not None:
            effective_factor = emission_factor_kg_kwh
        elif cfg.emission_factor_kg_kwh is not None:
            effective_factor = cfg.emission_factor_kg_kwh

        if effective_factor is not None:
            ok_ef, err_ef = EnvironmentalValidator.validate_emission_factor(effective_factor)
            if not ok_ef:
                raise ValueError(f"Invalid grid emission factor: {err_ef}")
            effective_factor = float(effective_factor)
        else:
            reasons.append(
                "Grid emission factor is unconfigured (null). Explicit parameter or scenario required."
            )

        # 3. Determine if calculation can proceed
        can_calculate = (
            annual_energy is not None
            and effective_factor is not None
            and len(reasons) == 0
        )

        if can_calculate:
            # CO2 avoided (kg/year) = E (kWh/year) * EF (kg CO2/kWh)
            co2_kg = round(float(annual_energy * effective_factor), 2)
            # CO2 avoided (tonnes/year) = kg / 1000.0
            co2_tonnes = round(float(co2_kg / 1000.0), 4)
            co2_status = "estimated"
        else:
            co2_kg = None
            co2_tonnes = None
            co2_status = "insufficient_data"

        return CO2CalculationResult(
            co2_label=assumptions.co2_label,
            co2_status=co2_status,
            annual_energy_kwh_year=annual_energy,
            emission_factor_kg_kwh=effective_factor,
            co2_avoided_kg_year=co2_kg,
            co2_avoided_tonnes_year=co2_tonnes,
            formula=f"{assumptions.formula_kg}; {assumptions.formula_tonnes}",
            unavailability_reasons=reasons,
            energy_details=energy_details_dict,
            emission_factor_details=cfg.to_dict(),
            assumptions_and_limitations=assumptions.to_dict(),
        )
