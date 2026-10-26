"""
SolarMap-India — Scientific Assumptions & Environmental Baselines (Phase 7).

Defines the mathematical formulations, emission-factor ranges, qualitative limitations,
and scientific baselines for calculating avoided greenhouse gas (CO2) emissions.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

# Authoritative Indian National Grid Emission Factors from Central Electricity Authority (CEA)
# Source: "CO2 Baseline Database for the Indian Power Sector", Ministry of Power, GoI.
# Version 19.0 / 20.0 (covering national synchronous grid NEWNE + SR).
# Weighted Average Grid Emission Factor including Renewable Energy Sources (RES): ~0.716 kg CO2/kWh (tCO2/MWh)
STANDARD_CEA_GRID_EMISSION_FACTOR = 0.716

# Thermal/fossil-dominant generation baseline (excluding RES): ~0.820 kg CO2/kWh
CEA_THERMAL_GRID_EMISSION_FACTOR = 0.820

# Physical validation bounds for electrical grid emission factors globally
# 0.05 kg CO2/kWh: Near-zero hydro/nuclear/renewable grids (e.g., Norway, France)
# 1.50 kg CO2/kWh: Extreme low-efficiency subcritical coal-fired grids
MIN_PLAUSIBLE_EMISSION_FACTOR = 0.05
MAX_PLAUSIBLE_EMISSION_FACTOR = 1.50


@dataclass
class EnvironmentalAssumptions:
    """Documents the comprehensive physical assumptions and limitations of the CO2 avoidance model."""
    formula_kg: str = "CO2_avoided_kg_year = E_annual_kWh * EF_kg_per_kWh"
    formula_tonnes: str = "CO2_avoided_tonnes_year = CO2_avoided_kg_year / 1000.0"
    model_name: str = "First-Order Photovoltaic Grid Carbon Avoidance Model"
    co2_label: str = "Estimated Avoided CO2 Emissions (NOT Measured or Verified Carbon Reductions)"
    default_factor_source: str = (
        "Central Electricity Authority (CEA), Ministry of Power, Government of India — "
        "CO2 Baseline Database for the Indian Power Sector."
    )
    limitations: List[str] = field(default_factory=lambda: [
        "Grid Emission Factor Dynamics: National grid emission factors represent annual averages; real avoided emissions depend on the marginal fuel displaced during hours of peak solar irradiance.",
        "Transmission & Distribution (T&D) Losses: Distributed rooftop solar offsets generation behind the meter; actual utility-level displacement may be higher if avoided T&D losses (~15-20% in India) are credited.",
        "Grid Decarbonization Trajectory: Annual baseline factors decline over time as utility-scale renewables enter the grid; applying a static factor over a 25-year panel lifetime overestimates lifetime avoidance.",
        "Unified vs. Regional Dispatch: While India operates a single synchronous national grid, state-level merit-order dispatch and local generation mixes vary across regions.",
        "Upstream Lifecycle Emissions: This first-order model estimates operating displacement only, excluding embodied emissions from PV cell silicon purification, transport, mounting structures, and inverter manufacturing.",
        "Regulatory Status: Outputs are analytical engineering estimates for preliminary assessment, not certified emission reductions (CERs) audited under CDM, VCS, or Gold Standard carbon registries.",
    ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
