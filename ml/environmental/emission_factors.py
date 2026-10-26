"""
SolarMap-India — Grid Emission Factors & Provenance (Phase 7).

Manages electricity grid carbon intensity factors, official source provenance,
and configurable scenario generators for carbon offset estimation.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from ml.environmental.assumptions import (
    STANDARD_CEA_GRID_EMISSION_FACTOR,
    CEA_THERMAL_GRID_EMISSION_FACTOR,
)


@dataclass
class EmissionFactorProvenance:
    """Detailed provenance tracking for grid emission factor sources."""
    source_agency: str
    database_product: str
    reporting_version: str
    reporting_period: str
    grid_scope: str
    factor_value: float
    factor_unit: str
    equivalent_unit: str
    methodology: str
    is_official_reference: bool
    citation_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create_cea_default(
        cls,
        factor: float = STANDARD_CEA_GRID_EMISSION_FACTOR,
    ) -> "EmissionFactorProvenance":
        """Builds provenance record for Central Electricity Authority (CEA) India baseline."""
        return cls(
            source_agency="Central Electricity Authority (CEA), Ministry of Power, Government of India",
            database_product="CO2 Baseline Database for the Indian Power Sector",
            reporting_version="User Guide Version 19.0 / Version 20.0",
            reporting_period="Baseline Fiscal Year (2022-2023 / 2023-2024)",
            grid_scope="Indian Unified National Grid (NEWNE + Southern Region)",
            factor_value=factor,
            factor_unit="kg CO2/kWh",
            equivalent_unit="tCO2/MWh",
            methodology="Weighted Average Grid Emission Factor (including Renewable Energy Sources)",
            is_official_reference=True,
            citation_url="https://cea.nic.in/cdm-co2-baseline-database/",
        )

    @classmethod
    def create_cea_thermal(
        cls,
        factor: float = CEA_THERMAL_GRID_EMISSION_FACTOR,
    ) -> "EmissionFactorProvenance":
        """Builds provenance record for CEA thermal generation weighted average."""
        return cls(
            source_agency="Central Electricity Authority (CEA), Ministry of Power, Government of India",
            database_product="CO2 Baseline Database for the Indian Power Sector",
            reporting_version="User Guide Version 19.0 / Version 20.0",
            reporting_period="Baseline Fiscal Year (2022-2023 / 2023-2024)",
            grid_scope="Indian Unified National Grid (NEWNE + Southern Region)",
            factor_value=factor,
            factor_unit="kg CO2/kWh",
            equivalent_unit="tCO2/MWh",
            methodology="Weighted Average Emission Factor of Thermal Power Stations (excluding RES)",
            is_official_reference=True,
            citation_url="https://cea.nic.in/cdm-co2-baseline-database/",
        )


@dataclass
class GridEmissionConfig:
    """
    Configuration parameters for the CO2 avoidance model.
    Values default strictly to None so that assumptions are never silently applied.
    """
    emission_factor_kg_kwh: Optional[float] = None
    factor_source_note: Optional[str] = None
    provenance: Optional[EmissionFactorProvenance] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if self.provenance is not None:
            res["provenance"] = self.provenance.to_dict()
        return res

    @classmethod
    def create_default(cls) -> "GridEmissionConfig":
        """Returns the unconfigured baseline where emission factor is an explicit null."""
        return cls(
            emission_factor_kg_kwh=None,
            factor_source_note="No grid emission factor provided; explicit parameter required.",
            provenance=None,
        )

    @classmethod
    def create_cea_benchmark_scenario(
        cls,
        emission_factor: float = STANDARD_CEA_GRID_EMISSION_FACTOR,
    ) -> "GridEmissionConfig":
        """
        Creates an explicit benchmark scenario using the authoritative CEA India national grid factor.
        """
        provenance = EmissionFactorProvenance.create_cea_default(factor=emission_factor)
        return cls(
            emission_factor_kg_kwh=emission_factor,
            factor_source_note=(
                f"Authoritative Indian National Grid Emission Factor ({emission_factor:.4f} kg CO2/kWh), "
                "published by the Central Electricity Authority (CEA), Ministry of Power, Government of India."
            ),
            provenance=provenance,
        )

    @classmethod
    def create_custom_scenario(
        cls,
        emission_factor: float,
        source_note: str,
        provenance: Optional[EmissionFactorProvenance] = None,
    ) -> "GridEmissionConfig":
        """
        Creates a custom scenario with user-specified emission factor and justification.
        """
        return cls(
            emission_factor_kg_kwh=emission_factor,
            factor_source_note=source_note,
            provenance=provenance,
        )
