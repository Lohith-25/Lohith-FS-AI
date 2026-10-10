"""
SolarMap-India — Scientific Data Provenance.

Encapsulates complete audit trail and provenance metadata for every
external data retrieval, ensuring reproducibility and scientific integrity.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class DataProvenance:
    """Scientific provenance record for external data retrieval."""
    source_name: str
    source_url: str
    dataset_product_name: str
    variable_name: str
    variable_description: str
    native_units: str
    retrieval_date_utc: str
    coordinate_used: Dict[str, float]
    returned_value: Optional[float]
    spatial_resolution: str
    temporal_resolution: str
    licensing: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create_nasa_power_provenance(
        cls,
        latitude: float,
        longitude: float,
        returned_value: Optional[float],
        retrieval_date_utc: Optional[str] = None,
    ) -> "DataProvenance":
        """Generates standard provenance for NASA POWER Climatology irradiance."""
        dt_str = retrieval_date_utc or datetime.now(timezone.utc).isoformat()
        return cls(
            source_name="NASA Langley Research Center — POWER Project",
            source_url="https://power.larc.nasa.gov/api/temporal/climatology/point",
            dataset_product_name="SYN1DEG / CERES 20-Year Climatology (2001-2020)",
            variable_name="ALLSKY_SFC_SW_DWN",
            variable_description="All Sky Surface Shortwave Downward Irradiance (GHI daily average)",
            native_units="kW-hr/m^2/day",
            retrieval_date_utc=dt_str,
            coordinate_used={"latitude": float(latitude), "longitude": float(longitude)},
            returned_value=returned_value,
            spatial_resolution="1.0 x 1.0 degree global grid (~110 km)",
            temporal_resolution="Multi-annual monthly & annual climatological average",
            licensing="NASA Open Data Policy (public domain / US government work)",
        )
