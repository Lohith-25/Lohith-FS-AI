"""
SolarMap-India — Solar Resource Client & Caching Foundation.

Fetches authoritative multi-annual solar climatology data from NASA POWER,
with deterministic disk caching, strict offline/failure handling, and scientific provenance.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import urllib.error
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.external_data.provenance import DataProvenance
from ml.external_data.validation import ExternalDataValidator

DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "external_data" / "cache"
NASA_POWER_CLIMATOLOGY_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
DEFAULT_PARAMETER = "ALLSKY_SFC_SW_DWN"


@dataclass
class SolarResourceResult:
    """Standardized representation of retrieved solar resource data."""
    source: str
    variable: str
    daily_value_kwh_m2_day: Optional[float]
    annual_value_kwh_m2_year: Optional[float]
    unit: str
    monthly_values: Dict[str, float] = field(default_factory=dict)
    temporal_basis: str = "20-year multi-annual climatology (2001-2020)"
    retrieval_date: str = ""
    status: str = "unavailable"  # "available" | "unavailable"
    error_message: Optional[str] = None
    from_cache: bool = False
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SolarResourceService:
    """
    Service for querying NASA POWER climatology API with deterministic local caching.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout_seconds: float = 12.0,
    ):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.timeout_seconds = timeout_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        latitude: float,
        longitude: float,
        parameter: str = DEFAULT_PARAMETER,
        temporal_basis: str = "climatology",
    ) -> str:
        """
        Generates a deterministic hash-based cache key for coordinates and parameters.
        Coordinates are normalized to 4 decimal places (~11 meters).
        """
        lat_norm = f"{latitude:.4f}"
        lon_norm = f"{longitude:.4f}"
        key_raw = f"NASA_POWER|{lat_norm}|{lon_norm}|{parameter}|{temporal_basis}"
        digest = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]
        return f"solar_res_{lat_norm}_{lon_norm}_{digest}.json"

    def get_solar_resource(
        self,
        latitude: float,
        longitude: float,
        parameter: str = DEFAULT_PARAMETER,
        use_cache: bool = True,
    ) -> SolarResourceResult:
        """
        Retrieves solar resource for coordinates. Checks cache first, then API.
        If offline, API fails, or coordinates are invalid, returns status 'unavailable'
        without substituting fake values.
        """
        # 1. Validate coordinates
        is_valid_coords, coord_errors = ExternalDataValidator.validate_coordinates(latitude, longitude)
        if not is_valid_coords:
            return SolarResourceResult(
                source="NASA Langley Research Center — POWER Project",
                variable=parameter,
                daily_value_kwh_m2_day=None,
                annual_value_kwh_m2_year=None,
                unit="kW-hr/m^2/day",
                status="unavailable",
                error_message=f"Coordinate validation failed: {'; '.join(coord_errors)}",
            )

        cache_filename = self.generate_cache_key(latitude, longitude, parameter)
        cache_file_path = self.cache_dir / cache_filename

        # 2. Check local disk cache
        if use_cache and cache_file_path.is_file():
            try:
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_data["from_cache"] = True
                return SolarResourceResult(**cached_data)
            except Exception:
                # Cache corrupted; proceed to re-fetch
                pass

        # 3. Build API request
        params = {
            "parameters": parameter,
            "community": "RE",
            "longitude": f"{longitude:.6f}",
            "latitude": f"{latitude:.6f}",
            "format": "JSON",
        }
        url = f"{NASA_POWER_CLIMATOLOGY_URL}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "SolarMap-India/1.0 (Geospatial Research)"}

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    return SolarResourceResult(
                        source="NASA Langley Research Center — POWER Project",
                        variable=parameter,
                        daily_value_kwh_m2_day=None,
                        annual_value_kwh_m2_year=None,
                        unit="kW-hr/m^2/day",
                        status="unavailable",
                        error_message=f"HTTP status error: {response.status}",
                    )
                raw_bytes = response.read()
                payload = json.loads(raw_bytes.decode("utf-8"))

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as net_err:
            return SolarResourceResult(
                source="NASA Langley Research Center — POWER Project",
                variable=parameter,
                daily_value_kwh_m2_day=None,
                annual_value_kwh_m2_year=None,
                unit="kW-hr/m^2/day",
                status="unavailable",
                error_message=f"Network retrieval failed: {str(net_err)}",
            )
        except json.JSONDecodeError as json_err:
            return SolarResourceResult(
                source="NASA Langley Research Center — POWER Project",
                variable=parameter,
                daily_value_kwh_m2_day=None,
                annual_value_kwh_m2_year=None,
                unit="kW-hr/m^2/day",
                status="unavailable",
                error_message=f"Malformed JSON in response: {str(json_err)}",
            )

        # 4. Validate payload contents
        is_valid_payload, payload_err = ExternalDataValidator.validate_nasa_power_payload(payload)
        if not is_valid_payload:
            return SolarResourceResult(
                source="NASA Langley Research Center — POWER Project",
                variable=parameter,
                daily_value_kwh_m2_day=None,
                annual_value_kwh_m2_year=None,
                unit="kW-hr/m^2/day",
                status="unavailable",
                error_message=f"Payload validation failed: {payload_err}",
            )

        # 5. Extract values and monthly data
        param_data = payload["properties"]["parameter"][parameter]
        ann_daily = float(param_data["ANN"])
        ann_yearly = round(ann_daily * 365.25, 2)  # Mathematically sound daily-to-annual irradiation conversion

        monthly = {k: float(v) for k, v in param_data.items() if k != "ANN" and v != -999.0}

        # 6. Build scientific provenance
        provenance = DataProvenance.create_nasa_power_provenance(
            latitude=latitude,
            longitude=longitude,
            returned_value=ann_daily,
        )

        result = SolarResourceResult(
            source="NASA Langley Research Center — POWER Project",
            variable=parameter,
            daily_value_kwh_m2_day=ann_daily,
            annual_value_kwh_m2_year=ann_yearly,
            unit="kW-hr/m^2/day",
            monthly_values=monthly,
            temporal_basis="20-year multi-annual climatology (2001-2020)",
            retrieval_date=provenance.retrieval_date_utc,
            status="available",
            error_message=None,
            from_cache=False,
            provenance=provenance.to_dict(),
        )

        # 7. Write to deterministic disk cache
        try:
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception:
            pass

        return result
