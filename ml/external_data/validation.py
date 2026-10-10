"""
SolarMap-India — External Data Validation Foundation.

Validates input coordinates, API response formats, physical value bounds,
and missing fill-values for solar resource datasets.
"""

from typing import Any, Dict, List, Optional, Tuple

# Physical bounds for surface solar irradiance
# Earth extraterrestrial solar constant is ~1361 W/m^2; daily top-of-atmosphere is ~11 kWh/m^2/day.
# Surface daily average GHI cannot physically exceed ~12.0 kWh/m^2/day anywhere on Earth,
# and for long-term climatology in India it typically ranges between 3.5 and 7.5 kWh/m^2/day.
MIN_DAILY_IRRADIANCE_KWH = 0.1   # minimum plausible non-zero daily GHI
MAX_DAILY_IRRADIANCE_KWH = 12.0  # physical maximum plausible daily surface GHI

INDIA_LAT_BOUNDS = (8.0, 38.0)
INDIA_LON_BOUNDS = (68.0, 98.0)


class ExternalDataValidator:
    """Validation utilities for coordinates and external solar resource responses."""

    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, List[str]]:
        """
        Validates latitude and longitude for global sanity and India bounds.
        """
        errors = []
        if not (-90.0 <= latitude <= 90.0):
            errors.append(f"Latitude out of global bounds [-90, 90]: {latitude}")
        if not (-180.0 <= longitude <= 180.0):
            errors.append(f"Longitude out of global bounds [-180, 180]: {longitude}")

        if not errors:
            if not (INDIA_LAT_BOUNDS[0] <= latitude <= INDIA_LAT_BOUNDS[1] and
                    INDIA_LON_BOUNDS[0] <= longitude <= INDIA_LON_BOUNDS[1]):
                errors.append(
                    f"Coordinates ({latitude}, {longitude}) are outside expected India bounding box "
                    f"[{INDIA_LAT_BOUNDS}, {INDIA_LON_BOUNDS}]."
                )

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_nasa_power_payload(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates the structure of a raw NASA POWER JSON response.
        """
        if not isinstance(payload, dict):
            return False, "Payload is not a valid JSON dictionary."

        properties = payload.get("properties")
        if not properties or not isinstance(properties, dict):
            return False, "Missing 'properties' key in NASA POWER response."

        parameters = properties.get("parameter")
        if not parameters or not isinstance(parameters, dict):
            return False, "Missing 'parameter' map in 'properties'."

        param_data = parameters.get("ALLSKY_SFC_SW_DWN")
        if param_data is None:
            return False, "Requested parameter 'ALLSKY_SFC_SW_DWN' not found in response."

        if not isinstance(param_data, dict):
            return False, "Parameter 'ALLSKY_SFC_SW_DWN' data is not a dictionary."

        ann = param_data.get("ANN")
        if ann is None:
            return False, "Missing annual climatology ('ANN') in parameter data."

        # NASA POWER fill value is -999.0
        if ann == -999.0 or ann < 0:
            return False, f"Missing or fill-value returned for annual irradiance: {ann}"

        # Physical reasonableness check
        if not (MIN_DAILY_IRRADIANCE_KWH <= ann <= MAX_DAILY_IRRADIANCE_KWH):
            return False, (
                f"Irradiance value {ann} kWh/m^2/day is outside physically plausible surface "
                f"range [{MIN_DAILY_IRRADIANCE_KWH}, {MAX_DAILY_IRRADIANCE_KWH}]."
            )

        return True, None

    @staticmethod
    def validate_units(unit_string: str, expected_units: List[str]) -> bool:
        """Validates that a unit string conforms to approved scientific units."""
        norm = unit_string.strip().lower().replace(" ", "")
        for exp in expected_units:
            if norm == exp.strip().lower().replace(" ", ""):
                return True
        return False
