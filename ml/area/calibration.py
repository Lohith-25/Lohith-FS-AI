"""
SolarMap-India — Image Calibration & Spatial Scale Resolution Engine.

Implements a scientifically defensible two-path calibration workflow:
Path A: Automatic Calibration (GeoTIFF tags, ModelPixelScale, CRS, Jacobian affine determinants)
Path B: Manual Calibration (User-supplied GSD in meters/pixel with provenance citation)

Strict Scientific Safeguards:
1. Validates that user GSD is numeric, finite, strictly positive, and within terrestrial Earth observation bounds.
2. Rejects generic unverified guesses without silent clamping.
3. Labels user input as USER_SUPPLIED, never automatically VERIFIED.
4. Enforces strict precedence: authoritative embedded GeoTIFF metadata takes precedence over user input.
5. Withholds physical area, energy, and CO2 whenever calibration is unavailable.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional, Tuple, Union

from ml.area.gsd_provider import PROHIBITED_GENERIC_GSDS
from ml.energy.area import AreaCalculationResult, PhysicalAreaCalculator
from ml.geospatial.metadata import GeoMetadata

# Physical plausibility bounds for terrestrial Earth observation imagery:
# - Minimum: 0.005 m/px (5 mm/px; practical lower limit for ultra-high-res drone mapping)
# - Maximum: 100.0 m/px (100 m/px; beyond which rooftop solar panels are sub-pixel)
MIN_PLAUSIBLE_GSD_METERS = 0.005
MAX_PLAUSIBLE_GSD_METERS = 100.0
MIN_PLAUSIBLE_GSD_M = MIN_PLAUSIBLE_GSD_METERS
MAX_PLAUSIBLE_GSD_M = MAX_PLAUSIBLE_GSD_METERS


@dataclass
class ImageCalibration:
    """Standardized representation of image spatial calibration."""
    calibration_type: str                   # "AUTOMATIC" | "MANUAL" | "NONE"
    calibration_status: str                 # "EXTRACTED_METADATA" | "USER_SUPPLIED" | "UNAVAILABLE" | "REJECTED"
    is_calibrated: bool
    gsd_meters: Optional[float]
    gsd_x_meters: Optional[float]
    gsd_y_meters: Optional[float]
    pixel_area_m2: Optional[float]
    calibration_source: str
    is_validated_metadata: bool
    is_user_supplied: bool
    rejection_reason: Optional[str] = None
    units: str = "meters/pixel"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create_uncalibrated(cls) -> "ImageCalibration":
        return resolve_image_calibration(geo_metadata=None, custom_gsd=None, calibration_source=None)


def validate_user_gsd(
    custom_gsd: Any,
    calibration_source: Optional[str] = None,
) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Validates user-supplied Ground Sampling Distance (GSD).

    Args:
        custom_gsd: Candidate GSD value.
        calibration_source: Optional source, flight reference, or description.

    Returns:
        (is_valid, parsed_gsd_float, clean_source_str, rejection_reason)
    """
    # 1. Type and Numeric Parsing
    if custom_gsd is None:
        return False, None, None, "GSD value is missing (null)."

    try:
        val = float(custom_gsd)
    except (ValueError, TypeError):
        return False, None, None, f"GSD must be a valid numeric value, got '{custom_gsd}'."

    # 2. Finite Check (NaN, +Inf, -Inf)
    if not math.isfinite(val):
        return False, None, None, f"GSD must be a finite real number, got '{val}'."

    # 3. Strictly Positive
    if val <= 0.0:
        return False, None, None, f"GSD must be strictly positive (> 0.0), got {val}."

    # 4. Plausibility Bounds
    if val < MIN_PLAUSIBLE_GSD_METERS:
        return (
            False,
            None,
            None,
            f"GSD value of {val:.6f} m/pixel is below plausible terrestrial Earth observation scale "
            f"(minimum supported: {MIN_PLAUSIBLE_GSD_METERS} m/pixel).",
        )

    if val > MAX_PLAUSIBLE_GSD_METERS:
        return (
            False,
            None,
            None,
            f"GSD value of {val:.2f} m/pixel exceeds plausible rooftop solar observation scale "
            f"(maximum supported: {MAX_PLAUSIBLE_GSD_METERS} m/pixel).",
        )

    # 5. Prohibited Generic Assumed Values (without substantial citation)
    clean_source = calibration_source.strip() if (calibration_source and isinstance(calibration_source, str)) else None
    rounded_val = round(val, 4)

    for prohibited_val, prohibited_desc in PROHIBITED_GENERIC_GSDS.items():
        if abs(val - prohibited_val) < 1e-5:
            # If candidate matches a prohibited assumption, require an explicit, meaningful source (>8 chars)
            if not clean_source or len(clean_source) < 8:
                return (
                    False,
                    None,
                    None,
                    f"Rejected unverified generic assumption: {prohibited_desc}. "
                    "An explicit calibration reference (e.g. sensor model, flight log) is required.",
                )

    effective_source = clean_source or "User-supplied Ground Sampling Distance (manual entry)"
    return True, val, effective_source, None


def resolve_image_calibration(
    geo_metadata: Optional[GeoMetadata] = None,
    custom_gsd: Optional[Any] = None,
    calibration_source: Optional[str] = None,
) -> ImageCalibration:
    """
    Resolves image calibration adhering strictly to precedence rules:
    1. If embedded GeoTIFF metadata has validated spatial scale -> AUTOMATIC calibration takes precedence.
    2. If automatic is unavailable and user-supplied GSD is provided -> Validate and apply MANUAL calibration.
    3. If neither is available -> NONE / UNAVAILABLE.

    Args:
        geo_metadata: Optional GeoMetadata extracted from raster tags.
        custom_gsd: Optional user-supplied GSD in meters/pixel.
        calibration_source: Optional citation or description for user GSD.

    Returns:
        ImageCalibration object with definitive status and provenance.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Path A: Automatic GeoTIFF Calibration
    has_auto_scale = (
        geo_metadata is not None
        and geo_metadata.is_georeferenced
        and geo_metadata.gsd_status == "VALIDATED"
        and geo_metadata.pixel_area_m2 is not None
        and geo_metadata.pixel_area_m2 > 0.0
    )

    if has_auto_scale:
        assert geo_metadata is not None
        eff_gsd = (
            geo_metadata.gsd_x_meters
            if geo_metadata.gsd_x_meters is not None
            else math.sqrt(geo_metadata.pixel_area_m2)
        )
        prov: Dict[str, Any] = {
            "resolved_at_utc": now_iso,
            "method": "automatic_geotiff_extraction",
            "crs_name": geo_metadata.crs_name,
            "crs_epsg": geo_metadata.crs_epsg,
            "is_projected": geo_metadata.is_projected,
            "is_rotated_or_sheared": geo_metadata.is_rotated_or_sheared,
            "metadata_provenance": geo_metadata.metadata_provenance,
        }
        if custom_gsd is not None:
            prov["user_override_notice"] = (
                "User-supplied custom_gsd was ignored because authentic georeferenced GeoTIFF "
                "metadata takes strict precedence."
            )

        return ImageCalibration(
            calibration_type="AUTOMATIC",
            calibration_status="EXTRACTED_METADATA",
            is_calibrated=True,
            gsd_meters=round(eff_gsd, 4),
            gsd_x_meters=round(geo_metadata.gsd_x_meters, 4) if geo_metadata.gsd_x_meters else round(eff_gsd, 4),
            gsd_y_meters=round(geo_metadata.gsd_y_meters, 4) if geo_metadata.gsd_y_meters else round(eff_gsd, 4),
            pixel_area_m2=round(geo_metadata.pixel_area_m2, 6),
            calibration_source=f"Embedded GeoTIFF ({geo_metadata.crs_name or 'Projected CRS'})",
            is_validated_metadata=True,
            is_user_supplied=False,
            rejection_reason=None,
            units="meters/pixel",
            provenance=prov,
        )

    # Path B: Manual User-Supplied Calibration (for ordinary rasters: PNG, JPEG, unreferenced TIFF)
    if custom_gsd is not None:
        is_val, gsd_float, eff_src, reject_reason = validate_user_gsd(
            custom_gsd=custom_gsd,
            calibration_source=calibration_source,
        )

        if not is_val:
            return ImageCalibration(
                calibration_type="MANUAL",
                calibration_status="REJECTED",
                is_calibrated=False,
                gsd_meters=None,
                gsd_x_meters=None,
                gsd_y_meters=None,
                pixel_area_m2=None,
                calibration_source=calibration_source or "Rejected user input",
                is_validated_metadata=False,
                is_user_supplied=True,
                rejection_reason=reject_reason,
                units="meters/pixel",
                provenance={
                    "resolved_at_utc": now_iso,
                    "method": "manual_entry_rejected",
                    "candidate_value": custom_gsd,
                    "rejection_reason": reject_reason,
                },
            )

        assert gsd_float is not None
        assert eff_src is not None
        pixel_area = float(gsd_float * gsd_float)

        return ImageCalibration(
            calibration_type="MANUAL",
            calibration_status="USER_SUPPLIED",
            is_calibrated=True,
            gsd_meters=round(gsd_float, 4),
            gsd_x_meters=round(gsd_float, 4),
            gsd_y_meters=round(gsd_float, 4),
            pixel_area_m2=round(pixel_area, 6),
            calibration_source=eff_src,
            is_validated_metadata=False,
            is_user_supplied=True,
            rejection_reason=None,
            units="meters/pixel",
            provenance={
                "resolved_at_utc": now_iso,
                "method": "manual_user_entry",
                "entered_gsd_m_per_pixel": gsd_float,
                "calibration_source": eff_src,
                "is_independently_verified": False,
                "disclaimer": (
                    "Calibration value was user-supplied and is not independently ground-truthed. "
                    "Physical area estimates directly depend on user accuracy."
                ),
            },
        )

    # Path C: No Calibration Available
    return ImageCalibration(
        calibration_type="NONE",
        calibration_status="UNAVAILABLE",
        is_calibrated=False,
        gsd_meters=None,
        gsd_x_meters=None,
        gsd_y_meters=None,
        pixel_area_m2=None,
        calibration_source="Unreferenced standard raster without calibration",
        is_validated_metadata=False,
        is_user_supplied=False,
        rejection_reason=None,
        units="meters/pixel",
        provenance={
            "resolved_at_utc": now_iso,
            "method": "none",
        },
    )


def calculate_calibrated_area(
    solar_pixels: int,
    total_pixels: int,
    calibration: ImageCalibration,
) -> AreaCalculationResult:
    """
    Computes physical area strictly adhering to the resolved ImageCalibration.
    Reuses existing PhysicalAreaCalculator.
    """
    if not calibration.is_calibrated or calibration.pixel_area_m2 is None:
        return PhysicalAreaCalculator.calculate(
            solar_pixels=solar_pixels,
            total_pixels=total_pixels,
            is_gsd_validated=False,
        )

    return PhysicalAreaCalculator.calculate(
        solar_pixels=solar_pixels,
        total_pixels=total_pixels,
        gsd_meters=calibration.gsd_meters,
        gsd_x_meters=calibration.gsd_x_meters,
        gsd_y_meters=calibration.gsd_y_meters,
        pixel_area_m2=calibration.pixel_area_m2,
        is_gsd_validated=True,
    )
