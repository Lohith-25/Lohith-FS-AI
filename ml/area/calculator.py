"""
SolarMap-India — Physical Area Calculator.

Computes physical solar panel surface area (m²) strictly when a validated GSD is present.
Enforces that when GSD is UNAVAILABLE, solar_area_m2 = null and area_status = "insufficient_data".
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from ml.area.gsd_provider import GSDResult


@dataclass
class PhysicalAreaResult:
    """Outcome of GSD-based physical area calculation."""
    solar_pixels: int
    total_pixels: Optional[int]
    pixel_coverage_percent: Optional[float]
    gsd_m_per_pixel: Optional[float]
    pixel_area_m2: Optional[float]
    solar_area_m2: Optional[float]
    area_status: str                     # "calculated" | "insufficient_data"
    gsd_status: str                      # "VALIDATED" | "UNAVAILABLE"
    source: str
    provenance: Dict[str, Any]
    status_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_physical_area(
    solar_pixels: int,
    gsd_result: GSDResult,
    total_pixels: Optional[int] = None,
) -> PhysicalAreaResult:
    """
    Computes physical solar surface area from pixel counts and GSDResult.

    Args:
        solar_pixels: Total count of foreground solar pixels (from Phase 2 analytics).
        gsd_result: GSDResult object from GSDProvider.
        total_pixels: Optional total raster pixel count (e.g. 409,600).

    Returns:
        PhysicalAreaResult with area_status "calculated" or "insufficient_data".
    """
    if not isinstance(solar_pixels, int) or solar_pixels < 0:
        raise ValueError(f"Solar pixels must be a non-negative integer, got {solar_pixels}")

    cov_pct = None
    if total_pixels is not None and total_pixels > 0:
        cov_pct = round((solar_pixels / total_pixels) * 100.0, 4)

    # If GSD is UNAVAILABLE:
    if gsd_result.status != "VALIDATED" or gsd_result.gsd_m_per_pixel is None:
        return PhysicalAreaResult(
            solar_pixels=solar_pixels,
            total_pixels=total_pixels,
            pixel_coverage_percent=cov_pct,
            gsd_m_per_pixel=None,
            pixel_area_m2=None,
            solar_area_m2=None,
            area_status="insufficient_data",
            gsd_status=gsd_result.status,
            source=gsd_result.source,
            provenance=gsd_result.provenance,
            status_reason=(
                gsd_result.rejection_reason
                or "GSD is UNAVAILABLE; physical area cannot be derived without calibrated spatial scale."
            ),
        )

    # If GSD is VALIDATED:
    gsd = float(gsd_result.gsd_m_per_pixel)
    pixel_area_m2 = gsd * gsd
    solar_area_m2 = float(solar_pixels * pixel_area_m2)

    return PhysicalAreaResult(
        solar_pixels=solar_pixels,
        total_pixels=total_pixels,
        pixel_coverage_percent=cov_pct,
        gsd_m_per_pixel=gsd,
        pixel_area_m2=pixel_area_m2,
        solar_area_m2=solar_area_m2,
        area_status="calculated",
        gsd_status="VALIDATED",
        source=gsd_result.source,
        provenance=gsd_result.provenance,
        status_reason="Physical solar area calculated from validated Ground Sampling Distance.",
    )
