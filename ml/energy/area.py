"""
SolarMap-India — Physical Solar Area Calculator.

Computes physical solar surface area (m²) from segmented solar pixels and validated GSD or pixel area.
Strictly returns status 'insufficient_data' and solar_area_m2 = null if GSD or pixel area is absent, unverified, or non-projected.
Supports:
- Isotropic GSD (meters per pixel)
- Anisotropic GSD (meters_x, meters_y)
- Exact affine determinant pixel area (m² per pixel), correctly handling rotated or sheared raster transformations.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional

from ml.energy.validation import EnergyValidator


@dataclass
class AreaCalculationResult:
    """Outcome of physical area calculation."""
    solar_pixels: int
    total_pixels: int
    pixel_coverage_fraction: float
    pixel_coverage_percent: float
    gsd_status: str                     # "VALIDATED" | "UNAVAILABLE" | "INSUFFICIENT_DATA"
    gsd_meters: Optional[float]
    gsd_x_meters: Optional[float]
    gsd_y_meters: Optional[float]
    pixel_area_m2: Optional[float]
    solar_area_m2: Optional[float]
    physical_area_status: str           # "calculated" | "insufficient_data"
    status_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def insufficient_reason(self) -> Optional[str]:
        return self.status_reason if self.physical_area_status != "calculated" else None


class PhysicalAreaCalculator:
    """
    Computes physical surface area strictly when validated spatial scale (GSD or pixel area) is present.
    """

    @classmethod
    def calculate(
        cls,
        solar_pixels: int,
        total_pixels: int,
        gsd_meters: Optional[float] = None,
        gsd_x_meters: Optional[float] = None,
        gsd_y_meters: Optional[float] = None,
        pixel_area_m2: Optional[float] = None,
        is_gsd_validated: bool = False,
    ) -> AreaCalculationResult:
        """
        Calculates physical area in m².

        Args:
            solar_pixels: Number of pixels classified as solar panel.
            total_pixels: Total pixel count of the raster image.
            gsd_meters: Ground Sampling Distance (meters/pixel) for isotropic pixels.
            gsd_x_meters: Horizontal GSD for anisotropic pixels.
            gsd_y_meters: Vertical GSD for anisotropic pixels.
            pixel_area_m2: Direct pixel area in m² (e.g. from affine matrix determinant).
            is_gsd_validated: Explicit flag confirming GSD / pixel area is from verified imagery metadata.
        """
        # Validate pixel counts
        ok_pix, err_pix = EnergyValidator.validate_solar_pixels(solar_pixels, total_pixels)
        if not ok_pix:
            raise ValueError(f"Invalid pixel parameters: {err_pix}")

        cov_frac = float(solar_pixels / total_pixels) if total_pixels > 0 else 0.0
        cov_pct = cov_frac * 100.0

        # Determine effective GSD & pixel area availability
        has_direct_area = (pixel_area_m2 is not None and pixel_area_m2 > 0)
        has_anisotropic = (gsd_x_meters is not None and gsd_y_meters is not None)
        has_isotropic = (gsd_meters is not None)

        # Enforce validation requirement
        if not is_gsd_validated or (not has_direct_area and not has_isotropic and not has_anisotropic):
            return AreaCalculationResult(
                solar_pixels=solar_pixels,
                total_pixels=total_pixels,
                pixel_coverage_fraction=cov_frac,
                pixel_coverage_percent=cov_pct,
                gsd_status="UNAVAILABLE",
                gsd_meters=None,
                gsd_x_meters=None,
                gsd_y_meters=None,
                pixel_area_m2=None,
                solar_area_m2=None,
                physical_area_status="insufficient_data",
                status_reason=(
                    "Ground Sampling Distance (GSD) is unavailable or unverified for imagery. "
                    "Physical area cannot be derived without calibrated spatial scale."
                ),
            )

        # If direct pixel area (e.g. from affine determinant) is provided:
        if has_direct_area:
            if pixel_area_m2 is None or pixel_area_m2 <= 0.0:
                return AreaCalculationResult(
                    solar_pixels=solar_pixels,
                    total_pixels=total_pixels,
                    pixel_coverage_fraction=cov_frac,
                    pixel_coverage_percent=cov_pct,
                    gsd_status="UNAVAILABLE",
                    gsd_meters=None,
                    gsd_x_meters=None,
                    gsd_y_meters=None,
                    pixel_area_m2=None,
                    solar_area_m2=None,
                    physical_area_status="insufficient_data",
                    status_reason="Pixel area must be strictly positive.",
                )

            calc_area = float(solar_pixels * pixel_area_m2)
            eff_gsd = math.sqrt(pixel_area_m2)
            gx = float(gsd_x_meters) if gsd_x_meters is not None else eff_gsd
            gy = float(gsd_y_meters) if gsd_y_meters is not None else eff_gsd

            return AreaCalculationResult(
                solar_pixels=solar_pixels,
                total_pixels=total_pixels,
                pixel_coverage_fraction=cov_frac,
                pixel_coverage_percent=cov_pct,
                gsd_status="VALIDATED",
                gsd_meters=round(eff_gsd, 4),
                gsd_x_meters=round(gx, 4),
                gsd_y_meters=round(gy, 4),
                pixel_area_m2=round(float(pixel_area_m2), 6),
                solar_area_m2=round(calc_area, 2),
                physical_area_status="calculated",
                status_reason="Physical area computed from validated spatial scale and pixel segmentation.",
            )

        # Validate numeric GSD
        if has_anisotropic:
            ok_x, err_x = EnergyValidator.validate_gsd(gsd_x_meters)
            ok_y, err_y = EnergyValidator.validate_gsd(gsd_y_meters)
            if not ok_x or not ok_y:
                return AreaCalculationResult(
                    solar_pixels=solar_pixels,
                    total_pixels=total_pixels,
                    pixel_coverage_fraction=cov_frac,
                    pixel_coverage_percent=cov_pct,
                    gsd_status="UNAVAILABLE",
                    gsd_meters=None,
                    gsd_x_meters=gsd_x_meters,
                    gsd_y_meters=gsd_y_meters,
                    pixel_area_m2=None,
                    solar_area_m2=None,
                    physical_area_status="insufficient_data",
                    status_reason=f"Invalid anisotropic GSD: {err_x or err_y}",
                )
            pixel_area = float(gsd_x_meters * gsd_y_meters)
            eff_gsd = (float(gsd_x_meters) + float(gsd_y_meters)) / 2.0
            eff_gx = float(gsd_x_meters)
            eff_gy = float(gsd_y_meters)
        else:
            ok_iso, err_iso = EnergyValidator.validate_gsd(gsd_meters)
            if not ok_iso:
                return AreaCalculationResult(
                    solar_pixels=solar_pixels,
                    total_pixels=total_pixels,
                    pixel_coverage_fraction=cov_frac,
                    pixel_coverage_percent=cov_pct,
                    gsd_status="UNAVAILABLE",
                    gsd_meters=gsd_meters,
                    gsd_x_meters=None,
                    gsd_y_meters=None,
                    pixel_area_m2=None,
                    solar_area_m2=None,
                    physical_area_status="insufficient_data",
                    status_reason=f"Invalid isotropic GSD: {err_iso}",
                )
            pixel_area = float(gsd_meters ** 2)
            eff_gsd = float(gsd_meters)
            eff_gx = float(gsd_meters)
            eff_gy = float(gsd_meters)

        # Calculate physical solar area (solar_pixel_count * pixel_area_m2)
        solar_area = float(solar_pixels * pixel_area)

        return AreaCalculationResult(
            solar_pixels=solar_pixels,
            total_pixels=total_pixels,
            pixel_coverage_fraction=cov_frac,
            pixel_coverage_percent=cov_pct,
            gsd_status="VALIDATED",
            gsd_meters=round(eff_gsd, 4),
            gsd_x_meters=round(eff_gx, 4),
            gsd_y_meters=round(eff_gy, 4),
            pixel_area_m2=round(pixel_area, 6),
            solar_area_m2=round(solar_area, 2),
            physical_area_status="calculated",
            status_reason="Physical area computed from validated spatial scale and pixel segmentation.",
        )
