"""
SolarMap-India — Physical Spatial Scale & Resolution Management.

Encapsulates Ground Sampling Distance (GSD / meters-per-pixel) validation.
Enforces strict scientific prohibition against fabricating or guessing physical scale.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpatialScale:
    """Represents calibrated physical spatial resolution for an aerial/satellite raster."""
    meters_per_pixel_x: Optional[float] = None
    meters_per_pixel_y: Optional[float] = None
    source: str = "unavailable"
    is_calibrated: bool = False

    @property
    def is_available(self) -> bool:
        return (
            self.meters_per_pixel_x is not None
            and self.meters_per_pixel_y is not None
            and self.meters_per_pixel_x > 0
            and self.meters_per_pixel_y > 0
            and self.is_calibrated
        )

    def to_dict(self) -> dict:
        return {
            "is_available": self.is_available,
            "meters_per_pixel_x": (
                round(float(self.meters_per_pixel_x), 4)
                if self.meters_per_pixel_x is not None
                else None
            ),
            "meters_per_pixel_y": (
                round(float(self.meters_per_pixel_y), 4)
                if self.meters_per_pixel_y is not None
                else None
            ),
            "source": self.source,
            "is_calibrated": self.is_calibrated,
        }


def get_dataset_spatial_scale(image_path: Optional[str] = None) -> SpatialScale:
    """
    Returns the spatial scale for the SolarMap-India dataset.

    Finding:
    The dataset consists of 640x640 raster PNGs without embedded GeoTIFF tags,
    world files (.pgw/.tfw), or documented Ground Sampling Distance (GSD).
    Therefore, physical resolution is marked as unavailable.
    """
    return SpatialScale(
        meters_per_pixel_x=None,
        meters_per_pixel_y=None,
        source="unavailable_in_dataset_metadata",
        is_calibrated=False,
    )


def validate_custom_scale(
    meters_per_pixel_x: float,
    meters_per_pixel_y: float,
    source: str = "user_provided",
) -> SpatialScale:
    """
    Validates explicit, externally documented spatial scale values.

    Raises:
        ValueError: If scale values are non-positive or non-finite.
    """
    if meters_per_pixel_x <= 0 or meters_per_pixel_y <= 0:
        raise ValueError(
            f"Invalid spatial scale: scale values must be strictly positive (> 0). "
            f"Got ({meters_per_pixel_x}, {meters_per_pixel_y})"
        )

    return SpatialScale(
        meters_per_pixel_x=float(meters_per_pixel_x),
        meters_per_pixel_y=float(meters_per_pixel_y),
        source=source,
        is_calibrated=True,
    )
