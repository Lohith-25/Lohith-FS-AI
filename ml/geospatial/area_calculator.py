"""
SolarMap-India — Physical Area Calculation & Centroid Georeferencing API.

Rules:
1. If spatial scale is validated and available -> calculate physical area in m^2.
2. If spatial scale is unavailable -> return status='insufficient_data' and preserve pixel counts.
3. If spatial scale is invalid (<= 0) -> raise clear validation error.
4. Never silently use a default/fallback scale.
5. Do not convert pixel centroids to geographic coordinates without an affine transform matrix.
"""

from typing import Any, Dict, List, Optional, Union

from ml.analytics.components import SolarRegion
from ml.geospatial.spatial_scale import SpatialScale


def calculate_physical_area(
    pixel_count: int,
    meters_per_pixel_x: Optional[float] = None,
    meters_per_pixel_y: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes physical area in square meters strictly when validated scale is provided.

    Args:
        pixel_count: Number of pixels (int).
        meters_per_pixel_x: Horizontal pixel scale in meters/pixel (optional).
        meters_per_pixel_y: Vertical pixel scale in meters/pixel (optional).

    Returns:
        dict with status, area_m2, area_pixels, and explanation.

    Raises:
        ValueError: If scale values are provided but non-positive (<= 0).
    """
    if pixel_count < 0:
        raise ValueError(f"Pixel count cannot be negative: {pixel_count}")

    # Check if scale is unavailable
    if meters_per_pixel_x is None or meters_per_pixel_y is None:
        return {
            "status": "insufficient_data",
            "area_m2": None,
            "area_pixels": int(pixel_count),
            "reason": (
                "Physical spatial scale (meters/pixel) is not available in the dataset metadata. "
                "Physical area cannot be scientifically computed without calibrated resolution."
            ),
        }

    # Validate provided scale values
    if meters_per_pixel_x <= 0 or meters_per_pixel_y <= 0:
        raise ValueError(
            f"Invalid physical scale: meters_per_pixel must be strictly positive (> 0). "
            f"Got x={meters_per_pixel_x}, y={meters_per_pixel_y}"
        )

    area_m2 = float(pixel_count * meters_per_pixel_x * meters_per_pixel_y)

    return {
        "status": "calculated",
        "area_m2": round(area_m2, 4),
        "area_pixels": int(pixel_count),
        "scale_x": float(meters_per_pixel_x),
        "scale_y": float(meters_per_pixel_y),
        "unit": "square_meters",
    }


def evaluate_component_centroids(
    regions: List[SolarRegion],
    affine_transform: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluates geographic coordinate availability for individual solar components.

    In the absence of an affine raster transform matrix (e.g. from GeoTIFF or world file),
    component centroids remain strictly in 2D image pixel space.
    """
    evaluated: List[Dict[str, Any]] = []

    for reg in regions:
        rec = {
            "id": reg.id,
            "area_pixels": reg.area_pixels,
            "centroid_pixel_x": reg.centroid_x,
            "centroid_pixel_y": reg.centroid_y,
            "geographic_coordinates_available": False,
            "geographic_latitude": None,
            "geographic_longitude": None,
            "status": "unavailable_no_affine_transform",
        }

        if affine_transform is not None:
            # If transform is provided in future phases, apply it
            pass

        evaluated.append(rec)

    return evaluated
