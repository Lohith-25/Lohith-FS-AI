"""
SolarMap-India — Geospatial Metadata & GeoTIFF Parser.

Provides safe, deterministic extraction of spatial metadata from GeoTIFF imagery:
- Parses TIFF tags (ModelPixelScaleTag 33550, ModelTiepointTag 33922, ModelTransformationTag 34264, GeoKeyDirectoryTag 34735).
- Resolves Coordinate Reference Systems (CRS) using pyproj without external GDAL/rasterio C-libraries.
- Accurately distinguishes Projected CRS (linear metric coordinates) from Geographic CRS (angular degrees).
- Calculates true pixel ground area via 2D affine transformation determinants (handling rotated/sheared transforms).
- Calculates local metric ground sampling distance (GSD) for Geographic CRS using geodesic ellipsoidal distance at scene latitude.
- Enforces strict scientific provenance: gracefully reports UNAVAILABLE / INSUFFICIENT_DATA when tags are missing, corrupt, or unverified.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import pyproj
from pyproj import CRS, Geod

# Standard OGC GeoTIFF TIFF Tag Identifiers
TAG_MODEL_PIXEL_SCALE = 33550       # (ScaleX, ScaleY, ScaleZ)
TAG_MODEL_TIEPOINT = 33922          # (I, J, K, X, Y, Z, ...)
TAG_MODEL_TRANSFORMATION = 34264    # 4x4 matrix (16 doubles)
TAG_GEO_KEY_DIRECTORY = 34735       # GeoKey directory entries
TAG_GEO_DOUBLE_PARAMS = 34736       # GeoDoubleParamsTag
TAG_GEO_ASCII_PARAMS = 34737        # GeoAsciiParamsTag

# Standard GeoKey Identifiers
GEOKEY_GT_MODEL_TYPE = 1024         # 1 = Projected, 2 = Geographic, 3 = Geocentric
GEOKEY_GT_RASTER_TYPE = 1025        # 1 = RasterPixelIsArea, 2 = RasterPixelIsPoint
GEOKEY_GEOGRAPHIC_TYPE = 2048       # EPSG Geographic CRS code (e.g. 4326)
GEOKEY_PROJECTED_CS_TYPE = 3072     # EPSG Projected CRS code (e.g. 32643)
GEOKEY_PROJ_LINEAR_UNITS = 3076     # 9001 = metre, 9002 = foot, etc.

# Recognized linear unit EPSG codes
LINEAR_UNIT_METRE = 9001
LINEAR_UNIT_FOOT = 9002
LINEAR_UNIT_US_SURVEY_FOOT = 9003

METERS_PER_FOOT = 0.3048
METERS_PER_US_SURVEY_FOOT = 1200.0 / 3937.0


@dataclass(frozen=True)
class GeoMetadata:
    """Represents extracted and validated geospatial metadata from raster imagery."""
    is_georeferenced: bool
    format: str
    width: int
    height: int
    crs_epsg: Optional[int]
    crs_name: Optional[str]
    is_projected: bool
    is_geographic: bool
    map_units: str
    transform_matrix: Optional[Tuple[float, ...]]  # (a, b, c, d, e, f) 2D affine
    is_rotated_or_sheared: bool
    pixel_width: Optional[float]                   # In native map units
    pixel_height: Optional[float]                  # In native map units
    gsd_x_meters: Optional[float]                  # Horizontal GSD in meters
    gsd_y_meters: Optional[float]                  # Vertical GSD in meters
    pixel_area_m2: Optional[float]                 # Exact ground surface area of single pixel
    gsd_status: str                                # "VALIDATED" | "UNAVAILABLE" | "INSUFFICIENT_DATA"
    metadata_provenance: str                       # Description of source tags
    status_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_geokey_directory(
    raw_keys: Union[List[int], Tuple[int, ...]]
) -> Dict[int, int]:
    """
    Decodes the GeoKeyDirectoryTag header and integer key-value entries.
    """
    if not raw_keys or len(raw_keys) < 4:
        return {}

    num_keys = int(raw_keys[3])
    entries: Dict[int, int] = {}

    for i in range(num_keys):
        idx = 4 + i * 4
        if idx + 3 >= len(raw_keys):
            break
        key_id = int(raw_keys[idx])
        key_location = int(raw_keys[idx + 1])
        key_val_offset = int(raw_keys[idx + 3])

        # If key_location == 0, the value is an integer stored directly in key_val_offset
        if key_location == 0:
            entries[key_id] = key_val_offset

    return entries


def extract_geospatial_metadata(
    image_path_or_bytes: Union[str, Path, bytes, Image.Image]
) -> GeoMetadata:
    """
    Extracts spatial reference, affine transformation, and GSD from imagery.

    Args:
        image_path_or_bytes: Path to image file, raw byte stream, or loaded PIL Image.

    Returns:
        GeoMetadata dataclass containing validated spatial parameters.
    """
    try:
        if isinstance(image_path_or_bytes, Image.Image):
            pil_img = image_path_or_bytes
        elif isinstance(image_path_or_bytes, (str, Path)):
            pil_img = Image.open(image_path_or_bytes)
        else:
            import io
            pil_img = Image.open(io.BytesIO(image_path_or_bytes))
    except Exception as exc:
        return GeoMetadata(
            is_georeferenced=False,
            format="UNKNOWN",
            width=0,
            height=0,
            crs_epsg=None,
            crs_name=None,
            is_projected=False,
            is_geographic=False,
            map_units="unknown",
            transform_matrix=None,
            is_rotated_or_sheared=False,
            pixel_width=None,
            pixel_height=None,
            gsd_x_meters=None,
            gsd_y_meters=None,
            pixel_area_m2=None,
            gsd_status="UNAVAILABLE",
            metadata_provenance="none",
            status_reason=f"Failed to open image for metadata inspection: {exc}",
        )

    img_format = str(pil_img.format or "UNKNOWN").upper()
    width, height = pil_img.size

    # Non-TIFF images (PNG, JPEG, BMP, WebP) do not standardly carry GeoTIFF tags
    if img_format not in ("TIFF", "TIF") or not hasattr(pil_img, "tag_v2"):
        return GeoMetadata(
            is_georeferenced=False,
            format=img_format,
            width=width,
            height=height,
            crs_epsg=None,
            crs_name=None,
            is_projected=False,
            is_geographic=False,
            map_units="pixels",
            transform_matrix=None,
            is_rotated_or_sheared=False,
            pixel_width=None,
            pixel_height=None,
            gsd_x_meters=None,
            gsd_y_meters=None,
            pixel_area_m2=None,
            gsd_status="UNAVAILABLE",
            metadata_provenance=f"standard_{img_format.lower()}_no_geotags",
            status_reason=f"{img_format} raster format contains no embedded GeoTIFF spatial metadata tags.",
        )

    tags = pil_img.tag_v2

    has_pixel_scale = TAG_MODEL_PIXEL_SCALE in tags
    has_tiepoints = TAG_MODEL_TIEPOINT in tags
    has_matrix = TAG_MODEL_TRANSFORMATION in tags
    has_geokeys = TAG_GEO_KEY_DIRECTORY in tags

    if not (has_pixel_scale or has_matrix) or not (has_tiepoints or has_matrix):
        return GeoMetadata(
            is_georeferenced=False,
            format="TIFF",
            width=width,
            height=height,
            crs_epsg=None,
            crs_name=None,
            is_projected=False,
            is_geographic=False,
            map_units="pixels",
            transform_matrix=None,
            is_rotated_or_sheared=False,
            pixel_width=None,
            pixel_height=None,
            gsd_x_meters=None,
            gsd_y_meters=None,
            pixel_area_m2=None,
            gsd_status="UNAVAILABLE",
            metadata_provenance="standard_tiff_unreferenced",
            status_reason="TIFF file does not contain GeoTIFF tiepoint, pixel scale, or transformation tags.",
        )

    # 1. Parse Coordinate Reference System (CRS)
    epsg_code: Optional[int] = None
    crs_name: Optional[str] = None
    is_projected: bool = False
    is_geographic: bool = False
    pyproj_crs: Optional[CRS] = None
    linear_unit_code: int = LINEAR_UNIT_METRE
    map_units: str = "unknown"

    if has_geokeys:
        raw_keys = tags[TAG_GEO_KEY_DIRECTORY]
        geokeys = _parse_geokey_directory(raw_keys)

        proj_code = geokeys.get(GEOKEY_PROJECTED_CS_TYPE)
        geo_code = geokeys.get(GEOKEY_GEOGRAPHIC_TYPE)
        linear_unit_code = geokeys.get(GEOKEY_PROJ_LINEAR_UNITS, LINEAR_UNIT_METRE)

        if proj_code and proj_code > 0:
            epsg_code = int(proj_code)
        elif geo_code and geo_code > 0:
            epsg_code = int(geo_code)

    if epsg_code is not None:
        try:
            pyproj_crs = CRS.from_epsg(epsg_code)
            crs_name = str(pyproj_crs.name)
            is_projected = bool(pyproj_crs.is_projected)
            is_geographic = bool(pyproj_crs.is_geographic)
            if is_projected:
                # Detect unit from projected CRS axis info
                axis_units = [a.unit_name for a in pyproj_crs.axis_info if hasattr(a, "unit_name")]
                map_units = axis_units[0] if axis_units else "metre"
            elif is_geographic:
                map_units = "degree"
        except Exception:
            pyproj_crs = None
            crs_name = f"EPSG:{epsg_code} (unresolved)"

    # 2. Extract Affine Transform Parameters: (a, b, c, d, e, f)
    # Mapping image coordinate (col, row) -> map coordinate (X, Y):
    # X = a * col + b * row + c
    # Y = d * col + e * row + f
    a: Optional[float] = None
    b: float = 0.0
    c: Optional[float] = None
    d: float = 0.0
    e: Optional[float] = None
    f: Optional[float] = None
    is_rotated_or_sheared = False

    if has_matrix:
        m = list(tags[TAG_MODEL_TRANSFORMATION])
        if len(m) >= 16:
            # 4x4 matrix:
            # [ a, b, 0, c ]
            # [ d, e, 0, f ]
            # [ 0, 0, 0, 0 ]
            # [ 0, 0, 0, 1 ]
            a, b, c = float(m[0]), float(m[1]), float(m[3])
            d, e, f = float(m[4]), float(m[5]), float(m[7])
            is_rotated_or_sheared = (abs(b) > 1e-9 or abs(d) > 1e-9)

    elif has_pixel_scale and has_tiepoints:
        scale_tag = tags[TAG_MODEL_PIXEL_SCALE]
        tiepoint_tag = tags[TAG_MODEL_TIEPOINT]

        if len(scale_tag) >= 2 and len(tiepoint_tag) >= 6:
            sx, sy = float(scale_tag[0]), float(scale_tag[1])
            # Tiepoint: (I, J, K, X, Y, Z)
            i0, j0 = float(tiepoint_tag[0]), float(tiepoint_tag[1])
            x0, y0 = float(tiepoint_tag[3]), float(tiepoint_tag[4])

            # Standard north-up GeoTIFF: X increases with col (sx), Y decreases with row (-sy)
            a = sx
            b = 0.0
            c = x0 - (i0 * sx)
            d = 0.0
            e = -sy
            f = y0 + (j0 * sy)
            is_rotated_or_sheared = False

    # Validate that transform is valid
    if a is None or e is None or c is None or f is None:
        return GeoMetadata(
            is_georeferenced=False,
            format="TIFF",
            width=width,
            height=height,
            crs_epsg=epsg_code,
            crs_name=crs_name,
            is_projected=is_projected,
            is_geographic=is_geographic,
            map_units=map_units,
            transform_matrix=None,
            is_rotated_or_sheared=False,
            pixel_width=None,
            pixel_height=None,
            gsd_x_meters=None,
            gsd_y_meters=None,
            pixel_area_m2=None,
            gsd_status="INSUFFICIENT_DATA",
            metadata_provenance="geotiff_corrupt_transform",
            status_reason="GeoTIFF metadata contains invalid or incomplete affine transformation parameters.",
        )

    # 3. Compute Determinant & Ground Sampling Distance (GSD)
    # The geometric area represented by 1 pixel in map units is |det(J)| = |a*e - b*d|
    det = abs(a * e - b * d)
    if det <= 1e-15:
        return GeoMetadata(
            is_georeferenced=False,
            format="TIFF",
            width=width,
            height=height,
            crs_epsg=epsg_code,
            crs_name=crs_name,
            is_projected=is_projected,
            is_geographic=is_geographic,
            map_units=map_units,
            transform_matrix=(a, b, c, d, e, f),
            is_rotated_or_sheared=is_rotated_or_sheared,
            pixel_width=math.hypot(a, d),
            pixel_height=math.hypot(b, e),
            gsd_x_meters=None,
            gsd_y_meters=None,
            pixel_area_m2=None,
            gsd_status="INSUFFICIENT_DATA",
            metadata_provenance="geotiff_singular_matrix",
            status_reason="Affine transformation matrix is degenerate (zero spatial area per pixel).",
        )

    pixel_w_map = math.hypot(a, d)
    pixel_h_map = math.hypot(b, e)

    # 4. Physical Area & GSD Evaluation based on CRS type
    if is_projected and pyproj_crs is not None:
        # Check linear unit conversion factor to meters
        unit_scale = 1.0
        if linear_unit_code == LINEAR_UNIT_FOOT or map_units.lower() in ("foot", "ft"):
            unit_scale = METERS_PER_FOOT
            map_units = "foot"
        elif linear_unit_code == LINEAR_UNIT_US_SURVEY_FOOT or map_units.lower() in ("us survey foot", "us_survey_foot"):
            unit_scale = METERS_PER_US_SURVEY_FOOT
            map_units = "us_survey_foot"
        elif map_units.lower() in ("metre", "meter", "m") or linear_unit_code == LINEAR_UNIT_METRE:
            unit_scale = 1.0
            map_units = "metre"

        gsd_x = float(pixel_w_map * unit_scale)
        gsd_y = float(pixel_h_map * unit_scale)
        pixel_area_m2 = float(det * (unit_scale ** 2))

        return GeoMetadata(
            is_georeferenced=True,
            format="GeoTIFF",
            width=width,
            height=height,
            crs_epsg=epsg_code,
            crs_name=crs_name,
            is_projected=True,
            is_geographic=False,
            map_units=map_units,
            transform_matrix=(a, b, c, d, e, f),
            is_rotated_or_sheared=is_rotated_or_sheared,
            pixel_width=pixel_w_map,
            pixel_height=pixel_h_map,
            gsd_x_meters=gsd_x,
            gsd_y_meters=gsd_y,
            pixel_area_m2=pixel_area_m2,
            gsd_status="VALIDATED",
            metadata_provenance="geotiff_projected_crs",
            status_reason="Spatial scale and pixel area calibrated from projected metric coordinate system.",
        )

    elif is_geographic and pyproj_crs is not None:
        # In a Geographic CRS (e.g. WGS84 EPSG:4326), coordinates are in degrees
        # Longitude distance varies with latitude (cos(lat))
        # Compute scene centroid latitude from affine transform
        center_row = height / 2.0
        center_col = width / 2.0
        center_lon = a * center_col + b * center_row + c
        center_lat = d * center_col + e * center_row + f

        if -90.0 <= center_lat <= 90.0 and -180.0 <= center_lon <= 180.0:
            geod = Geod(ellps="WGS84")
            # Calculate distance in meters for dx (col step) and dy (row step)
            _, _, dist_x = geod.inv(center_lon, center_lat, center_lon + pixel_w_map, center_lat)
            _, _, dist_y = geod.inv(center_lon, center_lat, center_lon, center_lat + pixel_h_map)

            gsd_x = float(dist_x)
            gsd_y = float(dist_y)
            # Area in m^2 accounting for local latitude scale factor
            pixel_area_m2 = float(gsd_x * gsd_y)

            return GeoMetadata(
                is_georeferenced=True,
                format="GeoTIFF",
                width=width,
                height=height,
                crs_epsg=epsg_code,
                crs_name=crs_name,
                is_projected=False,
                is_geographic=True,
                map_units="degree",
                transform_matrix=(a, b, c, d, e, f),
                is_rotated_or_sheared=is_rotated_or_sheared,
                pixel_width=pixel_w_map,
                pixel_height=pixel_h_map,
                gsd_x_meters=gsd_x,
                gsd_y_meters=gsd_y,
                pixel_area_m2=pixel_area_m2,
                gsd_status="VALIDATED",
                metadata_provenance="geotiff_geographic_crs_ellipsoidal",
                status_reason=(
                    f"Geographic CRS (degrees) converted to metric spatial scale via WGS84 geodesic "
                    f"ellipsoid calculations at scene centroid latitude ({center_lat:.4f}°)."
                ),
            )
        else:
            return GeoMetadata(
                is_georeferenced=False,
                format="GeoTIFF",
                width=width,
                height=height,
                crs_epsg=epsg_code,
                crs_name=crs_name,
                is_projected=False,
                is_geographic=True,
                map_units="degree",
                transform_matrix=(a, b, c, d, e, f),
                is_rotated_or_sheared=is_rotated_or_sheared,
                pixel_width=pixel_w_map,
                pixel_height=pixel_h_map,
                gsd_x_meters=None,
                gsd_y_meters=None,
                pixel_area_m2=None,
                gsd_status="INSUFFICIENT_DATA",
                metadata_provenance="geotiff_invalid_geographic_coords",
                status_reason=f"Geographic coordinates out of valid range (lat={center_lat}, lon={center_lon}).",
            )

    else:
        # GeoTIFF has tiepoint and pixel scale, but missing or unresolvable CRS
        return GeoMetadata(
            is_georeferenced=False,
            format="GeoTIFF",
            width=width,
            height=height,
            crs_epsg=epsg_code,
            crs_name=crs_name,
            is_projected=False,
            is_geographic=False,
            map_units="unknown",
            transform_matrix=(a, b, c, d, e, f),
            is_rotated_or_sheared=is_rotated_or_sheared,
            pixel_width=pixel_w_map,
            pixel_height=pixel_h_map,
            gsd_x_meters=None,
            gsd_y_meters=None,
            pixel_area_m2=None,
            gsd_status="INSUFFICIENT_DATA",
            metadata_provenance="geotiff_missing_crs",
            status_reason=(
                "GeoTIFF contains spatial scale or tiepoints, but CRS is missing, unrecognized, "
                "or non-standard. Unit of measurement cannot be verified."
            ),
        )
