"""
SolarMap-India — Image & Dataset Metadata Inspector.

Inspects raster files, EXIF, GeoTIFF tags, sidecars, and COCO annotations
to evaluate the presence or absence of geospatial georeferencing and physical scale.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from PIL.ExifTags import TAGS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def inspect_image_metadata(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Examines an individual image file for geospatial tags, EXIF, and world files.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")

    with Image.open(path) as img:
        img_format = img.format
        mode = img.mode
        width, height = img.size
        info_keys = list(img.info.keys())
        exif = img.getexif()

    exif_tags: Dict[str, Any] = {}
    if exif:
        for tag_id, val in exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            exif_tags[tag_name] = str(val)

    # Check for adjacent world files (.tfw, .pgw, .jgw, .wld)
    world_extensions = [".pgw", ".tfw", ".jgw", ".wld"]
    world_file_found = None
    for ext in world_extensions:
        wf = path.with_suffix(ext)
        if wf.is_file():
            world_file_found = str(wf)
            break

    # Check for GeoTIFF tags
    has_geotiff_tags = bool(
        img_format in ("TIFF", "GeoTIFF") and any("geo" in k.lower() for k in info_keys)
    )

    return {
        "filename": path.name,
        "format": img_format,
        "mode": mode,
        "width": width,
        "height": height,
        "has_exif": bool(len(exif_tags) > 0),
        "exif_tags_count": len(exif_tags),
        "has_gps_exif": "GPSInfo" in exif_tags,
        "has_geotiff_tags": has_geotiff_tags,
        "world_file": world_file_found,
        "info_keys": info_keys,
        "crs": None,
        "affine_transform": None,
        "meters_per_pixel": None,
    }


def inspect_dataset_directory(
    image_dir: Union[str, Path],
    sample_limit: int = 50,
) -> Dict[str, Any]:
    """
    Audits a collection of images for any georeferencing sidecars or embedded metadata.
    """
    dir_path = Path(image_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path.resolve()}")

    image_files = sorted(
        [p for p in dir_path.glob("*.png")] + [p for p in dir_path.glob("*.jpg")]
    )
    total_images = len(image_files)

    world_files = (
        list(dir_path.glob("*.pgw"))
        + list(dir_path.glob("*.tfw"))
        + list(dir_path.glob("*.jgw"))
        + list(dir_path.glob("*.wld"))
    )

    # Sample images for EXIF audit
    samples_checked = 0
    gps_found = 0
    geotiff_found = 0

    for img_p in image_files[:sample_limit]:
        info = inspect_image_metadata(img_p)
        samples_checked += 1
        if info["has_gps_exif"]:
            gps_found += 1
        if info["has_geotiff_tags"]:
            geotiff_found += 1

    return {
        "image_directory": str(dir_path.resolve()),
        "total_images": total_images,
        "world_files_count": len(world_files),
        "samples_audited": samples_checked,
        "images_with_gps_exif": gps_found,
        "images_with_geotiff_tags": geotiff_found,
        "physical_pixel_scale_available": False,
        "raster_transform_available": False,
    }
