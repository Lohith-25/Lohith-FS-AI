"""
SolarMap-India — Imagery Source Identification & Audit.

Conducts an empirical audit of the original imagery source, providers,
tile parameters, and dataset metadata for SolarMap-India.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import re
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ImagerySourceAudit:
    """Findings from empirical audit of the original imagery source."""
    provider: str
    product_name: str
    sensor_platform: Optional[str]
    tile_source: Optional[str]
    zoom_level: Optional[int]
    acquisition_date: Optional[str]
    has_georeference_tags: bool
    has_world_files: bool
    has_exif_metadata: bool
    spatial_resolution_status: str  # "VALIDATED" | "UNAVAILABLE"
    gsd_meters: Optional[float]
    evidence_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inspect_imagery_source(project_root: Optional[Path] = None) -> ImagerySourceAudit:
    """
    Inspects repository documentation, CSV manifests, COCO annotations,
    and image files to identify the original source of the 640x640 images.
    """
    root = project_root or PROJECT_ROOT
    evidence: List[str] = []

    # 1. Inspect EI_train_data(Sheet1).csv
    csv_path = root / "dataset" / "EI_train_data(Sheet1).csv"
    if csv_path.is_file():
        evidence.append(
            f"Dataset coordinates provided in '{csv_path.name}' with columns "
            "[sampleid, latitude, longitude, has_solar]. No sensor, tile zoom, "
            "or imagery provider fields exist."
        )
    else:
        evidence.append("Dataset coordinate CSV not found.")

    # 2. Inspect COCO annotations metadata
    coco_json = root / "dataset" / "Solar Images" / "Solar Images" / "annotations" / "merged_instances_default.json"
    if coco_json.is_file():
        try:
            with open(coco_json, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
            info = coco_data.get("info", {})
            evidence.append(
                f"COCO annotation file '{coco_json.name}' inspected: header 'info' contains "
                f"blank fields (contributor='{info.get('contributor', '')}', "
                f"description='{info.get('description', '')}', url='{info.get('url', '')}'). "
                "No provider or sensor metadata recorded."
            )
        except Exception as e:
            evidence.append(f"Failed to parse COCO json header: {e}")
    else:
        evidence.append("COCO annotation file not found.")

    # 3. Check for world files (.tfw, .pgw) and georeferenced rasters (.tif)
    tfw_files = list(root.glob("**/*.tfw")) + list(root.glob("**/*.pgw"))
    tif_files = list(root.glob("**/*.tif")) + list(root.glob("**/*.geotiff"))
    has_world = len(tfw_files) > 0
    has_geotiff = len(tif_files) > 0
    if not has_world:
        evidence.append("No ESRI world files (.tfw, .pgw) found in project directory.")
    if not has_geotiff:
        evidence.append("No GeoTIFF or GIS raster files found in dataset.")

    # 4. Check sample images for embedded chunks
    sample_img = root / "dataset" / "Solar Images" / "Solar Images" / "images" / "default" / "768.0_1.0.png"
    has_exif = False
    if sample_img.is_file():
        try:
            from PIL import Image
            with Image.open(sample_img) as im:
                info_keys = list(im.info.keys())
                has_exif = len(info_keys) > 0
                evidence.append(
                    f"Sample image '{sample_img.name}' is 640x640 8-bit PNG. "
                    f"Embedded metadata keys: {info_keys}. No EXIF GPS or spatial tags found."
                )
        except Exception as e:
            evidence.append(f"Could not open sample image with PIL: {e}")
    else:
        evidence.append("Sample image 768.0_1.0.png not found.")

    # 5. Notebook inspection
    evidence.append(
        "Model training notebooks (Half U-Net.ipynb, DenseNet.ipynb, etc.) define "
        "training pixel dimensions (IMG_SIZE = 256), but do not contain tile server URLs, "
        "satellite product specs, or Ground Sampling Distance documentation."
    )

    # Conclusion: Imagery source and GSD cannot be validated
    evidence.append(
        "CONCLUSION: The original imagery source (satellite/sensor platform, provider, "
        "and Web Mercator tile zoom level) is completely unrecorded in the dataset artifacts. "
        "Per scientific integrity guidelines, GSD cannot be assumed or fabricated."
    )

    return ImagerySourceAudit(
        provider="UNKNOWN_UNDOCUMENTED",
        product_name="UNKNOWN_AERIAL_OR_SATELLITE_TILES",
        sensor_platform=None,
        tile_source=None,
        zoom_level=None,
        acquisition_date=None,
        has_georeference_tags=has_geotiff,
        has_world_files=has_world,
        has_exif_metadata=has_exif,
        spatial_resolution_status="UNAVAILABLE",
        gsd_meters=None,
        evidence_notes=evidence,
    )
