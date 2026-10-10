"""
SolarMap-India — Geospatial & Physical Area Foundation Package.
"""

from ml.geospatial.area_calculator import (
    calculate_physical_area,
    evaluate_component_centroids,
)
from ml.geospatial.coordinate_mapping import (
    CoordinateRecord,
    GeospatialCatalog,
)
from ml.geospatial.metadata_inspector import (
    inspect_dataset_directory,
    inspect_image_metadata,
)
from ml.geospatial.report import (
    generate_geospatial_status_report,
)
from ml.geospatial.spatial_scale import (
    SpatialScale,
    get_dataset_spatial_scale,
    validate_custom_scale,
)

__all__ = [
    "CoordinateRecord",
    "GeospatialCatalog",
    "SpatialScale",
    "get_dataset_spatial_scale",
    "validate_custom_scale",
    "calculate_physical_area",
    "evaluate_component_centroids",
    "inspect_image_metadata",
    "inspect_dataset_directory",
    "generate_geospatial_status_report",
]
