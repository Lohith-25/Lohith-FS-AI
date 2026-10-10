"""
SolarMap-India — Solar Segmentation Analytics Package.
"""

from ml.analytics.components import (
    ComponentStatistics,
    SolarRegion,
    analyze_components,
    extract_connected_components,
)
from ml.analytics.mask_analysis import (
    MaskPixelStats,
    compute_pixel_statistics,
    load_and_validate_mask,
    run_segmentation_analytics,
)
from ml.analytics.report import (
    build_report_dictionary,
    save_components_csv,
    save_json_report,
)
from ml.analytics.visualization import (
    create_colored_components_map,
    draw_bounding_boxes,
)

__all__ = [
    "SolarRegion",
    "ComponentStatistics",
    "MaskPixelStats",
    "load_and_validate_mask",
    "compute_pixel_statistics",
    "extract_connected_components",
    "analyze_components",
    "run_segmentation_analytics",
    "build_report_dictionary",
    "save_json_report",
    "save_components_csv",
    "draw_bounding_boxes",
    "create_colored_components_map",
]
