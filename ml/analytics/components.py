"""
SolarMap-India — Connected Component Analysis & Region Statistics.

Extracts, characterizes, filters, and summarizes individual solar panel regions
strictly from the binary segmentation mask in 2D image pixel space.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SolarRegion:
    """Represents a single connected solar panel region in image pixel coordinates."""
    id: int
    area_pixels: int
    x: int
    y: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "area_pixels": int(self.area_pixels),
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "centroid_x": round(float(self.centroid_x), 2),
            "centroid_y": round(float(self.centroid_y), 2),
        }


@dataclass
class ComponentStatistics:
    """Aggregate statistics for connected solar regions."""
    min_component_area_pixels: int
    raw_count: int
    filtered_count: int
    largest_area_pixels: Optional[int]
    smallest_area_pixels: Optional[int]
    mean_area_pixels: Optional[float]
    median_area_pixels: Optional[float]
    raw_solar_pixels: int
    filtered_solar_pixels: int
    retained_percentage: float
    region_density_per_pixel: float
    regions: List[SolarRegion]

    def to_dict(self) -> dict:
        return {
            "min_component_area_pixels": int(self.min_component_area_pixels),
            "raw_count": int(self.raw_count),
            "filtered_count": int(self.filtered_count),
            "largest_area_pixels": int(self.largest_area_pixels) if self.largest_area_pixels is not None else None,
            "smallest_area_pixels": int(self.smallest_area_pixels) if self.smallest_area_pixels is not None else None,
            "mean_area_pixels": round(float(self.mean_area_pixels), 2) if self.mean_area_pixels is not None else None,
            "median_area_pixels": round(float(self.median_area_pixels), 2) if self.median_area_pixels is not None else None,
            "raw_solar_pixels": int(self.raw_solar_pixels),
            "filtered_solar_pixels": int(self.filtered_solar_pixels),
            "retained_percentage": round(float(self.retained_percentage), 2),
            "region_density_per_pixel": float(self.region_density_per_pixel),
        }


def extract_connected_components(
    binary_mask: np.ndarray,
    connectivity: int = 8,
) -> Tuple[List[SolarRegion], np.ndarray]:
    """
    Identifies connected solar regions from a binary mask {0, 1}.

    Args:
        binary_mask: 2D uint8 numpy array with values strictly {0, 1}.
        connectivity: Pixel neighborhood connectivity (4 or 8). Default is 8.

    Returns:
        Tuple of (List[SolarRegion], label_map 2D array).

    Raises:
        ValueError: If binary_mask is not 2D or contains non-binary values.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got array with shape {binary_mask.shape}")

    unique_vals = set(np.unique(binary_mask))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Mask must contain strictly values {{0, 1}}, got {unique_vals}")

    h, w = binary_mask.shape
    if h == 0 or w == 0:
        return [], np.zeros((h, w), dtype=np.int32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=connectivity,
    )

    regions: List[SolarRegion] = []

    # Label 0 is background; iterate through solar components 1 .. num_labels - 1
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx = float(centroids[label_id, 0])
        cy = float(centroids[label_id, 1])

        # Scientific validations
        if area <= 0:
            raise ValueError(f"Invalid component area {area} for label {label_id}")

        if not (0 <= x < w and 0 <= y < h and 0 < x + width <= w and 0 < y + height <= h):
            raise ValueError(
                f"Component {label_id} bounding box [{x}, {y}, {width}, {height}] exceeds image boundaries ({w}x{h})"
            )

        if not (0.0 <= cx < w and 0.0 <= cy < h):
            raise ValueError(
                f"Component {label_id} centroid [{cx}, {cy}] lies outside image boundaries ({w}x{h})"
            )

        regions.append(
            SolarRegion(
                id=label_id,
                area_pixels=area,
                x=x,
                y=y,
                width=width,
                height=height,
                centroid_x=cx,
                centroid_y=cy,
            )
        )

    return regions, labels


def analyze_components(
    binary_mask: np.ndarray,
    min_component_area: int = 20,
    connectivity: int = 8,
) -> Tuple[ComponentStatistics, np.ndarray]:
    """
    Extracts, filters, and computes aggregate statistics for connected solar regions.

    Args:
        binary_mask: 2D uint8 numpy array with values strictly {0, 1}.
        min_component_area: Minimum pixel area threshold for retaining components.
        connectivity: Neighborhood connectivity (4 or 8). Default is 8.

    Returns:
        Tuple of (ComponentStatistics, label_map 2D array).
    """
    raw_regions, label_map = extract_connected_components(
        binary_mask,
        connectivity=connectivity,
    )
    raw_count = len(raw_regions)
    raw_solar_pixels = int(sum(r.area_pixels for r in raw_regions))

    # Apply area filtering
    filtered_regions = [
        r for r in raw_regions if r.area_pixels >= min_component_area
    ]
    filtered_count = len(filtered_regions)
    filtered_solar_pixels = int(sum(r.area_pixels for r in filtered_regions))

    # Handle edge case where no components exist or all are filtered out
    if filtered_count > 0:
        areas = [r.area_pixels for r in filtered_regions]
        largest_area = max(areas)
        smallest_area = min(areas)
        mean_area = float(np.mean(areas))
        median_area = float(np.median(areas))
    else:
        largest_area = None
        smallest_area = None
        mean_area = None
        median_area = None

    if raw_solar_pixels > 0:
        retained_percentage = (filtered_solar_pixels / raw_solar_pixels) * 100.0
    else:
        retained_percentage = 100.0

    total_image_pixels = int(binary_mask.size)
    region_density = (
        float(filtered_count / total_image_pixels) if total_image_pixels > 0 else 0.0
    )

    stats = ComponentStatistics(
        min_component_area_pixels=int(min_component_area),
        raw_count=raw_count,
        filtered_count=filtered_count,
        largest_area_pixels=largest_area,
        smallest_area_pixels=smallest_area,
        mean_area_pixels=mean_area,
        median_area_pixels=median_area,
        raw_solar_pixels=raw_solar_pixels,
        filtered_solar_pixels=filtered_solar_pixels,
        retained_percentage=retained_percentage,
        region_density_per_pixel=region_density,
        regions=filtered_regions,
    )

    return stats, label_map


def compute_pixel_area_metrics(
    binary_mask: np.ndarray,
    min_component_area: int = 20,
    connectivity: int = 8,
) -> Dict[str, Any]:
    """
    Computes standardized pixel-based solar area measurements directly from a binary mask.

    Adheres strictly to Phase 1 requirements:
    - Solar pixels are counted directly from the final binary mask {0, 1}.
    - Total pixels are computed from the mask spatial dimensions (width * height).
    - Solar coverage percentage is calculated as: (solar_area_pixels / total_image_pixels) * 100.
    - Connected solar regions are extracted via 8-connectivity analysis with noise filtering.
    - Empty masks (no solar pixels or all components filtered out) are handled safely with nulls.
    - Physical area safety: strictly reports physical_area_m2=None, physical_area_hectares=None,
      and physical_area_status="insufficient_data" unless verified physical scale is calibrated.

    Args:
        binary_mask: 2D uint8 numpy array with values in {0, 1}.
        min_component_area: Minimum pixel area threshold for connected solar regions.
        connectivity: Neighborhood connectivity (default 8).

    Returns:
        Standardized dictionary containing all pixel area metrics and physical area status.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got array with shape {binary_mask.shape}")

    unique_vals = set(np.unique(binary_mask))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Mask must contain strictly values {{0, 1}}, got {unique_vals}")

    h, w = binary_mask.shape
    total_image_pixels = int(w * h)
    solar_area_pixels = int(np.count_nonzero(binary_mask == 1))

    solar_coverage_percent = (
        (solar_area_pixels / total_image_pixels) * 100.0
        if total_image_pixels > 0
        else 0.0
    )

    # Use existing connected-component analysis for region metrics
    stats, _ = analyze_components(
        binary_mask,
        min_component_area=min_component_area,
        connectivity=connectivity,
    )

    detected_region_count = int(stats.filtered_count)
    if detected_region_count > 0:
        largest_region_area_pixels = int(stats.largest_area_pixels)
        smallest_region_area_pixels = int(stats.smallest_area_pixels)
        mean_region_area_pixels = round(float(stats.mean_area_pixels), 4)
    else:
        largest_region_area_pixels = None
        smallest_region_area_pixels = None
        mean_region_area_pixels = None

    return {
        "image_width": int(w),
        "image_height": int(h),
        "total_image_pixels": total_image_pixels,
        "solar_area_pixels": solar_area_pixels,
        "solar_coverage_percent": round(solar_coverage_percent, 4),
        "detected_region_count": detected_region_count,
        "largest_region_area_pixels": largest_region_area_pixels,
        "smallest_region_area_pixels": smallest_region_area_pixels,
        "mean_region_area_pixels": mean_region_area_pixels,
        "physical_area_m2": None,
        "physical_area_hectares": None,
        "physical_area_status": "insufficient_data",
    }
