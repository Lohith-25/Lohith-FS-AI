"""
SolarMap-India — Component & Bounding Box Visualizations.

Generates:
1. Component bounding box overlay onto the original image.
2. Color-coded multi-region segmentation visualization.
Preserves original image spatial dimensions without distortion.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ml.analytics.components import SolarRegion


def draw_bounding_boxes(
    image: Union[Image.Image, np.ndarray],
    regions: List[SolarRegion],
    box_color: Tuple[int, int, int] = (0, 255, 127),    # Spring green
    text_color: Tuple[int, int, int] = (255, 255, 255),  # White
    box_thickness: int = 2,
    draw_labels: bool = True,
) -> Image.Image:
    """
    Draws rectangular bounding boxes and component IDs onto an image.

    Args:
        image: Original image as PIL Image or numpy array (RGB).
        regions: List of SolarRegion instances to annotate.
        box_color: Bounding box outline color (R, G, B).
        text_color: Label text color (R, G, B).
        box_thickness: Outline line thickness in pixels.
        draw_labels: Whether to render component ID badges.

    Returns:
        PIL Image with bounding boxes drawn, preserving original dimensions.
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_rgb = image.copy()

    # OpenCV operates in BGR for drawing operations
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    for region in regions:
        x, y, w, h = region.x, region.y, region.width, region.height
        b_color_bgr = (box_color[2], box_color[1], box_color[0])
        t_color_bgr = (text_color[2], text_color[1], text_color[0])

        # Draw bounding rectangle
        cv2.rectangle(
            img_bgr,
            (x, y),
            (x + w, y + h),
            b_color_bgr,
            thickness=box_thickness,
        )

        if draw_labels:
            label_text = f"#{region.id}"
            font_scale = 0.4
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )

            # Draw background tag above or inside the bounding box
            tag_y1 = max(0, y - text_h - 4)
            tag_y2 = tag_y1 + text_h + 4
            tag_x2 = min(img_bgr.shape[1], x + text_w + 4)

            cv2.rectangle(
                img_bgr,
                (x, tag_y1),
                (tag_x2, tag_y2),
                b_color_bgr,
                thickness=-1,
            )
            cv2.putText(
                img_bgr,
                label_text,
                (x + 2, tag_y2 - 2),
                font,
                font_scale,
                (0, 0, 0),  # Black text for high contrast on green box
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb, mode="RGB")


def create_colored_components_map(
    label_map: np.ndarray,
    retained_ids: Optional[set] = None,
) -> Image.Image:
    """
    Generates a color-coded map where each retained connected component has a unique hue.

    Args:
        label_map: 2D integer numpy array produced by connectedComponents.
        retained_ids: Set of component IDs to include (filters out noise if provided).

    Returns:
        PIL Image of original dimensions with distinct component colors.
    """
    h, w = label_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    max_label = int(label_map.max())
    if max_label == 0:
        return Image.fromarray(colored, mode="RGB")

    # Generate deterministic distinguishable palette using HSV colormap
    np.random.seed(42)
    colors = np.random.randint(40, 255, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    for label_id in range(1, max_label + 1):
        if retained_ids is not None and label_id not in retained_ids:
            continue
        colored[label_map == label_id] = colors[label_id]

    return Image.fromarray(colored, mode="RGB")
