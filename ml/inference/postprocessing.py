"""
SolarMap-India — Postprocessing & Mask Generation Pipeline.

Handles:
- Softmax probability calculation
- Argmax / threshold-based binary segmentation mask generation
- Nearest-neighbor resizing back to original image dimensions
- Visual overlay creation blending solar masks with original imagery
"""

from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import torch


def process_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes class probabilities and extracts the binary solar mask at model resolution (256x256).

    Args:
        logits: Output tensor from Half U-Net, shape [1, 2, 256, 256].
        threshold: Foreground probability threshold for class 1 (solar panel). Default is 0.5.

    Returns:
        Tuple of (binary_mask_256 [256, 256] uint8 with values {0, 1},
                  solar_probability_map [256, 256] float32 with values in [0.0, 1.0]).

    Raises:
        RuntimeError: If output logits contain NaN or Infinite values.
    """
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("Model inference produced NaN or Infinite logit values.")

    # Compute softmax along class dimension (dim=1)
    probs = torch.softmax(logits, dim=1)  # [1, 2, 256, 256]

    # Foreground class 1 = solar panel
    solar_prob = probs[0, 1].detach().cpu().numpy().astype(np.float32)

    # Thresholding / argmax: prob >= threshold assigns class 1
    binary_mask = (solar_prob >= threshold).astype(np.uint8)

    return binary_mask, solar_prob


def resize_mask_to_original(
    mask: np.ndarray,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resizes binary mask back to original image resolution using nearest-neighbor interpolation.

    Args:
        mask: 2D numpy array [H, W] of uint8 values {0, 1}.
        original_size: Tuple of (original_width, original_height).

    Returns:
        2D numpy array [orig_h, orig_w] of uint8 strictly containing values {0, 1}.
    """
    orig_w, orig_h = original_size
    if mask.shape[1] == orig_w and mask.shape[0] == orig_h:
        return mask.copy()

    # Nearest-neighbor interpolation preserves discrete class IDs {0, 1}
    resized_mask = cv2.resize(
        mask,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Sanity verify discrete binary set
    unique_vals = set(np.unique(resized_mask))
    if not unique_vals.issubset({0, 1}):
        # In case interpolation introduced artifacts, re-binarize
        resized_mask = (resized_mask > 0).astype(np.uint8)

    return resized_mask


def mask_to_visual_image(binary_mask: np.ndarray) -> Image.Image:
    """
    Converts logical binary mask {0, 1} into a viewable grayscale image {0, 255}.

    Args:
        binary_mask: 2D uint8 numpy array with values {0, 1}.

    Returns:
        PIL Image in 'L' mode with pixel values 0 (background) and 255 (solar).
    """
    visual_array = (binary_mask * 255).astype(np.uint8)
    return Image.fromarray(visual_array, mode="L")


def create_overlay(
    original_image: Image.Image,
    binary_mask: np.ndarray,
    alpha: float = 0.45,
    color_rgb: Tuple[int, int, int] = (255, 69, 0),  # Solar red-orange
) -> Image.Image:
    """
    Blends the detected solar mask onto the original RGB image.

    Args:
        original_image: Original PIL Image (RGB mode).
        binary_mask: 2D uint8 numpy array with values {0, 1} matching image dimensions.
        alpha: Opacity factor for detected solar panel pixels (0.0 - 1.0).
        color_rgb: RGB color tuple for highlighting detected solar pixels.

    Returns:
        PIL Image of original dimensions with blended solar overlay.
    """
    orig_rgb = np.array(original_image.convert("RGB"), dtype=np.uint8)
    orig_h, orig_w, _ = orig_rgb.shape

    if binary_mask.shape != (orig_h, orig_w):
        binary_mask = resize_mask_to_original(binary_mask, (orig_w, orig_h))

    # Create colored mask layer
    colored_layer = np.zeros_like(orig_rgb)
    solar_indices = binary_mask == 1
    colored_layer[solar_indices] = color_rgb

    # Blend original and colored layer where mask is active
    blended = orig_rgb.copy()
    if np.any(solar_indices):
        blended[solar_indices] = (
            (1.0 - alpha) * orig_rgb[solar_indices] + alpha * np.array(color_rgb)
        ).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")
