"""
SolarMap-India — Postprocessing & Mask Generation Pipeline.

Handles:
- Softmax probability calculation
- Argmax / threshold-based binary segmentation mask generation
- Nearest-neighbor resizing back to original image dimensions
- Unpadding and reconstruction for aspect-ratio preserved letterboxed masks
- Visual overlay creation blending solar masks with original imagery
"""

from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from ml.inference.preprocessing import LetterboxParams


def process_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes class probabilities and extracts the binary solar mask at model resolution.

    Args:
        logits: Output tensor from Half U-Net, shape [1, 2, H, W] or [B, 2, H, W].
        threshold: Foreground probability threshold for class 1 (solar panel). Default is 0.5.

    Returns:
        Tuple of (binary_mask [H, W] uint8 with values {0, 1},
                  solar_probability_map [H, W] float32 with values in [0.0, 1.0]).

    Raises:
        RuntimeError: If output logits contain NaN or Infinite values.
    """
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("Model inference produced NaN or Infinite logit values.")

    # Compute softmax along class dimension (dim=1)
    probs = torch.softmax(logits, dim=1)  # [B, 2, H, W]

    # Foreground class 1 = solar panel (first sample if batched)
    solar_prob = probs[0, 1].detach().cpu().numpy().astype(np.float32)

    # Thresholding / argmax: prob >= threshold assigns class 1
    binary_mask = (solar_prob >= threshold).astype(np.uint8)

    return binary_mask, solar_prob


def unpad_and_restore_mask(
    letterbox_mask: np.ndarray,
    params: LetterboxParams,
) -> np.ndarray:
    """
    Crops away padding added during letterboxing and scales the unpadded mask
    back to original image resolution using nearest-neighbor interpolation.

    Args:
        letterbox_mask: 2D uint8 numpy array [target_h, target_w] containing {0, 1}.
        params: LetterboxParams recorded during preprocessing.

    Returns:
        2D uint8 numpy array [orig_h, orig_w] strictly containing values {0, 1}.
    """
    new_w, new_h = params.scaled_size
    pad_l = params.pad_left
    pad_t = params.pad_top
    orig_w, orig_h = params.original_size

    # Crop out scaled active image region, discarding letterbox borders
    cropped_mask = letterbox_mask[pad_t : pad_t + new_h, pad_l : pad_l + new_w]

    # Resize cropped mask back to original image dimensions
    if cropped_mask.shape[1] == orig_w and cropped_mask.shape[0] == orig_h:
        return cropped_mask.copy()

    restored_mask = cv2.resize(
        cropped_mask,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Sanity verify discrete binary set
    unique_vals = set(np.unique(restored_mask))
    if not unique_vals.issubset({0, 1}):
        restored_mask = (restored_mask > 0).astype(np.uint8)

    return restored_mask


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
        resized_mask = (resized_mask > 0).astype(np.uint8)

    return resized_mask


def unpad_and_restore_probability_map(
    letterbox_prob: np.ndarray,
    params: LetterboxParams,
) -> np.ndarray:
    """
    Crops away padding added during letterboxing and scales the continuous probability map
    back to original image resolution using bilinear interpolation.

    Args:
        letterbox_prob: 2D float32 numpy array [target_h, target_w] with values in [0.0, 1.0].
        params: LetterboxParams recorded during preprocessing.

    Returns:
        2D float32 numpy array [orig_h, orig_w] with values in [0.0, 1.0].
    """
    new_w, new_h = params.scaled_size
    pad_l = params.pad_left
    pad_t = params.pad_top
    orig_w, orig_h = params.original_size

    # Crop out scaled active image region, discarding letterbox borders
    cropped_prob = letterbox_prob[pad_t : pad_t + new_h, pad_l : pad_l + new_w]

    if cropped_prob.shape[1] == orig_w and cropped_prob.shape[0] == orig_h:
        return np.clip(cropped_prob.copy(), 0.0, 1.0).astype(np.float32)

    # Bilinear interpolation preserves continuous probability transitions
    restored_prob = cv2.resize(
        cropped_prob,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(restored_prob, 0.0, 1.0).astype(np.float32)


def resize_probability_map_to_original(
    prob_map: np.ndarray,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resizes continuous probability map back to original image resolution using bilinear interpolation.

    Args:
        prob_map: 2D float32 numpy array [H, W] with values in [0.0, 1.0].
        original_size: Tuple of (original_width, original_height).

    Returns:
        2D float32 numpy array [orig_h, orig_w] with values in [0.0, 1.0].
    """
    orig_w, orig_h = original_size
    if prob_map.shape[1] == orig_w and prob_map.shape[0] == orig_h:
        return np.clip(prob_map.copy(), 0.0, 1.0).astype(np.float32)

    resized_prob = cv2.resize(
        prob_map,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized_prob, 0.0, 1.0).astype(np.float32)


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
