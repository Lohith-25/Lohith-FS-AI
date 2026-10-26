"""
SolarMap-India — Image Preprocessing Pipeline for Variable Resolutions.

Supports:
1. Legacy direct resize (for backward compatibility / 256x256 images).
2. Aspect-ratio-preserving letterboxing with padding parameters for exact mask restoration.
3. Modular tiling preparation with coordinate mapping and Hann window generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

MODEL_INPUT_SIZE: Tuple[int, int] = (256, 256)

# Standard evaluation transform matching Half U-Net training pipeline
# Converts PIL [0, 255] RGB image to [0.0, 1.0] FloatTensor of shape [C, H, W]
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
])

TO_TENSOR_TRANSFORM = transforms.ToTensor()


@dataclass(frozen=True)
class LetterboxParams:
    """Stores exact scaling and padding metadata for aspect-ratio-preserving inference."""
    original_size: Tuple[int, int]  # (orig_w, orig_h)
    scaled_size: Tuple[int, int]    # (new_w, new_h)
    target_size: Tuple[int, int]    # (target_w, target_h)
    scale: float                    # Uniform scale factor min(tw/w, th/h)
    pad_left: int                   # Padding on left
    pad_top: int                    # Padding on top
    pad_right: int                  # Padding on right
    pad_bottom: int                 # Padding on bottom


def load_and_validate_image(
    image_path: Union[str, Path]
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Loads an image from disk, validates integrity, and converts to RGB.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (PIL Image in RGB mode, (original_width, original_height)).

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file cannot be decoded as a valid image or has zero dimension.
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path.resolve()}")

    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify image integrity
    except Exception as exc:
        raise ValueError(
            f"Corrupted or invalid image file: {img_path.name} ({exc})"
        ) from exc

    try:
        # Reopen for actual loading after verify()
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Failed to read and convert image to RGB: {img_path.name} ({exc})"
        ) from exc

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions ({width}x{height}) for image: {img_path.name}"
        )

    return image, (width, height)


def compute_letterbox_params(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> LetterboxParams:
    """
    Calculates exact scale factor and padding offsets to preserve aspect ratio.

    Args:
        original_size: (width, height) of the source image.
        target_size: (width, height) of the model input canvas (default (256, 256)).

    Returns:
        LetterboxParams dataclass containing scale and border padding values.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid original image dimensions: ({orig_w}, {orig_h}). Must be > 0.")
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target dimensions: ({target_w}, {target_h}). Must be > 0.")

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    return LetterboxParams(
        original_size=(orig_w, orig_h),
        scaled_size=(new_w, new_h),
        target_size=(target_w, target_h),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def letterbox_image(
    image: Image.Image,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, LetterboxParams]:
    """
    Resizes an image preserving aspect ratio and pads borders to target_size.

    Args:
        image: Source PIL Image in RGB format.
        target_size: Desired output canvas dimensions (default (256, 256)).
        pad_color: RGB color tuple for padding background (default black (0, 0, 0)).

    Returns:
        Tuple of (padded PIL Image of target_size, LetterboxParams).
    """
    params = compute_letterbox_params(image.size, target_size=target_size)
    new_w, new_h = params.scaled_size

    # Resize using bilinear interpolation matching model expectations
    resized_img = image.resize((new_w, new_h), resample=Image.BILINEAR)

    # Place resized image onto centered padded canvas
    padded_canvas = Image.new("RGB", target_size, color=pad_color)
    padded_canvas.paste(resized_img, (params.pad_left, params.pad_top))

    return padded_canvas, params


def preprocess_image(
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """
    Legacy direct preprocessing: stretches image to [1, 3, 256, 256].
    Retained for backward compatibility.
    """
    tensor = INFERENCE_TRANSFORM(image)  # [3, 256, 256]
    tensor = tensor.unsqueeze(0)          # [1, 3, 256, 256]
    return tensor.to(device)


def preprocess_image_letterbox(
    image: Image.Image,
    device: torch.device,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> Tuple[torch.Tensor, LetterboxParams]:
    """
    Preprocesses a PIL RGB image with aspect-ratio preservation and padding.

    Args:
        image: PIL Image in RGB format.
        device: torch.device to place output tensor on.
        target_size: Target tensor spatial dimensions (default (256, 256)).

    Returns:
        Tuple of (Tensor of shape [1, 3, target_h, target_w] on device, LetterboxParams).
    """
    padded_image, params = letterbox_image(image, target_size=target_size)
    tensor = TO_TENSOR_TRANSFORM(padded_image)  # [3, target_h, target_w]
    tensor = tensor.unsqueeze(0)                 # [1, 3, target_h, target_w]
    return tensor.to(device), params


def compute_tile_coordinates(
    canvas_dim: int,
    tile_dim: int = 256,
    overlap_ratio: float = 0.25,
) -> List[int]:
    """
    Generates 1D starting coordinates covering an entire dimension with overlap.

    Ensures:
    - 0 is always included.
    - All coordinates satisfy 0 <= c <= canvas_dim - tile_dim.
    - Final coordinate is canvas_dim - tile_dim, ensuring complete edge coverage.

    Args:
        canvas_dim: Dimension of canvas (must be >= tile_dim).
        tile_dim: Size of tile along this dimension (default: 256).
        overlap_ratio: Minimum fractional overlap between consecutive tiles (default: 0.25).

    Returns:
        Sorted list of integer start offsets.
    """
    if canvas_dim <= tile_dim:
        return [0]

    step = max(1, int(round(tile_dim * (1.0 - overlap_ratio))))
    coords = list(range(0, canvas_dim - tile_dim, step))
    if coords[-1] + tile_dim < canvas_dim:
        coords.append(canvas_dim - tile_dim)

    return sorted(list(set(coords)))


def generate_hann_window(
    tile_size: Tuple[int, int] = MODEL_INPUT_SIZE,
    min_weight: float = 1e-3,
) -> np.ndarray:
    """
    Generates a 2D Hann blending window with non-zero minimum weight at edges.

    Args:
        tile_size: (tile_w, tile_h) dimensions (default: (256, 256)).
        min_weight: Floor value to ensure edge and corner pixels have non-zero weight.

    Returns:
        2D float32 numpy array of shape [tile_h, tile_w] with values in [min_weight, 1.0].
    """
    tile_w, tile_h = tile_size
    wh = np.hanning(tile_h + 2)[1:-1]
    ww = np.hanning(tile_w + 2)[1:-1]
    win2d = np.outer(wh, ww).astype(np.float32)
    return np.clip(win2d, min_weight, 1.0)
