"""
SolarMap-India — Tiled Inference Engine for Large Aerial Imagery.

Implements overlapping window inference with 2D Hann blending:
- Divides high-resolution imagery into configurable model-sized tiles (default 256x256).
- Applies configurable minimum overlap (default 25% = 64px overlap, stride 192px).
- Generates 2D Hann window weighting to blend overlapping probability predictions seamlessly.
- Batches tile inference to respect memory limits and maximize GPU/CPU efficiency.
- Reconstructs exact full-resolution segmentation probability maps and binary masks.
- Handles images smaller than a tile, odd dimensions, and dimensions not divisible by 8.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from ml.inference.model import HalfUNet
from ml.inference.preprocessing import compute_tile_coordinates, generate_hann_window


@dataclass(frozen=True)
class TilingConfig:
    """Configuration parameters for tiled inference."""
    tile_size: Tuple[int, int] = (256, 256)      # (width, height)
    overlap_ratio: float = 0.25                  # Minimum fractional overlap (e.g., 0.25 -> 25%)
    batch_size: int = 4                          # Number of tiles per forward pass
    threshold: float = 0.5                       # Binary classification threshold
    max_image_dimension: int = 16384             # Hard ceiling to prevent out-of-memory on extreme inputs


def predict_tiled(
    image: Image.Image,
    model: HalfUNet,
    device: torch.device,
    config: Optional[TilingConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Executes overlapping tiled inference on an arbitrary-resolution PIL Image.

    Args:
        image: Source PIL Image in RGB format.
        model: Pre-loaded HalfUNet neural network.
        device: Target execution torch.device.
        config: Optional TilingConfig dataclass specifying tile size, overlap, and batch size.

    Returns:
        Tuple of:
            - binary_mask: 2D uint8 numpy array [orig_h, orig_w] in {0, 1}.
            - probability_map: 2D float32 numpy array [orig_h, orig_w] in [0.0, 1.0].
            - total_tiles_processed: Total count of 256x256 tiles evaluated.

    Raises:
        ValueError: If image dimensions exceed safety limits or are non-positive.
    """
    if config is None:
        config = TilingConfig()

    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions for tiled inference: {orig_w}x{orig_h}")

    if orig_w > config.max_image_dimension or orig_h > config.max_image_dimension:
        raise ValueError(
            f"Image dimensions ({orig_w}x{orig_h}) exceed maximum configured limit "
            f"of {config.max_image_dimension}x{config.max_image_dimension}."
        )

    tile_w, tile_h = config.tile_size

    # Handle image dimensions smaller than a tile by padding canvas with reflect/edge values
    pad_right = max(0, tile_w - orig_w)
    pad_bottom = max(0, tile_h - orig_h)
    eff_w = orig_w + pad_right
    eff_h = orig_h + pad_bottom

    if pad_right > 0 or pad_bottom > 0:
        canvas = Image.new("RGB", (eff_w, eff_h), color=(0, 0, 0))
        canvas.paste(image, (0, 0))
    else:
        canvas = image

    # Compute overlapping 1D coordinates
    xs = compute_tile_coordinates(eff_w, tile_dim=tile_w, overlap_ratio=config.overlap_ratio)
    ys = compute_tile_coordinates(eff_h, tile_dim=tile_h, overlap_ratio=config.overlap_ratio)

    tile_coords: List[Tuple[int, int]] = [(x, y) for y in ys for x in xs]
    total_tiles = len(tile_coords)

    # Accumulators for blending: float32 canvas
    prob_accumulator = np.zeros((eff_h, eff_w), dtype=np.float32)
    weight_accumulator = np.zeros((eff_h, eff_w), dtype=np.float32)

    # 2D Hann blending window
    blend_window = generate_hann_window(tile_size=config.tile_size)

    to_tensor = transforms.ToTensor()
    batch_size = max(1, config.batch_size)

    model.eval()

    # Process tiles in mini-batches
    for batch_start in range(0, total_tiles, batch_size):
        batch_slice = tile_coords[batch_start : batch_start + batch_size]
        tile_tensors: List[torch.Tensor] = []

        for (tx, ty) in batch_slice:
            crop = canvas.crop((tx, ty, tx + tile_w, ty + tile_h))
            tensor = to_tensor(crop)  # [3, tile_h, tile_w]
            tile_tensors.append(tensor)

        batch_tensor = torch.stack(tile_tensors, dim=0).to(device)  # [B, 3, tile_h, tile_w]

        with torch.no_grad():
            logits = model(batch_tensor)  # [B, 2, tile_h, tile_w]
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # [B, tile_h, tile_w]

        for i, (tx, ty) in enumerate(batch_slice):
            tile_prob = probs[i]
            prob_accumulator[ty : ty + tile_h, tx : tx + tile_w] += tile_prob * blend_window
            weight_accumulator[ty : ty + tile_h, tx : tx + tile_w] += blend_window

    # Normalize accumulated probabilities by weights
    safe_weights = np.maximum(weight_accumulator, 1e-6)
    full_prob_map = prob_accumulator / safe_weights

    # Crop out padded border back to exact original image dimensions
    restored_prob_map = full_prob_map[:orig_h, :orig_w].astype(np.float32)

    # Generate binary mask based on probability threshold
    binary_mask = (restored_prob_map >= config.threshold).astype(np.uint8)

    return binary_mask, restored_prob_map, total_tiles
