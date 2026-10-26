"""
SolarMap-India — Half U-Net Production Inference Package.
"""

from ml.inference.model import HalfUNet, get_device, load_halfunet_model
from ml.inference.postprocessing import (
    create_overlay,
    mask_to_visual_image,
    process_logits,
    resize_mask_to_original,
    unpad_and_restore_mask,
)
from ml.inference.predict import InferenceResult, predict_image
from ml.inference.preprocessing import (
    MODEL_INPUT_SIZE,
    LetterboxParams,
    compute_letterbox_params,
    compute_tile_coordinates,
    generate_hann_window,
    letterbox_image,
    load_and_validate_image,
    preprocess_image,
    preprocess_image_letterbox,
)
from ml.inference.tiling import TilingConfig, predict_tiled

__all__ = [
    "HalfUNet",
    "get_device",
    "load_halfunet_model",
    "load_and_validate_image",
    "preprocess_image",
    "preprocess_image_letterbox",
    "letterbox_image",
    "compute_letterbox_params",
    "LetterboxParams",
    "compute_tile_coordinates",
    "generate_hann_window",
    "process_logits",
    "resize_mask_to_original",
    "unpad_and_restore_mask",
    "mask_to_visual_image",
    "create_overlay",
    "predict_image",
    "InferenceResult",
    "MODEL_INPUT_SIZE",
    "TilingConfig",
    "predict_tiled",
]
