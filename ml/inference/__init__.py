"""
SolarMap-India — Half U-Net Production Inference Package.
"""

from ml.inference.model import HalfUNet, get_device, load_halfunet_model
from ml.inference.postprocessing import (
    create_overlay,
    mask_to_visual_image,
    process_logits,
    resize_mask_to_original,
)
from ml.inference.predict import InferenceResult, predict_image
from ml.inference.preprocessing import (
    MODEL_INPUT_SIZE,
    load_and_validate_image,
    preprocess_image,
)

__all__ = [
    "HalfUNet",
    "get_device",
    "load_halfunet_model",
    "load_and_validate_image",
    "preprocess_image",
    "process_logits",
    "resize_mask_to_original",
    "mask_to_visual_image",
    "create_overlay",
    "predict_image",
    "InferenceResult",
    "MODEL_INPUT_SIZE",
]
