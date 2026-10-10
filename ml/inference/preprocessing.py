"""
SolarMap-India — Image Preprocessing Pipeline.

Reproduces the exact training preprocessing:
PIL Image -> RGB -> Resize(256, 256) -> ToTensor() -> Add batch dimension.
"""

from pathlib import Path
from typing import Tuple, Union

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


def preprocess_image(
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocesses a PIL RGB image for Half U-Net inference.

    Args:
        image: PIL Image in RGB format.
        device: torch.device to place the output tensor on.

    Returns:
        Tensor of shape [1, 3, 256, 256] with values in [0.0, 1.0].
    """
    tensor = INFERENCE_TRANSFORM(image)  # [3, 256, 256]
    tensor = tensor.unsqueeze(0)          # [1, 3, 256, 256]
    return tensor.to(device)
