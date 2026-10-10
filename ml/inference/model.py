"""
SolarMap-India — Half U-Net Architecture and Model Loader.

This module defines the exact Half U-Net neural network architecture
and provides safe checkpoint loading utilities for production inference.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Expected trained parameter count for validation
EXPECTED_PARAM_COUNT = 1_928_450


class ConvBlock(nn.Module):
    """
    Standard double convolution block with Batch Normalization and ReLU.
    Matches the exact training implementation in model.ipynb / Half U-Net.ipynb.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HalfUNet(nn.Module):
    """
    Half U-Net architecture for semantic segmentation.

    Encoder: 3 levels (3 -> 32 -> 64 -> 128)
    Bottleneck: (128 -> 256)
    Decoder: 3 levels with skip connections (256 -> 128 -> 64 -> 32)
    Classifier: 1x1 Conv (32 -> 2 classes)
    Interpolation: Bilinear back to input spatial size
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 4, c * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        # Segmentation head
        self.classifier = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Classification logits
        out = self.classifier(d1)

        # Bilinear interpolation back to input size
        out = F.interpolate(
            out,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return out


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Selects target execution device. Supports CUDA with automatic fallback to CPU.
    """
    if preferred_device:
        pref = preferred_device.lower().strip()
        if pref.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(pref)
            else:
                return torch.device("cpu")
        elif pref == "cpu":
            return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_halfunet_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> HalfUNet:
    """
    Instantiates HalfUNet and loads the trained checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: torch.device to place the model on (defaults to get_device()).

    Returns:
        HalfUNet model in eval mode placed on target device.

    Raises:
        FileNotFoundError: If checkpoint path does not exist.
        RuntimeError: If checkpoint cannot be loaded or state_dict keys mismatch.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path.resolve()}"
        )

    if device is None:
        device = get_device()

    model = HalfUNet(num_classes=2, in_channels=3, base_channels=32)

    total_params = sum(p.numel() for p in model.parameters())
    if total_params != EXPECTED_PARAM_COUNT:
        raise RuntimeError(
            f"Architecture parameter count mismatch: got {total_params}, expected {EXPECTED_PARAM_COUNT}"
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format in {ckpt_path}: expected dictionary containing model_state_dict"
        )

    # Load with strict verification
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load state_dict into HalfUNet model: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model
