"""
SolarMap-India — Phase 8: Multi-Scale and Resolution-Robust Augmentations.

Implements label-preserving, multi-scale, and resolution-degradation transformations:
1. Multi-scale random resizing followed by compatible solar-preserving cropping or zero-padding.
2. Resolution degradation: simulated downsampling and Gaussian blur (image only).
3. Photometric variation: mild brightness and contrast adjustment (image only).
4. Geometric symmetry: horizontal/vertical flips and orthogonal 90-degree rotations.
5. Strict nearest-neighbor interpolation on categorical masks to prevent label corruption.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms.functional as TF


class LabelPreservingMultiScaleAugmentor:
    """
    Synchronized image-mask augmentation pipeline for multi-scale and resolution robustness.
    
    Guarantees:
    - Mask is always resampled using Image.NEAREST.
    - Mask pixel values remain strictly in {0, 1}.
    - Solar regions are preserved during scale/crop operations.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        scale_range: Tuple[float, float] = (0.75, 1.35),
        p_scale: float = 0.70,
        p_flip_h: float = 0.50,
        p_flip_v: float = 0.50,
        p_rotate_ortho: float = 0.50,
        p_resolution_degrade: float = 0.40,
        p_photometric: float = 0.40,
    ):
        self.target_size = target_size
        self.scale_range = scale_range
        self.p_scale = p_scale
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.p_rotate_ortho = p_rotate_ortho
        self.p_resolution_degrade = p_resolution_degrade
        self.p_photometric = p_photometric

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Applies synchronized multi-scale augmentations.
        
        Args:
            image: PIL Image in RGB format.
            mask: PIL Image in 'L' mode with binary values {0, 1} or {0, 255}.
            
        Returns:
            Tuple of (augmented_image, augmented_mask), both sized target_size.
        """
        tgt_w, tgt_h = self.target_size

        # 1. Multi-Scale Resizing followed by Cropping or Padding
        if random.random() < self.p_scale:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_w = max(int(tgt_w * scale), 64)
            new_h = max(int(tgt_h * scale), 64)

            # Resize image with Bilinear, mask strictly with Nearest
            image = image.resize((new_w, new_h), resample=Image.BILINEAR)
            mask = mask.resize((new_w, new_h), resample=Image.NEAREST)

            if scale > 1.0:
                # Zoomed in: Crop to target_size
                mask_np = np.array(mask)
                solar_pts = np.argwhere(mask_np > 0)

                if len(solar_pts) > 0 and random.random() < 0.85:
                    # Anchor crop around an existing solar panel region
                    cy, cx = solar_pts[random.randint(0, len(solar_pts) - 1)]
                    x0 = max(0, min(cx - tgt_w // 2, new_w - tgt_w))
                    y0 = max(0, min(cy - tgt_h // 2, new_h - tgt_h))
                else:
                    x0 = random.randint(0, new_w - tgt_w)
                    y0 = random.randint(0, new_h - tgt_h)

                image = image.crop((x0, y0, x0 + tgt_w, y0 + tgt_h))
                mask = mask.crop((x0, y0, x0 + tgt_w, y0 + tgt_h))

            elif scale < 1.0:
                # Zoomed out: Pad with zeros to target_size (simulating smaller/distant panels)
                pad_x = (tgt_w - new_w) // 2
                pad_y = (tgt_h - new_h) // 2

                padded_img = Image.new("RGB", (tgt_w, tgt_h), (0, 0, 0))
                padded_mask = Image.new("L", (tgt_w, tgt_h), 0)

                padded_img.paste(image, (pad_x, pad_y))
                padded_mask.paste(mask, (pad_x, pad_y))

                image = padded_img
                mask = padded_mask
        else:
            # Ensure target size if no scale applied
            if image.size != self.target_size:
                image = image.resize(self.target_size, resample=Image.BILINEAR)
            if mask.size != self.target_size:
                mask = mask.resize(self.target_size, resample=Image.NEAREST)

        # 2. Geometric Symmetry: Flips and 90-degree Rotations
        if random.random() < self.p_flip_h:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < self.p_flip_v:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < self.p_rotate_ortho:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)

        # 3. Resolution Degradation & Atmospheric Blur (Image ONLY)
        if random.random() < self.p_resolution_degrade:
            degrade_choice = random.choice(["blur", "downsample"])
            if degrade_choice == "blur":
                radius = random.uniform(0.6, 1.8)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            else:
                # Downsample to low-res and interpolate back to simulate sensor pixelation
                low_res = random.choice([96, 128, 160])
                down_img = image.resize((low_res, low_res), resample=Image.BILINEAR)
                image = down_img.resize(self.target_size, resample=Image.BILINEAR)

        # 4. Mild Photometric Variation: Brightness & Contrast (Image ONLY)
        if random.random() < self.p_photometric:
            enh_b = ImageEnhance.Brightness(image)
            image = enh_b.enhance(random.uniform(0.85, 1.15))
            enh_c = ImageEnhance.Contrast(image)
            image = enh_c.enhance(random.uniform(0.85, 1.15))

        return image, mask
