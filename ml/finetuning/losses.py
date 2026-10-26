"""
SolarMap-India — Phase 3: Fine-Tuning Loss Functions.

Implements exact loss functions matching the original Half U-Net training pipeline:
- CrossEntropyLoss (for per-pixel classification)
- DiceLoss (for global region overlap and boundary alignment)
- CompoundSegmentationLoss (CrossEntropy + Dice)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss matching the original Half U-Net training pipeline.
    
    Args:
        smooth: Smoothing Laplace factor to prevent division by zero (default: 1.0).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes Dice Loss between predicted logits and target masks.
        
        Args:
            logits: Predicted tensor of shape (B, num_classes, H, W).
            targets: Ground-truth integer mask tensor of shape (B, H, W) with class indices in [0, num_classes - 1].
            
        Returns:
            Scalar tensor representing 1.0 - mean(Dice).
        """
        if logits.dim() != 4:
            raise ValueError(f"Expected 4D logits tensor (B, C, H, W), got {logits.shape}")
        if targets.dim() != 3:
            raise ValueError(f"Expected 3D targets tensor (B, H, W), got {targets.shape}")

        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CompoundSegmentationLoss(nn.Module):
    """
    Compound loss combining Cross-Entropy Loss and Dice Loss:
        Loss = weight_ce * CrossEntropy + weight_dice * DiceLoss
    """

    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes compound loss.
        
        Args:
            logits: (B, 2, H, W) float tensor
            targets: (B, H, W) int64 tensor with values {0, 1}
            
        Returns:
            total_loss: scalar tensor for backpropagation
            components: dictionary containing detached scalar loss values
        """
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        total_loss = self.weight_ce * loss_ce + self.weight_dice * loss_dice

        components = {
            "loss_total": float(total_loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_dice": float(loss_dice.item()),
        }
        return total_loss, components
