"""
SolarMap-India — Phase 3: Learning Rate Scheduler.

Implements two-epoch linear warmup followed by cosine learning-rate decay.
"""

import math
from typing import List, Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR


def compute_epoch_learning_rate(
    epoch: int,
    warmup_epochs: int = 2,
    total_epochs: int = 15,
    base_lr: float = 5e-5,
    min_lr: float = 1e-6,
) -> float:
    """
    Computes deterministic learning rate for a specific epoch (0-indexed).

    Schedule:
        - Epoch 0 to warmup_epochs - 1: Linear warmup from (base_lr / warmup_epochs) to base_lr
        - Epoch warmup_epochs to total_epochs - 1: Half-period cosine annealing down to min_lr
    """
    if epoch < warmup_epochs:
        # Linear warmup: epoch 0 -> base_lr * (1/W), epoch 1 -> base_lr * (2/W) = base_lr
        factor = float(epoch + 1) / float(warmup_epochs)
        return base_lr * factor
    else:
        decay_epochs = total_epochs - warmup_epochs
        if decay_epochs <= 0:
            return min_lr
        step = epoch - warmup_epochs + 1
        progress = float(step) / float(decay_epochs)
        # Clamp progress to [0, 1]
        progress = min(max(progress, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_factor


def get_learning_rate_schedule_table(
    warmup_epochs: int = 2,
    total_epochs: int = 15,
    base_lr: float = 5e-5,
    min_lr: float = 1e-6,
) -> List[Tuple[int, float, str]]:
    """
    Generates a schedule table for documentation and reporting.
    Returns list of (epoch_1_indexed, lr, phase_description).
    """
    schedule = []
    for e in range(total_epochs):
        lr = compute_epoch_learning_rate(
            epoch=e,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            base_lr=base_lr,
            min_lr=min_lr,
        )
        phase = (
            f"Linear Warmup ({e + 1}/{warmup_epochs})"
            if e < warmup_epochs
            else f"Cosine Decay ({e - warmup_epochs + 1}/{total_epochs - warmup_epochs})"
        )
        schedule.append((e + 1, lr, phase))
    return schedule


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 2,
    total_epochs: int = 15,
    base_lr: float = 5e-5,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """
    Builds a PyTorch LambdaLR scheduler implementing linear warmup + cosine decay.
    """
    def lr_lambda(epoch: int) -> float:
        current_lr = compute_epoch_learning_rate(
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            base_lr=base_lr,
            min_lr=min_lr,
        )
        return current_lr / base_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
