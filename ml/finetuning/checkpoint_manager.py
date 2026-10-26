"""
SolarMap-India — Phase 3: Checkpoint Safety & Integrity Manager.

Enforces strict baseline checkpoint protection:
1. Locks and verifies baseline checkpoint: outputs/HalfUNet/models/halfunet_best.pth
2. Blocks any attempt to overwrite, delete, or rename the baseline checkpoint.
3. Automatically computes and verifies SHA-256 hashes before and after all operations.
4. Directs all fine-tuning candidate checkpoints to outputs/experiments/pv01_finetuning/models/.
"""

import hashlib
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
)
DEFAULT_EXPERIMENTAL_DIR = (
    PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "models"
)


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 checksum of a file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class CheckpointSafetyManager:
    """
    Manages saving experimental checkpoints while actively verifying baseline immutability.
    """

    def __init__(
        self,
        baseline_path: Path = BASELINE_CHECKPOINT_PATH,
        experimental_dir: Path = DEFAULT_EXPERIMENTAL_DIR,
    ):
        self.baseline_path = baseline_path.resolve()
        self.experimental_dir = experimental_dir.resolve()
        self.experimental_dir.mkdir(parents=True, exist_ok=True)

        if not self.baseline_path.is_file():
            raise FileNotFoundError(
                f"Baseline checkpoint not found at: {self.baseline_path}"
            )

        # Record initial baseline state
        self.initial_baseline_sha256 = compute_file_sha256(self.baseline_path)
        self.initial_baseline_mtime = self.baseline_path.stat().st_mtime
        self.initial_baseline_size = self.baseline_path.stat().st_size

    def assert_baseline_unmodified(self) -> None:
        """
        Asserts that the baseline checkpoint is byte-for-byte identical to its initial state.
        Raises AssertionError if modified.
        """
        current_mtime = self.baseline_path.stat().st_mtime
        current_size = self.baseline_path.stat().st_size
        current_sha256 = compute_file_sha256(self.baseline_path)

        if current_sha256 != self.initial_baseline_sha256:
            raise AssertionError(
                f"CRITICAL VIOLATION: Baseline checkpoint has been modified!\n"
                f"Expected SHA256: {self.initial_baseline_sha256}\n"
                f"Actual SHA256:   {current_sha256}\n"
                f"Target file:     {self.baseline_path}"
            )

        if current_size != self.initial_baseline_size:
            raise AssertionError(
                f"CRITICAL VIOLATION: Baseline checkpoint size changed from "
                f"{self.initial_baseline_size} to {current_size} bytes!"
            )

    def save_experimental_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_metrics: Dict[str, Any],
        filename: str = "halfunet_pv01_best.pth",
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Saves experimental checkpoint strictly inside the experimental directory.
        Re-verifies baseline checkpoint before and after saving.
        """
        # 1. Assert baseline untouched before operation
        self.assert_baseline_unmodified()

        # 2. Safety check on destination path
        target_path = (self.experimental_dir / filename).resolve()
        if not target_path.is_relative_to(self.experimental_dir) or target_path == self.baseline_path:
            raise PermissionError(
                f"SECURITY VIOLATION: Attempted to save outside experimental directory or over baseline: {target_path}"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Build payload
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "baseline_reference_sha256": self.initial_baseline_sha256,
        }
        if additional_info:
            payload["additional_info"] = additional_info

        # 4. Save checkpoint
        torch.save(payload, target_path)

        # 5. Assert baseline untouched after operation
        self.assert_baseline_unmodified()

        return target_path
