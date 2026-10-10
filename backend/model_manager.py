"""
SolarMap-India — Backend Model Manager.

Maintains an in-memory singleton instance of the Half U-Net model
to avoid redundant checkpoint reloading per HTTP request.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from backend.config import DEFAULT_CHECKPOINT
from ml.inference.model import HalfUNet, get_device as resolve_device, load_halfunet_model

logger = logging.getLogger("solarmap.model_manager")


class ModelManager:
    """Singleton manager for the Half U-Net segmentation model."""

    _instance: Optional["ModelManager"] = None

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device: torch.device = resolve_device()
        self.model: Optional[HalfUNet] = None
        self._load_error: Optional[str] = None

    @classmethod
    def get_instance(cls, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path)
        return cls._instance

    def load_model(self) -> None:
        """Loads the model checkpoint into memory if available."""
        if not self.checkpoint_path.is_file():
            self._load_error = f"Checkpoint file not found: {self.checkpoint_path.resolve()}"
            logger.error(self._load_error)
            self.model = None
            return

        try:
            logger.info("Loading Half U-Net model from %s on device %s", self.checkpoint_path, self.device)
            self.model = load_halfunet_model(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            self._load_error = None
            logger.info("Half U-Net model loaded successfully and set to eval() mode.")
        except Exception as exc:
            self._load_error = f"Failed to load Half U-Net checkpoint: {exc}"
            logger.exception(self._load_error)
            self.model = None

    @property
    def is_ready(self) -> bool:
        """Returns True if checkpoint exists and model is loaded in memory."""
        return self.model is not None and self.checkpoint_path.is_file()

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of model and checkpoint."""
        return {
            "model_name": "HalfUNet",
            "is_ready": self.is_ready,
            "checkpoint_available": self.checkpoint_path.is_file(),
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "device": str(self.device),
            "error": self._load_error,
        }


# Global singleton instance
model_manager = ModelManager.get_instance()
