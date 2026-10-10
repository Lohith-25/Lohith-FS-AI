"""
SolarMap-India — Backend Configuration Settings.

Defines environment configuration, paths, model settings, and upload limits.
"""

from pathlib import Path
from typing import Set

# Root directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Storage paths
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "generated"

# File validation constraints
ALLOWED_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Server defaults
API_TITLE = "SolarMap-India Inference API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production REST API for satellite-based rooftop solar panel detection "
    "and pixel-area measurement powered by the Half U-Net segmentation pipeline."
)
