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

# Environment configuration
import os
DEFAULT_CHECKPOINT = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth")))
FRONTEND_DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", str(PROJECT_ROOT / "frontend" / "dist")))

_cors_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS: list[str] = ["*"] if _cors_env.strip() == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]

# Database Configuration (Phase 8: MongoDB Integration)
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", os.getenv("DATABASE_NAME", "solarmap_india"))
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "predictions")
MONGODB_SERVER_TIMEOUT_MS: int = int(os.getenv("MONGODB_SERVER_TIMEOUT_MS", "3000"))

