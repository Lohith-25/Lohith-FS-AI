"""
SolarMap-India — Pydantic API Schemas.

Defines request/response contracts adhering strictly to Phase 1 & Phase 2 standards.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for GET /health endpoint."""
    status: str = Field(..., description="API operational status ('healthy' or 'unhealthy')")
    model: str = Field(default="HalfUNet", description="Model architecture name")
    checkpoint_available: bool = Field(..., description="Whether model weights exist on disk")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    device: str = Field(..., description="Computation device (e.g. 'cuda' or 'cpu')")
    error: Optional[str] = Field(default=None, description="Diagnostic error details if unhealthy")


class PredictResponse(BaseModel):
    """Schema for POST /predict inference response."""
    success: bool = Field(default=True, description="Whether inference completed successfully")
    model: str = Field(default="HalfUNet", description="Model architecture used for inference")
    prediction_id: str = Field(..., description="Unique UUID identifier for this inference run")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    total_image_pixels: int = Field(..., description="Total pixel count (width * height)")
    solar_area_pixels: int = Field(..., description="Count of segmented solar pixels")
    solar_coverage_percent: float = Field(..., description="Percentage of image covered by solar panels")
    detected_region_count: int = Field(..., description="Number of detected connected solar regions")
    largest_region_area_pixels: Optional[int] = Field(None, description="Largest region area in pixels")
    smallest_region_area_pixels: Optional[int] = Field(None, description="Smallest region area in pixels")
    mean_region_area_pixels: Optional[float] = Field(None, description="Mean region area in pixels")
    physical_area_m2: Optional[float] = Field(None, description="Physical area in m^2 (strictly null without verified GSD)")
    physical_area_hectares: Optional[float] = Field(None, description="Physical area in hectares (strictly null without verified GSD)")
    physical_area_status: str = Field(default="insufficient_data", description="Status of physical resolution calibration")
    mask_url: str = Field(..., description="Relative API URL to retrieve binary segmentation mask")
    overlay_url: str = Field(..., description="Relative API URL to retrieve visual overlay")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message summary")
    detail: Optional[Any] = Field(default=None, description="Detailed error information")
