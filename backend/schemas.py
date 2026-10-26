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
    energy_status: str = Field(default="insufficient_data", description="Status of solar energy potential calculation")
    estimated_energy_kwh_year: Optional[float] = Field(default=None, description="Estimated solar energy potential (kWh/year)")
    co2_status: str = Field(default="insufficient_data", description="Status of CO2 avoidance calculation")
    co2_avoided_kg_year: Optional[float] = Field(default=None, description="Estimated CO2 avoided (kg/year)")
    co2_avoided_tonnes_year: Optional[float] = Field(default=None, description="Estimated CO2 avoided (metric tonnes/year)")
    latitude: Optional[float] = Field(default=None, description="Image-level scene centroid latitude (Level B georeferencing)")
    longitude: Optional[float] = Field(default=None, description="Image-level scene centroid longitude (Level B georeferencing)")
    georeferencing_level: str = Field(default="UNREFERENCED", description="Georeferencing classification (e.g. LEVEL_B or UNREFERENCED)")
    location_name: Optional[str] = Field(default=None, description="Approximate geographic region name if cataloged")
    mask_url: str = Field(..., description="Relative API URL to retrieve binary segmentation mask")
    overlay_url: str = Field(..., description="Relative API URL to retrieve visual overlay")
    original_image_url: Optional[str] = Field(default=None, description="Relative API URL to retrieve original uploaded image")
    saved_to_history: bool = Field(default=False, description="Whether prediction record was persisted to database history")
    processing_strategy: Optional[str] = Field(default="direct", description="Inference strategy used ('direct', 'letterbox', or 'tiled')")
    gsd_status: str = Field(default="UNAVAILABLE", description="Ground Sampling Distance calibration status ('VALIDATED', 'UNAVAILABLE', or 'USER_SUPPLIED')")
    source_resolution: Optional[str] = Field(default=None, description="Calibrated pixel resolution in metric or map units")
    source_crs: Optional[str] = Field(default=None, description="Coordinate Reference System name and EPSG code if georeferenced")
    metadata_provenance: Optional[str] = Field(default="unreferenced", description="Source of geospatial resolution and spatial reference")
    calibration_type: Optional[str] = Field(default="NONE", description="Calibration method applied: 'AUTOMATIC', 'MANUAL', or 'NONE'")
    calibration_status: str = Field(default="UNAVAILABLE", description="Calibration status: 'EXTRACTED_METADATA', 'USER_SUPPLIED', 'UNAVAILABLE', or 'REJECTED'")
    calibration_source: Optional[str] = Field(default=None, description="Calibration reference description or sensor citation")
    custom_gsd: Optional[float] = Field(default=None, description="User-supplied GSD in meters per pixel if provided")
    calibration_rejection_reason: Optional[str] = Field(default=None, description="Detailed explanation if custom calibration was rejected")
    physical_area_insufficient_reason: Optional[str] = Field(default=None, description="Reason why physical area could not be calculated")
    energy_insufficient_reasons: Optional[list[str]] = Field(default=None, description="Reasons why solar energy potential could not be calculated")
    co2_insufficient_reasons: Optional[list[str]] = Field(default=None, description="Reasons why avoided CO2 emissions could not be calculated")


class PredictionHistoryItem(BaseModel):
    """Schema for an individual persisted prediction record in history."""
    prediction_id: str = Field(..., description="Unique UUID identifier for this inference run")
    created_at_utc: str = Field(..., description="ISO 8601 UTC creation timestamp")
    filename: Optional[str] = Field(default=None, description="Original uploaded image filename")
    status: str = Field(default="success", description="Prediction processing status")
    model_name: str = Field(default="HalfUNet", description="Model architecture used")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    total_image_pixels: int = Field(..., description="Total pixel count")
    solar_area_pixels: int = Field(..., description="Count of segmented solar pixels")
    solar_coverage_percent: float = Field(..., description="Percentage of solar coverage")
    detected_region_count: int = Field(..., description="Number of connected solar regions")
    largest_region_area_pixels: Optional[int] = Field(default=None, description="Largest region area")
    smallest_region_area_pixels: Optional[int] = Field(default=None, description="Smallest region area")
    mean_region_area_pixels: Optional[float] = Field(default=None, description="Mean region area")
    physical_area_m2: Optional[float] = Field(default=None, description="Physical area in m^2 (strictly null without verified GSD)")
    physical_area_hectares: Optional[float] = Field(default=None, description="Physical area in hectares")
    physical_area_status: str = Field(default="insufficient_data", description="Status of spatial resolution calibration")
    energy_status: str = Field(default="insufficient_data", description="Status of solar energy potential calculation")
    estimated_energy_kwh_year: Optional[float] = Field(default=None, description="Estimated solar energy potential (kWh/year)")
    co2_status: str = Field(default="insufficient_data", description="Status of CO2 avoidance calculation")
    co2_avoided_kg_year: Optional[float] = Field(default=None, description="Estimated CO2 avoided (kg/year)")
    co2_avoided_tonnes_year: Optional[float] = Field(default=None, description="Estimated CO2 avoided (metric tonnes/year)")
    latitude: Optional[float] = Field(default=None, description="Image-level scene centroid latitude")
    longitude: Optional[float] = Field(default=None, description="Image-level scene centroid longitude")
    georeferencing_level: str = Field(default="UNREFERENCED", description="Georeferencing classification")
    location_name: Optional[str] = Field(default=None, description="Approximate geographic region name")
    mask_url: str = Field(..., description="Relative URL to binary mask")
    overlay_url: str = Field(..., description="Relative URL to visual overlay")
    original_image_url: Optional[str] = Field(default=None, description="Relative URL to original uploaded image")
    gsd_status: str = Field(default="UNAVAILABLE", description="Ground Sampling Distance calibration status")
    source_resolution: Optional[str] = Field(default=None, description="Calibrated pixel resolution")
    source_crs: Optional[str] = Field(default=None, description="Coordinate Reference System name")
    metadata_provenance: Optional[str] = Field(default="unreferenced", description="Source of geospatial resolution and spatial reference")
    calibration_type: Optional[str] = Field(default="NONE", description="Calibration method applied: 'AUTOMATIC', 'MANUAL', or 'NONE'")
    calibration_status: Optional[str] = Field(default="UNAVAILABLE", description="Calibration status")
    calibration_source: Optional[str] = Field(default=None, description="Calibration reference description")
    custom_gsd: Optional[float] = Field(default=None, description="User-supplied GSD in meters per pixel if provided")
    calibration_rejection_reason: Optional[str] = Field(default=None, description="Rejection reason if invalid")
    persisted_at_utc: Optional[str] = Field(default=None, description="Timestamp when record was written to MongoDB")
    details: Optional[dict] = Field(default=None, description="Optional nested metadata details")

    class Config:
        extra = "ignore"


class PredictionHistoryListResponse(BaseModel):
    """Schema for paginated prediction history response."""
    success: bool = Field(default=True)
    total: int = Field(..., description="Total number of prediction records available")
    skip: int = Field(..., description="Number of records skipped")
    limit: int = Field(..., description="Number of records requested")
    items: list[PredictionHistoryItem] = Field(default_factory=list, description="Paginated prediction records")


class DatabaseHealthResponse(BaseModel):
    """Schema for database health and connection status."""
    status: str = Field(..., description="Database status ('connected', 'disconnected', or 'unconfigured')")
    configured: bool = Field(..., description="Whether a non-empty MongoDB URI is configured")
    connected: bool = Field(..., description="Whether active connection to MongoDB is verified")
    database: Optional[str] = Field(default=None, description="Configured database name")
    collection: Optional[str] = Field(default=None, description="Configured collection name")
    error: Optional[str] = Field(default=None, description="Connection error message if unavailable")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message summary")
    detail: Optional[Any] = Field(default=None, description="Detailed error information")

