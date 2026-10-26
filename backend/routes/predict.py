"""
SolarMap-India — Prediction & Artifact Retrieval Routes.

Endpoints:
- POST /predict: Processes uploaded aerial/satellite image through Half U-Net.
- GET /outputs/{prediction_id}/mask: Safely serves the generated binary mask.
- GET /outputs/{prediction_id}/overlay: Safely serves the visual detection overlay.
"""

from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from typing import Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image

from backend.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, OUTPUT_DIR, UPLOAD_DIR
from backend.database import db_manager
from backend.model_manager import model_manager
from backend.schemas import ErrorResponse, PredictResponse
from ml.area.calibration import (
    calculate_calibrated_area,
    resolve_image_calibration,
    validate_user_gsd,
)
from ml.energy.area import PhysicalAreaCalculator
from ml.energy.assumptions import EnergyModelConfig
from ml.energy.energy_model import SolarEnergyModel
from ml.environmental.co2_model import CO2AvoidanceModel
from ml.environmental.emission_factors import GridEmissionConfig
from ml.external_data.solar_resource import SolarResourceService
from ml.geospatial.coordinate_mapping import GeospatialCatalog
from ml.geospatial.metadata import extract_geospatial_metadata
from ml.inference.predict import predict_image

logger = logging.getLogger("solarmap.api.predict")
router = APIRouter(tags=["Inference"])

_geospatial_catalog: Optional[GeospatialCatalog] = None


def get_geospatial_catalog() -> Optional[GeospatialCatalog]:
    global _geospatial_catalog
    if _geospatial_catalog is None:
        try:
            _geospatial_catalog = GeospatialCatalog()
        except Exception as exc:
            logger.warning("Geospatial catalog could not be loaded: %s", exc)
            return None
    return _geospatial_catalog


def _validate_prediction_id(prediction_id: str) -> uuid.UUID:
    """Validates that prediction_id is a valid UUID4, preventing path traversal attacks."""
    try:
        return uuid.UUID(prediction_id, version=4)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid prediction ID. Must be a valid UUID4.",
        ) from exc


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Upload Satellite Image & Run Solar Segmentation",
    description=(
        "Accepts a satellite or aerial image, runs the Half U-Net segmentation pipeline, "
        "and returns standardized pixel-based solar metrics and artifact access URLs."
    ),
    responses={
        200: {"description": "Inference succeeded with standardized pixel-area metrics."},
        400: {"model": ErrorResponse, "description": "Invalid file format, empty, or corrupt image data."},
        413: {"model": ErrorResponse, "description": "Uploaded image file exceeds size limit."},
        500: {"model": ErrorResponse, "description": "Internal server or inference failure."},
        503: {"model": ErrorResponse, "description": "Model checkpoint is not loaded or available."},
    },
)
async def predict_solar(
    file: UploadFile = File(..., description="Target aerial or satellite image file"),
    threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Foreground solar probability threshold (default: 0.5)",
    ),
    min_component_area: int = Query(
        20,
        ge=1,
        description="Minimum pixel area threshold to retain connected solar regions (default: 20)",
    ),
    strategy: str = Query(
        "auto",
        pattern="^(auto|letterbox|tiled|direct)$",
        description="Inference strategy: 'auto', 'letterbox', 'tiled', or 'direct' (default: auto)",
    ),
    custom_gsd: Optional[float] = Query(
        None,
        description="Optional user-supplied Ground Sampling Distance (GSD) in meters/pixel for ordinary rasters",
    ),
    calibration_source: Optional[str] = Query(
        None,
        description="Optional citation or reference for user-supplied GSD",
    ),
    custom_gsd_form: Optional[float] = Form(
        None,
        alias="custom_gsd",
        description="Optional user-supplied GSD in multipart form-data",
    ),
    calibration_source_form: Optional[str] = Form(
        None,
        alias="calibration_source",
        description="Optional reference for user-supplied GSD in multipart form-data",
    ),
    use_cea_benchmark: bool = Query(
        False,
        description="Whether to apply CEA India benchmark scenario for energy and CO2 estimation if physical area is calibrated",
    ),
    annual_ghi: Optional[float] = Query(
        None,
        description="Optional annual Global Horizontal Irradiance in kWh/m2/year",
    ),
) -> PredictResponse:
    # 1. Model Availability Check
    if not model_manager.is_ready:
        status_info = model_manager.get_status()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {status_info.get('error') or 'Model not initialized'}",
        )

    # 2. File Metadata Validation
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image extension '{ext}'. "
                f"Supported formats: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # 3. Read Content and Size Validation
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    # 4. Image Decoding & Integrity Verification
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image data: {exc}",
        ) from exc

    # 5. Staging Upload & Output Directories
    prediction_id = str(uuid.uuid4())
    pred_upload_dir = UPLOAD_DIR / prediction_id
    pred_output_dir = OUTPUT_DIR / prediction_id

    pred_upload_dir.mkdir(parents=True, exist_ok=True)
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = pred_upload_dir / f"input{ext}"
    input_image_path.write_bytes(content)

    # 6. Inference Execution using Existing Half U-Net Pipeline
    try:
        result = predict_image(
            image_path=input_image_path,
            model=model_manager.model,
            output_dir=pred_output_dir,
            threshold=threshold,
            min_component_area=min_component_area,
            strategy=strategy,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image content: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        ) from exc

    # 7. Spatial Calibration & Geospatial Resolution (Two-Path Calibration Architecture)
    eff_custom_gsd = custom_gsd if custom_gsd is not None else custom_gsd_form
    eff_calibration_source = calibration_source if calibration_source is not None else calibration_source_form

    # Validate custom_gsd if provided; strictly reject non-positive, infinite, NaN, or out-of-bounds GSD
    if eff_custom_gsd is not None:
        is_val, parsed_val, clean_src, reject_reason = validate_user_gsd(eff_custom_gsd, eff_calibration_source)
        if not is_val:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid spatial calibration: {reject_reason}",
            )
        eff_custom_gsd = parsed_val
        eff_calibration_source = clean_src

    # Extract automatic metadata from raster IFD / GeoKeys
    geo_meta = extract_geospatial_metadata(content)

    # Resolve calibration adhering to explicit precedence (Automatic GeoTIFF takes precedence over Manual GSD)
    calibration = resolve_image_calibration(
        geo_metadata=geo_meta,
        custom_gsd=eff_custom_gsd,
        calibration_source=eff_calibration_source,
    )

    geo_catalog = get_geospatial_catalog()
    geo_record = geo_catalog.lookup(file.filename) if geo_catalog else None

    # Derive scene coordinates: prioritize embedded GeoTIFF metadata if geographic/projected centroid exists, else fallback to Level B catalog
    latitude = round(geo_record.latitude, 6) if geo_record else None
    longitude = round(geo_record.longitude, 6) if geo_record else None
    georef_level = "LEVEL_A" if geo_meta.is_georeferenced else ("LEVEL_B" if geo_record else "UNREFERENCED")
    location_name = "Surat, Gujarat, India" if (geo_record and geo_record.is_within_india_bounds) else (
        f"{geo_meta.crs_name}" if geo_meta.is_georeferenced else ("Catalog Location" if geo_record else None)
    )

    # 8. Physical Area, Energy Potential, and CO2 Environmental Calculations
    area_calc = calculate_calibrated_area(
        solar_pixels=result.solar_pixels,
        total_pixels=result.total_image_pixels,
        calibration=calibration,
    )

    # Derive solar resource (annual GHI) if required
    effective_annual_ghi = annual_ghi
    if effective_annual_ghi is None and (latitude is not None and longitude is not None):
        try:
            solar_service = SolarResourceService()
            solar_res = solar_service.get_solar_resource(latitude=latitude, longitude=longitude)
            if solar_res.status == "available":
                effective_annual_ghi = solar_res.annual_value_kwh_m2_year
        except Exception as exc:
            logger.debug("Solar resource retrieval bypassed: %s", exc)

    if use_cea_benchmark:
        energy_cfg = EnergyModelConfig.create_benchmark_scenario()
        co2_cfg = GridEmissionConfig.create_cea_benchmark_scenario()
    else:
        energy_cfg = EnergyModelConfig.create_default()
        co2_cfg = GridEmissionConfig.create_default()

    energy_calc = SolarEnergyModel.estimate_energy(
        area_result=area_calc,
        annual_ghi_kwh_m2_year=effective_annual_ghi,
        config=energy_cfg,
    )
    co2_calc = CO2AvoidanceModel.estimate_avoided_emissions(
        energy_input=energy_calc,
        config=co2_cfg,
    )

    # Determine human-readable source resolution string and status
    if calibration.is_calibrated:
        if calibration.calibration_type == "AUTOMATIC":
            source_res_str = (
                f"{calibration.gsd_x_meters:.3f}m x {calibration.gsd_y_meters:.3f}m/px"
                if (calibration.gsd_x_meters is not None and calibration.gsd_y_meters is not None)
                else f"{calibration.gsd_meters:.3f}m/px"
            )
            res_gsd_status = "VALIDATED"
            res_provenance = geo_meta.metadata_provenance or "geotiff_embedded_tags"
        else:
            source_res_str = f"{calibration.gsd_meters:.4f} m/pixel (User-Supplied)"
            res_gsd_status = "USER_SUPPLIED"
            res_provenance = calibration.calibration_source
    else:
        source_res_str = None
        res_gsd_status = "UNAVAILABLE"
        res_provenance = geo_meta.metadata_provenance or "unreferenced"

    # 9. Persist to MongoDB (Phase 8 & 13: Prediction History with Analytics)
    saved_to_history = False
    now_utc = datetime.now(timezone.utc).isoformat()
    history_record = {
        "prediction_id": prediction_id,
        "created_at_utc": now_utc,
        "filename": file.filename,
        "status": "success",
        "model_name": "HalfUNet",
        "image_width": result.image_width,
        "image_height": result.image_height,
        "total_image_pixels": result.total_image_pixels,
        "solar_area_pixels": result.solar_area_pixels,
        "solar_coverage_percent": round(result.solar_coverage_percent, 4),
        "detected_region_count": result.detected_region_count,
        "largest_region_area_pixels": result.largest_region_area_pixels,
        "smallest_region_area_pixels": result.smallest_region_area_pixels,
        "mean_region_area_pixels": result.mean_region_area_pixels,
        "physical_area_m2": area_calc.solar_area_m2,
        "physical_area_hectares": round(area_calc.solar_area_m2 / 10000.0, 4) if area_calc.solar_area_m2 is not None else None,
        "physical_area_status": area_calc.physical_area_status,
        "energy_status": energy_calc.energy_status,
        "estimated_energy_kwh_year": energy_calc.estimated_energy_kwh_year,
        "co2_status": co2_calc.co2_status,
        "co2_avoided_kg_year": co2_calc.co2_avoided_kg_year,
        "co2_avoided_tonnes_year": co2_calc.co2_avoided_tonnes_year,
        "latitude": latitude,
        "longitude": longitude,
        "georeferencing_level": georef_level,
        "location_name": location_name,
        "mask_url": f"/outputs/{prediction_id}/mask",
        "overlay_url": f"/outputs/{prediction_id}/overlay",
        "original_image_url": f"/outputs/{prediction_id}/original",
        "applied_threshold": threshold,
        "applied_min_component_area": min_component_area,
        "processing_strategy": result.processing_strategy,
        "gsd_status": res_gsd_status,
        "source_resolution": source_res_str,
        "source_crs": geo_meta.crs_name,
        "metadata_provenance": res_provenance,
        "calibration_type": calibration.calibration_type,
        "calibration_status": calibration.calibration_status,
        "calibration_source": calibration.calibration_source,
        "custom_gsd": calibration.gsd_meters if calibration.is_user_supplied else None,
        "calibration_rejection_reason": calibration.rejection_reason,
        "details": {
            "area": area_calc.to_dict(),
            "energy": energy_calc.to_dict(),
            "co2": co2_calc.to_dict(),
            "geospatial": geo_meta.to_dict(),
            "calibration": calibration.to_dict(),
        },
    }

    if db_manager.is_connected:
        try:
            saved_to_history = await db_manager.save_prediction(history_record)
        except Exception as exc:
            logger.warning("Failed to persist prediction '%s' to history: %s", prediction_id, exc)
            saved_to_history = False

    # 10. Formulate Standardized Response
    return PredictResponse(
        success=True,
        model="HalfUNet",
        prediction_id=prediction_id,
        image_width=result.image_width,
        image_height=result.image_height,
        total_image_pixels=result.total_image_pixels,
        solar_area_pixels=result.solar_area_pixels,
        solar_coverage_percent=round(result.solar_coverage_percent, 4),
        detected_region_count=result.detected_region_count,
        largest_region_area_pixels=result.largest_region_area_pixels,
        smallest_region_area_pixels=result.smallest_region_area_pixels,
        mean_region_area_pixels=result.mean_region_area_pixels,
        physical_area_m2=area_calc.solar_area_m2,
        physical_area_hectares=round(area_calc.solar_area_m2 / 10000.0, 4) if area_calc.solar_area_m2 is not None else None,
        physical_area_status=area_calc.physical_area_status,
        energy_status=energy_calc.energy_status,
        estimated_energy_kwh_year=energy_calc.estimated_energy_kwh_year,
        co2_status=co2_calc.co2_status,
        co2_avoided_kg_year=co2_calc.co2_avoided_kg_year,
        co2_avoided_tonnes_year=co2_calc.co2_avoided_tonnes_year,
        latitude=latitude,
        longitude=longitude,
        georeferencing_level=georef_level,
        location_name=location_name,
        mask_url=f"/outputs/{prediction_id}/mask",
        overlay_url=f"/outputs/{prediction_id}/overlay",
        original_image_url=f"/outputs/{prediction_id}/original",
        saved_to_history=saved_to_history,
        processing_strategy=result.processing_strategy,
        gsd_status=res_gsd_status,
        source_resolution=source_res_str,
        source_crs=geo_meta.crs_name,
        metadata_provenance=res_provenance,
        calibration_type=calibration.calibration_type,
        calibration_status=calibration.calibration_status,
        calibration_source=calibration.calibration_source,
        custom_gsd=calibration.gsd_meters if calibration.is_user_supplied else None,
        calibration_rejection_reason=calibration.rejection_reason,
        physical_area_insufficient_reason=area_calc.insufficient_reason,
        energy_insufficient_reasons=energy_calc.unavailability_reasons if energy_calc.energy_status != "estimated" else None,
        co2_insufficient_reasons=co2_calc.unavailability_reasons if co2_calc.co2_status != "estimated" else None,
    )



@router.get(
    "/outputs/{prediction_id}/mask",
    summary="Retrieve Binary Segmentation Mask",
    description="Securely streams the generated binary segmentation mask PNG for a specific prediction.",
    responses={
        200: {"content": {"image/png": {}}, "description": "Binary segmentation mask image."},
        400: {"description": "Invalid prediction ID format."},
        404: {"description": "Mask file not found."},
    },
)
def get_prediction_mask(prediction_id: str) -> FileResponse:
    validated_id = str(_validate_prediction_id(prediction_id))
    masks_dir = OUTPUT_DIR / validated_id / "masks"

    if not masks_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction output for '{prediction_id}' was not found.",
        )

    # Find mask artifact
    mask_files = list(masks_dir.glob("*_mask.png"))
    if not mask_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Binary mask artifact not found for this prediction.",
        )

    target_file = mask_files[0]
    # Enforce filesystem containment
    if not target_file.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to requested file path is restricted.",
        )

    return FileResponse(
        path=target_file,
        media_type="image/png",
        filename=f"{prediction_id}_mask.png",
    )


@router.get(
    "/outputs/{prediction_id}/overlay",
    summary="Retrieve Visual Segmentation Overlay",
    description="Securely streams the visual overlay PNG with solar panels highlighted over original imagery.",
    responses={
        200: {"content": {"image/png": {}}, "description": "Visual segmentation overlay image."},
        400: {"description": "Invalid prediction ID format."},
        404: {"description": "Overlay file not found."},
    },
)
def get_prediction_overlay(prediction_id: str) -> FileResponse:
    validated_id = str(_validate_prediction_id(prediction_id))
    overlays_dir = OUTPUT_DIR / validated_id / "overlays"

    if not overlays_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction output for '{prediction_id}' was not found.",
        )

    overlay_files = list(overlays_dir.glob("*_overlay.png"))
    if not overlay_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visual overlay artifact not found for this prediction.",
        )

    target_file = overlay_files[0]
    # Enforce filesystem containment
    if not target_file.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to requested file path is restricted.",
        )

    return FileResponse(
        path=target_file,
        media_type="image/png",
        filename=f"{prediction_id}_overlay.png",
    )


@router.get(
    "/outputs/{prediction_id}/original",
    summary="Retrieve Original Input Image",
    description="Securely streams the uploaded original input raster for a specific prediction.",
    responses={
        200: {"content": {"image/*": {}}, "description": "Original input image."},
        400: {"description": "Invalid prediction ID format."},
        404: {"description": "Original image not found."},
    },
)
def get_prediction_original_image(prediction_id: str) -> FileResponse:
    validated_id = str(_validate_prediction_id(prediction_id))
    upload_dir = UPLOAD_DIR / validated_id

    if not upload_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload directory for '{prediction_id}' was not found.",
        )

    candidates = list(upload_dir.glob("input.*"))
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original image file not found for this prediction.",
        )

    target_file = candidates[0]
    if not target_file.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to requested file path is restricted.",
        )

    ext = target_file.suffix.lower()
    media_type = (
        "image/png"
        if ext == ".png"
        else ("image/jpeg" if ext in (".jpg", ".jpeg") else "application/octet-stream")
    )
    return FileResponse(
        path=target_file,
        media_type=media_type,
        filename=target_file.name,
    )
