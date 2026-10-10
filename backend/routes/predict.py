"""
SolarMap-India — Prediction & Artifact Retrieval Routes.

Endpoints:
- POST /predict: Processes uploaded aerial/satellite image through Half U-Net.
- GET /outputs/{prediction_id}/mask: Safely serves the generated binary mask.
- GET /outputs/{prediction_id}/overlay: Safely serves the visual detection overlay.
"""

import io
from pathlib import Path
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image

from backend.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, OUTPUT_DIR, UPLOAD_DIR
from backend.model_manager import model_manager
from backend.schemas import ErrorResponse, PredictResponse
from ml.inference.predict import predict_image

router = APIRouter(tags=["Inference"])


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
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        ) from exc

    # 7. Formulate Standardized Response
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
        physical_area_m2=result.physical_area_m2,
        physical_area_hectares=result.physical_area_hectares,
        physical_area_status=result.physical_area_status,
        mask_url=f"/outputs/{prediction_id}/mask",
        overlay_url=f"/outputs/{prediction_id}/overlay",
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
