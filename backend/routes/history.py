"""
SolarMap-India — Prediction History API Routes (Phase 8).

Endpoints:
- GET /api/predictions/history: Lists paginated prediction history sorted newest first.
- GET /api/predictions/history/{prediction_id}: Retrieves a single prediction record by ID.
- GET /predictions/history: Convenient alias without /api prefix.
- GET /predictions/history/{prediction_id}: Convenient alias without /api prefix.
- GET /health/db: Dedicated database health and connection status check.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from backend.database import db_manager
from backend.schemas import (
    DatabaseHealthResponse,
    ErrorResponse,
    PredictionHistoryItem,
    PredictionHistoryListResponse,
)

logger = logging.getLogger("solarmap.api.history")
router = APIRouter(tags=["History"])


def _validate_prediction_id(prediction_id: str) -> uuid.UUID:
    """Validates that prediction_id is a valid UUID4 format."""
    try:
        return uuid.UUID(prediction_id, version=4)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid prediction ID format. Must be a valid UUID4.",
        ) from exc


@router.get(
    "/api/predictions/history",
    response_model=PredictionHistoryListResponse,
    summary="List Prediction History (Paginated)",
    description="Retrieves a paginated list of previously saved solar predictions sorted newest first.",
    responses={
        200: {"description": "Paginated prediction records retrieved successfully."},
        503: {"model": ErrorResponse, "description": "Database service is unconfigured or unavailable."},
    },
)
@router.get(
    "/predictions/history",
    response_model=PredictionHistoryListResponse,
    include_in_schema=False,
)
async def list_prediction_history(
    skip: int = Query(0, ge=0, description="Number of records to skip (offset)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of records to return (1-100)"),
) -> PredictionHistoryListResponse:
    if not db_manager.is_connected:
        status_info = db_manager.get_status()
        err_msg = status_info.get("error") or "MongoDB is offline or unconfigured."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {err_msg}",
        )

    items, total = await db_manager.list_predictions(skip=skip, limit=limit)
    return PredictionHistoryListResponse(
        success=True,
        total=total,
        skip=skip,
        limit=limit,
        items=items,
    )


@router.get(
    "/api/predictions/history/{prediction_id}",
    response_model=PredictionHistoryItem,
    summary="Retrieve Individual Prediction Record",
    description="Fetches a single prediction history record by its UUID4 prediction identifier.",
    responses={
        200: {"description": "Prediction record found."},
        400: {"model": ErrorResponse, "description": "Invalid UUID4 format."},
        404: {"model": ErrorResponse, "description": "Prediction record not found."},
        503: {"model": ErrorResponse, "description": "Database service is unconfigured or unavailable."},
    },
)
@router.get(
    "/predictions/history/{prediction_id}",
    response_model=PredictionHistoryItem,
    include_in_schema=False,
)
async def get_prediction_history_item(prediction_id: str) -> PredictionHistoryItem:
    valid_id = str(_validate_prediction_id(prediction_id))

    if not db_manager.is_connected:
        status_info = db_manager.get_status()
        err_msg = status_info.get("error") or "MongoDB is offline or unconfigured."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {err_msg}",
        )

    doc = await db_manager.get_prediction(valid_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction history record '{prediction_id}' was not found.",
        )

    return PredictionHistoryItem(**doc)


@router.get(
    "/health/db",
    response_model=DatabaseHealthResponse,
    tags=["Health"],
    summary="Database Health & Connectivity Check",
    description="Returns detailed connectivity status of the MongoDB persistence layer without leaking secrets.",
)
async def get_database_health() -> DatabaseHealthResponse:
    info = db_manager.get_status()
    if info["connected"]:
        st = "connected"
    elif info["configured"]:
        st = "disconnected"
    else:
        st = "unconfigured"

    return DatabaseHealthResponse(
        status=st,
        configured=info["configured"],
        connected=info["connected"],
        database=info["database"],
        collection=info["collection"],
        error=info["error"],
    )
