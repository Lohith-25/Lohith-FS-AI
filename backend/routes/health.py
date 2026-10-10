"""
SolarMap-India — Health Check Route.

Provides GET /health for container readiness, orchestration, and model availability.
"""

from fastapi import APIRouter, Response, status

from backend.model_manager import model_manager
from backend.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API & Model Health Check",
    description="Returns operational health and verifies whether the Half U-Net checkpoint is loaded and ready.",
    responses={
        200: {"description": "API and model are healthy and ready to serve inference requests."},
        503: {"description": "Model is not ready or checkpoint is missing."},
    },
)
def get_health(response: Response) -> HealthResponse:
    info = model_manager.get_status()
    if not info["is_ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="unhealthy",
            model=info["model_name"],
            checkpoint_available=info["checkpoint_available"],
            checkpoint_path=info["checkpoint_path"],
            device=info["device"],
            error=info["error"] or "Model checkpoint is not loaded or missing",
        )

    return HealthResponse(
        status="healthy",
        model=info["model_name"],
        checkpoint_available=info["checkpoint_available"],
        checkpoint_path=info["checkpoint_path"],
        device=info["device"],
        error=None,
    )
