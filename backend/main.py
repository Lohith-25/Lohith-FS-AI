"""
SolarMap-India — Backend Inference Application Entrypoint.

Starts FastAPI application, initializes Half U-Net model on startup,
configures CORS middleware, registers health and prediction routes,
and exposes interactive OpenAPI documentation.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    FRONTEND_DIST_DIR,
)
from backend.database import db_manager
from backend.model_manager import model_manager
from backend.routes.health import router as health_router
from backend.routes.history import router as history_router
from backend.routes.predict import router as predict_router

# Configure backend logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("solarmap.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle."""
    logger.info("Starting up %s (v%s)...", API_TITLE, API_VERSION)
    # Preload the Half U-Net checkpoint into memory
    model_manager.load_model()
    # Connect to MongoDB if configured
    await db_manager.connect()
    yield
    # Cleanly disconnect database client
    await db_manager.disconnect()
    logger.info("Shutting down %s.", API_TITLE)


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register core API routes first so they take priority
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(history_router)


def _get_api_metadata() -> dict:
    """Returns standardized API root metadata and navigation endpoints."""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "db_health": "/health/db",
        "predict": "/predict",
        "history": "/api/predictions/history",
    }


@app.get(
    "/api",
    tags=["Root"],
    summary="API Root Information",
    description="Returns service metadata and navigation links to documentation and health check.",
)
def read_api_root():
    return _get_api_metadata()


@app.get(
    "/",
    tags=["Root"],
    summary="Application Entry Point",
    description="Serves the React frontend application when requested by browsers, or API service metadata when requested as JSON.",
)
def read_root(request: Request):
    accept_header = request.headers.get("accept", "")
    index_html = FRONTEND_DIST_DIR / "index.html"

    # If the client requests HTML (standard browser navigation) and frontend is built, serve index.html
    if index_html.is_file() and "text/html" in accept_header:
        return FileResponse(index_html)

    # Otherwise return API metadata
    return _get_api_metadata()


# Mount static frontend assets (/assets) if frontend dist exists
_assets_dir = FRONTEND_DIST_DIR / "assets"
if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")


@app.get(
    "/{filename:path}",
    include_in_schema=False,
)
async def serve_static_or_spa(request: Request, filename: str):
    """
    Safely serves root static assets (favicon.svg, sample_aerial.png) and provides
    SPA client-side navigation fallback without intercepting API routes.
    """
    # Guard: Strictly prevent intercepting API routes
    api_prefixes = ("api", "predict", "outputs", "health", "docs", "redoc", "openapi.json")
    for prefix in api_prefixes:
        if filename == prefix or filename.startswith(f"{prefix}/"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API route '/{filename}' not found",
            )

    # Guard: Check for static asset files in frontend/dist
    if FRONTEND_DIST_DIR.is_dir():
        candidate = (FRONTEND_DIST_DIR / filename).resolve()
        try:
            # Prevent path traversal attacks
            candidate.relative_to(FRONTEND_DIST_DIR.resolve())
            if candidate.is_file():
                return FileResponse(candidate)
        except ValueError:
            pass

        # SPA client-side routing fallback: if client requests HTML and index.html exists
        index_html = FRONTEND_DIST_DIR / "index.html"
        if index_html.is_file() and "text/html" in request.headers.get("accept", ""):
            return FileResponse(index_html)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")



# Custom Exception Handlers for Standardized JSON Responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "detail": None,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error in request parameters",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
        },
    )
