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
from fastapi.responses import JSONResponse

from backend.config import API_DESCRIPTION, API_TITLE, API_VERSION
from backend.model_manager import model_manager
from backend.routes.health import router as health_router
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
    yield
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router)
app.include_router(predict_router)


@app.get(
    "/",
    tags=["Root"],
    summary="API Root Information",
    description="Returns service metadata and navigation links to documentation and health check.",
)
def read_root():
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


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
