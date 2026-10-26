# ==============================================================================
# SolarMap-India — Multi-Stage Production Dockerfile
#
# Stage 1: Build the React 19 + Vite frontend into static production assets.
# Stage 2: Minimal Python 3.11 slim runtime running FastAPI and PyTorch CPU.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Frontend Builder
# ------------------------------------------------------------------------------
FROM node:22-alpine AS frontend-builder

WORKDIR /build

# Install dependencies reproducing package-lock.json
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy frontend source files and build production bundle
COPY frontend/ ./
# Empty VITE_API_BASE_URL ensures relative API calls in production
ENV VITE_API_BASE_URL=""
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Final Production Runtime (Python 3.11 slim)
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Set environment variables for Python performance, paths, and security
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ENVIRONMENT=production \
    MODEL_PATH=/app/outputs/HalfUNet/models/halfunet_best.pth \
    FRONTEND_DIST_DIR=/app/frontend/dist

# Install essential system runtime dependencies:
# - libgl1 and libglib2.0-0 for OpenCV headless operations
# - curl for container orchestration health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Install Python dependencies:
# 1. Install CPU-only PyTorch first to prevent downloading large CUDA binary wheels
RUN pip install --no-cache-dir torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining application requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create runtime directories with appropriate permissions
RUN mkdir -p \
    /app/outputs/HalfUNet/models \
    /app/backend/uploads \
    /app/backend/generated \
    /app/frontend/dist \
    && chown -R appuser:appuser /app

# Copy built frontend static assets from Stage 1
COPY --from=frontend-builder --chown=appuser:appuser /build/dist /app/frontend/dist

# Copy application source code
COPY --chown=appuser:appuser backend /app/backend
COPY --chown=appuser:appuser ml /app/ml
COPY --chown=appuser:appuser config /app/config
COPY --chown=appuser:appuser scripts /app/scripts
COPY --chown=appuser:appuser splits /app/splits

# Switch to non-root user
USER appuser

# Expose API & application port
EXPOSE 8000

# Docker healthcheck querying /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI via uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
