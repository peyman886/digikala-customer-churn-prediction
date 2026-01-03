# =============================================================================
# Churn Prediction API - GPU Dockerfile (CUDA 12.8)
# =============================================================================
# This Dockerfile builds the API with GPU (CUDA) support for faster inference
#
# Build:   docker build -t churn-api:gpu -f Dockerfile .
# Run:     docker run --gpus all -p 9000:9000 churn-api:gpu
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# -----------------------------------------------------------------------------
# Stage 2: Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir \
    torch==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Production image
# -----------------------------------------------------------------------------
FROM dependencies AS production

WORKDIR /app

# Copy application code
COPY __init__.py .
COPY config.py .
COPY schemas.py .
COPY services.py .
COPY main.py .
COPY models/ ./models/

# Environment variables
ENV API_PORT=9000
ENV MODEL_DIR=/app/models_v2
ENV FEATURES_PATH=/app/user_features.csv
ENV DEVICE=auto
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
