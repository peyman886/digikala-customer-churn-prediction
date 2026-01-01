# =============================================================================
# Churn Prediction API - Dockerfile
# =============================================================================
#
# Build:
#   docker build -t churn-api .
#
# Run:
#   docker run -p 9000:9000 \
#       -v $(pwd)/models_v2:/app/models_v2 \
#       -v $(pwd)/user_features.csv:/app/user_features.csv \
#       churn-api
#
# =============================================================================

FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
#RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY __init__.py .
COPY config.py .
COPY schemas.py .
COPY services.py .
COPY main.py .
COPY models/ ./models/

# Environment
ENV API_PORT=9000
ENV MODEL_DIR=/app/models_v2
ENV FEATURES_PATH=/app/user_features.csv
ENV DEVICE=cpu
ENV LOG_LEVEL=INFO

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
