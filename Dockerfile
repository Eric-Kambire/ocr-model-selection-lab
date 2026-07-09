# Use a lightweight python base image
FROM python:3.11-slim

ARG APP_VERSION=1.0.0
ARG SOURCE_URL=local-workspace

LABEL org.opencontainers.image.title="OCR Model Selection Lab" \
      org.opencontainers.image.description="Extensible OCR model benchmarking UI and CLI" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="${SOURCE_URL}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    PIP_NO_CACHE_DIR="1"

# Set working directory
WORKDIR /app

# The portable CPU image uses Ollama on the host and does not embed PyTorch.
# This keeps the image small enough to share. EasyOCR/CUDA remains available
# through Dockerfile.gpu.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/runs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://127.0.0.1:7860/ || exit 1

CMD ["python", "main.py"]
