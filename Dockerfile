# ==============================================================================
# FaceCheck API - Hardened Multi-Stage Production Dockerfile
#
# Security specifications:
# - Multi-stage build (build dependencies omitted from runtime image)
# - Non-root execution (UID/GID 10001: facecheck)
# - Read-only root filesystem compatible (read_only: true)
# - All Linux capabilities dropped (cap_drop: ALL)
# - No new privileges escalation (no-new-privileges: true)
# - Offline runtime (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, NO_PROXY)
# ==============================================================================

# Stage 1: Build stage
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY wheelhouse/ ./wheelhouse/

# Install python wheels to target directory (offline from ./wheelhouse if available, or requirements)
RUN if [ -d "./wheelhouse" ] && [ "$(ls -A ./wheelhouse 2>/dev/null)" ]; then \
        pip install --no-cache-dir --no-index --find-links=./wheelhouse --prefix=/install -r requirements.txt; \
    else \
        pip install --no-cache-dir --prefix=/install -r requirements.txt; \
    fi


# Stage 2: Hardened Runtime stage
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NO_PROXY="*"

WORKDIR /app

# Install minimal runtime libraries (libglib2.0, libgl1 for OpenCV/image ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root service group and user with fixed UID/GID
RUN groupadd -r -g 10001 facecheck && \
    useradd -r -u 10001 -g facecheck -d /app -s /sbin/nologin facecheck

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code and offline assets
COPY --chown=root:facecheck config.yaml /app/config.yaml
COPY --chown=root:facecheck app/ /app/app/
COPY --chown=root:facecheck models/ /app/models/

# Make application code strictly read-only for runtime process
RUN chmod -R 550 /app && \
    chmod 440 /app/config.yaml

# Create writable runtime storage & logs with restricted permissions
RUN mkdir -p /app/logs /app/storage/encrypted_images /tmp && \
    chown -R facecheck:facecheck /app/logs /app/storage /tmp && \
    chmod 770 /app/logs /app/storage /app/storage/encrypted_images /tmp

# Run as unprivileged non-root user
USER facecheck:facecheck

EXPOSE 8000

# Native healthcheck using built-in Python urllib (no curl required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')" || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
