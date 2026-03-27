FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/env

WORKDIR /app/env

# Install Python deps directly with pip (simple, no uv needed)
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "openai>=1.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0"

# Set PYTHONPATH so imports work
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Enable the built-in OpenEnv web UI at /web
ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
