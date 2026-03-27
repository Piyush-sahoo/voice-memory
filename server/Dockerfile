FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy everything to /app/env (where the server runs from)
COPY . /app/env

# Also put README at /app/README.md so OpenEnv web UI can find it
COPY README.md /app/README.md

WORKDIR /app/env

# Install Python deps
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "openai>=1.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0"

# PYTHONPATH so imports work
ENV PYTHONPATH="/app/env"

# Enable the built-in OpenEnv web UI at /web
ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
