# AIScope backend — single-container FastAPI + sklearn + notes data.
#
# This file sits at the GIT REPO ROOT ($HOME/Documents/ml) ON PURPOSE:
# Render / Railway automatically detect a root-level Dockerfile and use Docker
# — no manual "Dockerfile Path" needed. The build context is the repo root,
# and everything lives under other/ here, so every COPY path is other/….
#
#   docker build -f Dockerfile -t aiscope .     # from the repo root

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal OS libs required by sklearn/scipy wheels (OpenMP) and pymupdf.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- 1. Backend dependencies (cached layer) ----
COPY other/ai-learning-platform/backend/pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.34.0" \
        "scikit-learn>=1.6.0" \
        "numpy>=2.1.0" \
        "python-multipart>=0.0.20" \
        "markdown>=3.7" \
        "pymdown-extensions>=10.11" \
        "pygments>=2.18.0" \
        "latex2mathml>=3.81.0" \
        "pymupdf>=1.28.2" \
        "scipy>=1.14.0"

# ---- 2. Backend source ----
COPY other/ai-learning-platform/backend /app

# ---- 3. Note data (repo root: other/zh, other/en, other/images) ----
COPY other/zh /data/zh
COPY other/en /data/en
COPY other/images /data/images

ENV NOTES_ROOT=/data

EXPOSE 8000

# Health probe against the backend's own /api/health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

# Render/Railway inject $PORT; fall back to 8000 locally.
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
