# AIScope backend — single-container FastAPI + sklearn + notes data.
#
# HOW TO DEPLOY (one-click, no server, no nginx/systemd):
#   The GIT REPO ROOT is $HOME/Documents/ml (git rev-parse --show-toplevel).
#   Inside it everything lives under other/, so all COPY paths are other/…
#   Build context = the GIT REPO ROOT:
#     docker build -f other/ai-learning-platform/Dockerfile -t aiscope .
#
#   On Render / Railway / Fly:
#     - Root Directory:  (repo root / leave empty)
#     - Dockerfile Path: other/ai-learning-platform/Dockerfile
#   The platform builds from GitHub automatically.

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
# NOTES_ROOT in app.py defaults to the "parents[2]" of backend/, i.e.
# $HOME/Documents/ml/other — which is exactly where zh/ en/ images/ live.
# Here we place the same layout under /data and point NOTES_ROOT at it.
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
