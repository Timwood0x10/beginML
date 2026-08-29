#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Deploy AIScope (backend FastAPI + frontend React) to a Linux server
# ---------------------------------------------------------------
# Assumptions:
#   * Running as root or via sudo (needs write access to /opt and systemd)
#   * System has python>=3.10, uv, node, npm, nginx installed
#   * Domain (or IP) will be configured in /etc/nginx/sites-available/aiscope.conf
#
# 1. Build frontend
# 2. Install backend deps via uv
# 3. Copy files to /opt/aiscope
# 4. Install systemd service for backend
# 5. Install nginx config and enable site
# 6. Start/reload services

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ROOT="/opt/aiscope"
FRONTEND_SRC="$PROJECT_ROOT/frontend"
BACKEND_SRC="$PROJECT_ROOT/backend"

echo "=== Building frontend ==="
cd "$FRONTEND_SRC"
npm ci    # clean install
npm run build

echo "=== Installing backend dependencies ==="
cd "$BACKEND_SRC"
uv sync   # reads pyproject.toml (includes scipy now)

# ---------------------------------------------------------------
# Prepare deployment directory
# ---------------------------------------------------------------
echo "=== Preparing $DEPLOY_ROOT ==="
mkdir -p "$DEPLOY_ROOT/backend"
mkdir -p "$DEPLOY_ROOT/frontend"

# Copy backend source
rsync -a --delete "$BACKEND_SRC/" "$DEPLOY_ROOT/backend/"
# Copy built frontend assets (dist) only
rsync -a --delete "$FRONTEND_SRC/dist/" "$DEPLOY_ROOT/frontend/"

# ---------------------------------------------------------------
# Systemd service
# ---------------------------------------------------------------
echo "=== Installing systemd service ==="
cp "$PROJECT_ROOT/deploy/aiscope.service" /etc/systemd/system/aiscope.service
systemctl daemon-reload
systemctl enable aiscope.service
systemctl restart aiscope.service

# ---------------------------------------------------------------
# Nginx configuration
# ---------------------------------------------------------------
NGINX_CONF="/etc/nginx/sites-available/aiscope.conf"
cp "$PROJECT_ROOT/deploy/nginx.conf" "$NGINX_CONF"
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/aiscope.conf
nginx -t   # test config
systemctl reload nginx

echo "=== Deployment complete ==="
echo "Frontend served at http://<your-domain-or-ip>"
echo "API backend proxy at http://<your-domain-or-ip>/api"
