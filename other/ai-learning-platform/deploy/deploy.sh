#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
#  One‑click AIScope installer (FastAPI + React/Vite)
# --------------------------------------------------------------
# Usage (run as root or with sudo):
#   ./deploy.sh                           # use current directory
#   ./deploy.sh /path/to/project          # deploy a specific path
#   ./deploy.sh https://github.com/...git # clone then install
#   export DOMAIN=your.domain.com && ./deploy.sh  # also request HTTPS
# --------------------------------------------------------------

log()  { echo -e "\033[1;36m[+] $*\033[0m"; }
error() { echo -e "\033[1;31m[-] $*\033[0m" >&2; exit 1; }

# ---------- Helper: run as root if not root ----------
run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

# ---------- Detect whether systemd is available ----------
if [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; then
    HAS_SYSTEMD=1
else
    HAS_SYSTEMD=0
    log "systemd not detected — will start services directly in the background"
fi

# ---------- Helper: start/stop services ----------
PID_DIR="/tmp/aiscope"
BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"
BACKEND_LOG="/tmp/aiscope/backend.log"
FRONTEND_LOG="/tmp/aiscope/frontend.log"

start_backend() {
    run_as_root mkdir -p "$PID_DIR"
    log "Starting backend on http://127.0.0.1:8000 …"
    (
        cd "$PROJECT_ROOT/backend"
        uv run uvicorn app:app --host 0.0.0.0 --port 8000
    ) >"$BACKEND_LOG" 2>&1 &
    echo $! > "$BACKEND_PID_FILE"
}

start_frontend_nginx() {
    log "Starting nginx …"
    if [ "$HAS_SYSTEMD" -eq 1 ]; then
        run_as_root systemctl start nginx || true
    else
        nginx || log "nginx may already be running"
    fi
}

stop_services() {
    for f in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"; do
        if [[ -f "$f" ]]; then
            pid="$(cat "$f" 2>/dev/null || true)"
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                pkill -P "$pid" 2>/dev/null || true
                log "  stopped pid $pid ($(basename "$f"))"
            fi
            rm -f "$f"
        fi
    done
}

# ---------- 1️⃣ Detect OS ----------
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
Linux*) OS="linux" ;;
Darwin*) OS="macos" ;;
*) OS="unknown" ;;
esac

if [ "$OS" = "unknown" ]; then
    error "Unsupported operating system: $OS_TYPE. This installer only works on Linux."
fi

if [ "$OS" = "macos" ]; then
    error "macOS is not supported for production server deployment. Use Linux."
fi

# ---------- 2️⃣ Detect package manager ----------
if command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt-get"
    PKG_UPDATE="update -y"
    PKG_INSTALL="install -y"
elif command -v apt >/dev/null 2>&1; then
    PKG_MGR="apt"
    PKG_UPDATE="update -y"
    PKG_INSTALL="install -y"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
    PKG_UPDATE="makecache -y"
    PKG_INSTALL="install -y --allowerasing"
elif command -v yum >/dev/null 2>&1; then
    PKG_MGR="yum"
    PKG_UPDATE="check-update -y"
    PKG_INSTALL="install -y --allowerasing"
elif command -v apk >/dev/null 2>&1; then
    PKG_MGR="apk"
    PKG_UPDATE="update"
    PKG_INSTALL="add --no-interactive"
else
    error "No supported package manager found (apt, dnf, yum, apk)."
fi

log "Using $PKG_MGR to install system packages"
run_as_root "$PKG_MGR" $PKG_UPDATE

# ---------- 3️⃣ Determine package names ----------
if [ "$PKG_MGR" = "apk" ]; then
    PKGS="git curl wget build-base nginx ca-certificates \
          python3 py3-pip py3-virtualenv \
          openssl-dev libffi-dev \
          python3-dev gnupg lsb-release lsof rsync \
          certbot certbot-nginx"
else
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        IS_AMZN=0
        case "$ID" in
        amzn | alinux) IS_AMZN=1 ;;
        esac
    else
        IS_AMZN=0
    fi

    if [ "$IS_AMZN" -eq 1 ] && [ "$PKG_MGR" = "dnf" ]; then
        PKGS="git curl wget gcc gcc-c++ make \
              python3 python3-pip python3-virtualenv \
              python3-devel \
              nginx ca-certificates \
              openssl-devel libffi-devel \
              gnupg lsb-release lsof rsync \
              certbot python3-certbot-nginx"
    else
        PKGS="git curl wget build-essential \
              python3 python3-venv python3-pip \
              python3-dev \
              nginx ca-certificates \
              libssl-dev libffi-dev \
              gnupg lsb-release lsof rsync \
              certbot python3-certbot-nginx"
        if [ "$PKG_MGR" = "apt-get" ] || [ "$PKG_MGR" = "apt" ]; then
            PKGS="$PKGS apt-transport-https"
        fi
    fi
fi

run_as_root "$PKG_MGR" $PKG_INSTALL $PKGS || error "Failed to install system packages"

# Ensure CA certificates are up to date
if command -v update-ca-certificates >/dev/null 2>&1; then
    run_as_root update-ca-certificates || true
fi

# ---------- 4️⃣ Install uv ----------
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv…"
    tmp_uv=$(mktemp)
    curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh -o "$tmp_uv"
    sh "$tmp_uv"
    rm -f "$tmp_uv"
    export PATH="$HOME/.local/bin:$PATH"
else
    log "uv already present (v$(uv --version 2>/dev/null || echo 'unknown'))"
fi

# ---------- 5️⃣ Install Node.js LTS ----------
if ! command -v node >/dev/null 2>&1; then
    log "Node.js not found – installing…"
    tmp_node=$(mktemp)
    curl -fsSL https://deb.nodesource.com/setup_lts.x -o "$tmp_node"
    bash "$tmp_node"
    rm -f "$tmp_node"
    run_as_root "$PKG_MGR" $PKG_INSTALL nodejs || run_as_root "$PKG_MGR" $PKG_INSTALL node
else
    log "Node.js already present (v$(node -v))"
fi

# ---------- 6️⃣ Determine PROJECT_ROOT ----------
if [[ $# -eq 0 ]]; then
    PROJECT_ROOT="$(pwd)"
elif [[ "$1" =~ ^https?:// ]]; then
    GIT_URL="$1"
    PROJECT_ROOT="/opt/aiscope"
    log "Cloning $GIT_URL into $PROJECT_ROOT …"
    mkdir -p "$PROJECT_ROOT"
    git clone "$GIT_URL" "$PROJECT_ROOT"
else
    if command -v realpath >/dev/null 2>&1; then
        PROJECT_ROOT="$(realpath "$1")"
    else
        PROJECT_ROOT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    fi
fi

PROJECT_NAME="$(basename "$PROJECT_ROOT")"
log "Project root resolved to: $PROJECT_ROOT (name: $PROJECT_NAME)"

# ---------- 7️⃣ Verify expected sub-folders ----------
if [[ ! -d "$PROJECT_ROOT/backend" ]] || [[ ! -d "$PROJECT_ROOT/frontend" ]]; then
    error "Directory $PROJECT_ROOT does NOT contain 'backend' and 'frontend' sub-folders."
fi

# ---------- 8️⃣ Backend – install deps ----------
log "Installing backend Python dependencies (uv sync)…"
cd "$PROJECT_ROOT/backend"
uv sync

# ---------- 9️⃣ Frontend – install deps & build ----------
log "Installing frontend npm dependencies…"
cd "$PROJECT_ROOT/frontend"
npm install || error "Failed to install frontend dependencies"
log "Building React front‑end…"
npm run build || error "Frontend build failed"

# ---------- 🔟 Deploy static assets ----------
if [[ ! -d "$PROJECT_ROOT/frontend/dist" ]]; then
    error "Frontend build output not found at $PROJECT_ROOT/frontend/dist — 'npm run build' must produce it."
fi

WWW_ROOT="/var/www/$PROJECT_NAME"
log "Creating web root at $WWW_ROOT/frontend …"
run_as_root mkdir -p "$WWW_ROOT/frontend"
rsync -a --delete "$PROJECT_ROOT/frontend/dist/" "$WWW_ROOT/frontend/"

# ---------- 1️⃣1️⃣ Create systemd service (if systemd available) ----------
if [ "$HAS_SYSTEMD" -eq 1 ]; then
    SERVICE_FILE="/etc/systemd/system/aiscope.service"
    log "Creating systemd service at $SERVICE_FILE …"
    UV_PATH="$(command -v uv)"
    UV_DIR="$(dirname "$UV_PATH")"

    cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=AI Learning Platform backend (FastAPI)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_ROOT/backend
ExecStart=$UV_PATH run uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=NOTES_ROOT=$PROJECT_ROOT
Environment=PATH=$UV_DIR:/usr/local/bin:/usr/bin:/bin

# Optional resource limits – uncomment to enable
# MemoryLimit=200M
# CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

    run_as_root systemctl daemon-reload
    run_as_root systemctl enable aiscope.service
    run_as_root systemctl restart aiscope.service
else
    log "Skipping systemd service creation (no systemd)"
    run_as_root mkdir -p "$PID_DIR"
fi

# ---------- Start backend ----------
if [ "$HAS_SYSTEMD" -eq 1 ]; then
    run_as_root systemctl restart aiscope.service
else
    start_backend
fi

# Wait for backend to be healthy
log "Waiting for backend to become healthy…"
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        log "  Backend is up"
        break
    fi
    if [ "$HAS_SYSTEMD" -eq 1 ]; then
        if ! run_as_root systemctl is-active --quiet aiscope.service 2>/dev/null; then
            error "Backend failed to start — check journalctl -u aiscope.service"
        fi
    else
        if [[ -f "$BACKEND_PID_FILE" ]] && ! kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
            error "Backend failed to start — check $BACKEND_LOG"
        fi
    fi
    sleep 1
done

# Ensure nginx config directories exist
run_as_root mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled /etc/nginx/conf.d

# Determine which nginx config directory to use
if [ -d /etc/nginx/sites-enabled ] && grep -q "sites-enabled" /etc/nginx/nginx.conf 2>/dev/null; then
    NGINX_CONF="/etc/nginx/sites-available/aiscope.conf"
    NGINX_LINK="/etc/nginx/sites-enabled/aiscope.conf"
elif [ -d /etc/nginx/conf.d ]; then
    NGINX_CONF="/etc/nginx/conf.d/aiscope.conf"
    NGINX_LINK=""
else
    NGINX_CONF="/etc/nginx/sites-available/aiscope.conf"
    NGINX_LINK="/etc/nginx/sites-enabled/aiscope.conf"
fi

# ---------- 1️⃣2️⃣ Generate Nginx site config ----------
log "Generating Nginx config at $NGINX_CONF …"

# Remove default site if it exists (prevents conflicts)
if [ -f /etc/nginx/sites-enabled/default ]; then
    run_as_root rm -f /etc/nginx/sites-enabled/default
fi

cat >"$NGINX_CONF" <<EOF
server {
    listen 80;
    server_name _;

    # Front‑end static files
    root $WWW_ROOT/frontend;
    index index.html;

    # SPA fallback – any non‑file request goes to index.html
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # Backend API reverse proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
    }

    # Backend static assets (paper images, etc.)
    location /assets/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        add_header Cache-Control "public, immutable";
        expires 7d;
    }

    # Cache static build assets (hash‑named files)
    location ~* \.(js|css|png|jpg|svg|ico|woff2?)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Create symlink if needed (sites-available → sites-enabled)
if [ -n "$NGINX_LINK" ]; then
    run_as_root ln -sf "$NGINX_CONF" "$NGINX_LINK"
fi

# Test and reload/start nginx
nginx -t || error "Nginx config test failed"
if [ "$HAS_SYSTEMD" -eq 1 ]; then
    run_as_root systemctl reload nginx || run_as_root systemctl start nginx
else
    start_frontend_nginx
fi
log "Nginx configuration installed and reloaded."

# ---------- 1️⃣3️⃣ (Optional) Obtain HTTPS certificate ----------
if [[ -n "${DOMAIN:-}" ]]; then
    log "Attempting to obtain Let's Encrypt certificate for $DOMAIN …"
    certbot --nginx -d "$DOMAIN" || log "Certbot failed – you can get a cert later"
    log "TLS certificate installation attempted."
else
    log "DOMAIN variable not set – skipping TLS. Set it later with: export DOMAIN=your.domain.com && $0"
fi

# ---------- 🎉 Finished ----------
echo
log "===================================================="
log " AIScope deployment complete!"
log "   Front‑end URL:  http://$(hostname -I 2>/dev/null | awk '{print $1}')"
log "   API endpoint:   http://$(hostname -I 2>/dev/null | awk '{print $1}'):8000/api/health"
if [ "$HAS_SYSTEMD" -eq 1 ]; then
    log "   Systemd service: aiscope.service (status: $(run_as_root systemctl is-active aiscope.service))"
    log "   Stop backend:   sudo systemctl stop aiscope.service"
else
    log "   Backend PID:    $(cat "$BACKEND_PID_FILE" 2>/dev/null || echo 'N/A')"
    log "   Backend log:    $BACKEND_LOG"
    log "   Stop backend:   kill \$(cat $BACKEND_PID_FILE)"
fi
log "   Nginx site:     $NGINX_CONF"
log "   Web root:       $WWW_ROOT/frontend"
log "===================================================="