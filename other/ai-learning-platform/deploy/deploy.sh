#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
#  One‑click AIScope installer (FastAPI + React/Vite)
# --------------------------------------------------------------
# What the script does:
#   1️⃣ Installs required OS packages (git, curl, build‑essential, nginx …)
#   2️⃣ Installs the Python tool "uv"
#   3️⃣ Installs Node.js LTS (includes npm)
#   4️⃣ If a Git URL is supplied, clones the repo; otherwise works on the
#      current directory (or on a path you pass as the first argument).
#   5️⃣ Installs backend deps (uv sync) and frontend deps (npm ci)
#   6️⃣ Builds the React app (npm run build)
#   7️⃣ Copies the built assets to /var/www/<project>/frontend
#   8️⃣ Generates a systemd service (aiscope.service) and starts it
#   9️⃣ Generates an Nginx site config (using the embedded template)
#  10️⃣ (Optional) obtains a free TLS cert via certbot if a DOMAIN env‑var is set.
#
# Usage (run as root or with sudo):
#   ./deploy.sh                # use the current directory as project root
#   ./deploy.sh /path/to/project
#   ./deploy.sh https://github.com/your/repo.git   # clone then install
#   export DOMAIN=your.domain.com && ./deploy.sh   # also request HTTPS
# --------------------------------------------------------------

log() { echo -e "\033[1;36m[+] $*\033[0m"; }
error() {
    echo -e "\033[1;31m[-] $*\033[0m" >&2
    exit 1
}

# ---------- 1️⃣ Install OS packages ----------
log "Detecting operating system and installing required packages…"

# Helper: run a command with sudo if we are not root
run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

# ------------------------------------------------------------
# 1️⃣ Detect OS (Linux, macOS, or unsupported)
# ------------------------------------------------------------
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
Linux*) OS="linux" ;;
Darwin*) OS="macos" ;;
*) OS="unknown" ;;
esac

if [ "$OS" = "unknown" ]; then
    error "Unsupported operating system: $OS_TYPE. This installer only works on Linux or macOS."
fi

# ------------------------------------------------------------
# 2️⃣ Detect package manager for the current OS
# ------------------------------------------------------------
if [ "$OS" = "linux" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        PKG_MGR="apt-get"
        UPDATE_CMD="run_as_root $PKG_MGR update -y"
        INSTALL_CMD="run_as_root $PKG_MGR install -y"
    elif command -v apt >/dev/null 2>&1; then
        PKG_MGR="apt"
        UPDATE_CMD="run_as_root $PKG_MGR update -y"
        INSTALL_CMD="run_as_root $PKG_MGR install -y"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MGR="dnf"
        UPDATE_CMD="run_as_root $PKG_MGR check-update -y"
        INSTALL_CMD="run_as_root $PKG_MGR install -y"
    elif command -v yum >/dev/null 2>&1; then
        PKG_MGR="yum"
        UPDATE_CMD="run_as_root $PKG_MGR check-update -y"
        INSTALL_CMD="run_as_root $PKG_MGR install -y"
    elif command -v apk >/dev/null 2>&1; then
        PKG_MGR="apk"
        UPDATE_CMD="run_as_root $PKG_MGR update"
        INSTALL_CMD="run_as_root $PKG_MGR add"
    else
        error "No supported package manager found on this Linux system (apt, dnf, yum, apk). Install required packages manually and re‑run the script."
    fi
elif [ "$OS" = "macos" ]; then
    if command -v brew >/dev/null 2>&1; then
        PKG_MGR="brew"
        UPDATE_CMD="run_as_root $PKG_MGR update"
        INSTALL_CMD="run_as_root $PKG_MGR install"
    else
        error "Homebrew not found on macOS. Install Homebrew first: https://brew.sh/"
    fi
fi

log "Using $PKG_MGR to install system packages"
$UPDATE_CMD

# Packages we need on every platform
if [ "$PKG_MGR" = "apk" ]; then
    PKGS="git curl wget build-base nginx ca-certificates \
          python3 py3-pip py3-virtualenv \
          openssl-dev libffi-dev \
          certbot certbot-nginx"
else
    PKGS="git curl wget build-essential \
          python3 python3-venv python3-pip \
          nginx ca-certificates \
          libssl-dev libffi-dev \
          certbot python3-certbot-nginx"
fi

$INSTALL_CMD $PKGS || error "Failed to install required system packages with $PKG_MGR"

# ---------- 2️⃣ Install uv (Python package manager) ----------
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv…"
    tmp_uv=$(mktemp)
    log "Downloading uv installer…"
    curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh -o "$tmp_uv"
    sh "$tmp_uv"
    rm -f "$tmp_uv"
    export PATH="$HOME/.local/bin:$PATH"
else
    log "uv already present"
fi

# ---------- 3️⃣ Install Node.js LTS ----------
if ! command -v node >/dev/null 2>&1; then
    log "Node.js not found – installing…"
    if [ "$OS" = "linux" ]; then
        tmp_node=$(mktemp)
        log "Downloading Node.js setup script…"
        curl -fsSL https://deb.nodesource.com/setup_lts.x -o "$tmp_node"
        bash "$tmp_node"
        rm -f "$tmp_node"
        $INSTALL_CMD nodejs
    elif [ "$OS" = "macos" ]; then
        # Homebrew already installed the 'node' package in the generic PKGS list
        error "Node.js not found on macOS and automatic installation is not supported – please install it via Homebrew (brew install node) and re‑run the script."
    else
        error "Unsupported OS for automatic Node.js installation. Install Node.js manually and re‑run the script."
    fi
else
    log "Node.js already present (v$(node -v))"
fi

# ---------- 4️⃣ Determine PROJECT_ROOT ----------
if [[ $# -eq 0 ]]; then
    PROJECT_ROOT="$(pwd)"
elif [[ "$1" =~ ^https?:// ]]; then
    GIT_URL="$1"
    PROJECT_ROOT="/opt/aiscope"
    log "Cloning $GIT_URL into $PROJECT_ROOT …"
    mkdir -p "$PROJECT_ROOT"
    git clone "$GIT_URL" "$PROJECT_ROOT"
else
    PROJECT_ROOT="$(realpath "$1")"
fi

log "Project root resolved to: $PROJECT_ROOT"

# ---------- 5️⃣ Verify expected sub‑folders ----------
if [[ ! -d "$PROJECT_ROOT/backend" ]] || [[ ! -d "$PROJECT_ROOT/frontend" ]]; then
    error "Directory $PROJECT_ROOT does NOT contain expected 'backend' and 'frontend' sub‑folders."
fi

# ---------- 6️⃣ Backend – install deps ----------
log "Installing backend Python dependencies (uv sync)…"
cd "$PROJECT_ROOT/backend"
uv sync

# ---------- 7️⃣ Frontend – install deps & build ----------
log "Installing frontend npm dependencies…"
cd "$PROJECT_ROOT/frontend"
npm ci
log "Building React front‑end…"
npm run build

# ---------- 8️⃣ Deploy static assets ----------
WWW_ROOT="/var/www/$(basename "$PROJECT_ROOT")"
log "Copying built assets to $WWW_ROOT/frontend …"
mkdir -p "$WWW_ROOT/frontend"
rsync -a --delete "$PROJECT_ROOT/frontend/dist/" "$WWW_ROOT/frontend/"

# ---------- 9️⃣ Create systemd service ----------
SERVICE_FILE="/etc/systemd/system/aiscope.service"
log "Creating systemd service at $SERVICE_FILE …"
cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=AI Learning Platform backend (FastAPI)
After=network.target

[Service]
WorkingDirectory=$PROJECT_ROOT/backend
ExecStart=$(which uv) run uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
# Optional limits – uncomment to enable
# MemoryLimit=200M
# CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable aiscope.service
systemctl restart aiscope.service

# ---------- 🔟 Generate Nginx site config ----------
NGINX_CONF="/etc/nginx/sites-available/aiscope.conf"
log "Generating Nginx config at $NGINX_CONF …"
cat >"$NGINX_CONF" <<'EOT'
server {
    listen 80;
    server_name _;                     # replace with your domain if you have one

    # ---------- Front‑end static files ----------
    root /var/www/<PROJECT_ROOT>/frontend;
    index index.html;

    # SPA fallback – any non‑file request goes to index.html
    location / {
        try_files $uri $uri/ /index.html;
    }

    # ---------- Backend API ----------
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    # ---------- Assets served by FastAPI ----------
    location /assets/ {
        proxy_pass http://127.0.0.1:8000;
        add_header Cache-Control "public, immutable";
        expires 7d;
    }

    # Cache static build assets (hash‑named files)
    location ~* \.(js|css|png|jpg|svg|ico|woff2?)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOT
# Replace placeholder with the absolute project path
sed -i "s|<PROJECT_ROOT>|$PROJECT_ROOT|g" "$NGINX_CONF"

ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/aiscope.conf
nginx -t && systemctl reload nginx
log "Nginx configuration installed and reloaded."

# ---------- 1️⃣1️⃣ (Optional) Obtain HTTPS certificate ----------
if [[ -n "${DOMAIN:-}" ]]; then
    log "Attempting to obtain Let\'s Encrypt certificate for $DOMAIN …"
    certbot --nginx -d "$DOMAIN"
    log "TLS certificate installed."
else
    log "DOMAIN variable not set – skipping TLS. You can get a cert later with:\n    export DOMAIN=your.domain.com && $0"
fi

# ---------- 🎉 Finished ----------
log "===================================================="
log "✅  AIScope deployment complete!"
log "   Front‑end URL:  http://<your‑host-or‑IP> (or https://$DOMAIN if set)"
log "   API proxy:      http://<your‑host-or‑IP>/api"
log "   Systemd service: aiscope.service (status: $(systemctl is-active aiscope.service))"
log "   Nginx site:     /etc/nginx/sites-available/aiscope.conf"
log "===================================================="
