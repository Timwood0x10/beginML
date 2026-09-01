#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
#  AIScope — Vercel deployment (front-end SPA)
# --------------------------------------------------------------
# WHY: the Vercel dashboard complained with
#   "No Output Directory named \"public\" found after the Build completed."
#   That happens because a dashboard (Git-integration) build uses a manually
#   configured Root Directory / Output Directory, which pointed at `public`.
#   This script deploys from the PROJECT ROOT with the Vercel CLI, so the
#   repository's vercel.json (buildCommand + outputDirectory) is ALWAYS used
#   and the stale dashboard setting is bypassed.
#
# Usage:
#   ./deploy/vercel.sh                 # deploy to the linked Vercel project
#   ./deploy/vercel.sh --prod          # production (same as default here)
#   ./deploy/vercel.sh --dry-run       # only validate env + build, no upload
#   ./deploy/vercel.sh --link          # re-link to a different Vercel project
#
# Prereq: `vercel` CLI. Install once with:
#   npm i -g vercel
#   vercel login
# --------------------------------------------------------------

log()   { echo -e "\033[1;36m[+]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[!]\033[0m $*"; }
error() { echo -e "\033[1;31m[-]\033[0m $*" >&2; exit 1; }

# ---------- Resolve the PROJECT ROOT (this file lives in deploy/) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

log "Project root: $PROJECT_ROOT"

# ---------- Flags ----------
LINK=0
DRY_RUN=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --link)     LINK=1 ;;
        --dry-run)  DRY_RUN=1 ;;
        *)          ARGS+=("$arg") ;;
    esac
done

# ---------- 1. Node availability ----------
command -v node >/dev/null 2>&1 || error "Node.js not found — install it first (https://nodejs.org)"
log "Node $(node -v)"

# ---------- 2. Vercel CLI ----------
VER_CMD=""
if command -v vercel >/dev/null 2>&1; then
    VER_CMD="$(command -v vercel)"
else
    warn "vercel CLI not found. Trying npx (this may download it)…"
    if command -v npx >/dev/null 2>&1; then
        VER_CMD="npx vercel"
    else
        error "Neither 'vercel' nor 'npx' is available. Install vercel:  npm i -g vercel"
    fi
fi
log "Vercel CLI: $VER_CMD"

# ---------- 3. Expect a vercel.json at the project root ----------
if [[ ! -f "$PROJECT_ROOT/vercel.json" ]]; then
    error "No vercel.json at the project root. Deploy from the repo root."
fi
if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not found (only used to pretty-print vercel.json, safe to ignore)."
else
    echo "  -> $PROJECT_ROOT/vercel.json:"; python3 -m json.tool "$PROJECT_ROOT/vercel.json" >/dev/null 2>&1 \
        && echo "     (valid JSON)" || warn "     vercel.json is not valid JSON!"
fi

# ---------- 4. Local sanity: build must produce frontend/dist/index.html ----------
if [[ ! -d "$PROJECT_ROOT/frontend" ]]; then
    error "Expected a 'frontend' folder next to vercel.json."
fi
if [[ ! -f "$PROJECT_ROOT/frontend/dist/index.html" ]]; then
    warn "frontend/dist/index.html not found. Building locally first to be safe…"
    (cd "$PROJECT_ROOT/frontend" && npm install && npm run build) \
        || error "Local build failed — fix the build before deploying."
    log "Local build OK -> frontend/dist/index.html"
else
    log "frontend/dist/index.html already present (local build step skipped)."
fi

if [[ -f "$PROJECT_ROOT/frontend/dist/index.html" ]]; then
    log "Output directory (built): frontend/dist"
fi

# ---------- 5. Dry-run -------------
if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Dry-run complete — environment & build look good. Nothing uploaded."
    exit 0
fi

# ---------- 6. Link (optional) ----------
if [[ "$LINK" -eq 1 ]]; then
    log "Linking to a Vercel project…"
    $VER_CMD link --yes || warn "link cancelled/failed — you may still deploy to an existing linked project."
fi

# ---------- 7. Deploy ----------
log "Deploying to Vercel (production)…"
# --yes            : accept defaults / skip prompts
# --prod           : deploy to production
# --prebuilt?      : NO — let Verzel build remotely using vercel.json.
#   (Uploading prebuilt files would require pointing Vercel at frontend/dist,
#    which reintroduces the very output-directory mismatch we are avoiding.)
$VER_CMD deploy --prod --yes 2>&1 | tee /tmp/aiscope-vercel-deploy.log

# ---------- 8. Result ----------
if [[ -f /tmp/aiscope-vercel-deploy.log ]] && grep -q "Production:" /tmp/aiscope-vercel-deploy.log; then
    URL="$(grep -oE 'https://[^ /]+' /tmp/aiscope-vercel-deploy.log | grep -v 'vercel.com/vercel' | tail -1 || true)"
    log "Deployed! Open: ${URL:-the URL printed above}"
else
    warn "Deploy finished — check the log above (/tmp/aiscope-vercel-deploy.log)."
fi

log "Tip: vercel.json rewrites route /api/* to your backend — set the real URL in vercel.json."
