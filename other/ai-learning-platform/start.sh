#!/usr/bin/env bash
#
# AILearn — one-command launcher
# Starts the FastAPI backend (uv) and the Vite frontend (npm) together.
#
# Usage:
#   ./start.sh          # start both services
#   ./start.sh --stop   # stop running services
#
set -euo pipefail

PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PLATFORM_DIR/backend"
FRONTEND_DIR="$PLATFORM_DIR/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_URL="http://127.0.0.1:$BACKEND_PORT"
FRONTEND_URL="http://localhost:$FRONTEND_PORT"
PID_DIR="$PLATFORM_DIR/.run"
BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"
BACKEND_LOG="$PLATFORM_DIR/backend.log"
FRONTEND_LOG="$PLATFORM_DIR/frontend.log"

mkdir -p "$PID_DIR"

c_endpoint() { printf '\033[36m%s\033[0m\n' "$1"; }
c_ok()       { printf '\033[32m%s\033[0m\n' "$1"; }
c_warn()     { printf '\033[33m%s\033[0m\n' "$1"; }
c_err()      { printf '\033[31m%s\033[0m\n' "$1"; }

stop_services() {
  c_endpoint "Stopping AILearn services..."
  for f in "$FRONTEND_PID_FILE" "$BACKEND_PID_FILE"; do
    if [[ -f "$f" ]]; then
      pid="$(cat "$f" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        # also kill any child processes (uvicorn/npm spawn children)
        pkill -P "$pid" 2>/dev/null || true
        c_ok "  stopped pid $pid ($(basename "$f"))"
      fi
      rm -f "$f"
    fi
  done
  # best-effort cleanup of anything still bound to our ports
  lsof -ti tcp:"$BACKEND_PORT"  2>/dev/null | xargs kill -9 2>/dev/null || true
  lsof -ti tcp:"$FRONTEND_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
}

if [[ "${1:-}" == "--stop" ]]; then
  stop_services
  exit 0
fi

trap 'c_warn "\nInterrupted — shutting down..."; stop_services; exit 130' INT TERM

# --- preflight --------------------------------------------------------------
command -v uv   >/dev/null || { c_err "uv is not installed (https://docs.astral.sh/uv/)"; exit 1; }
command -v npm  >/dev/null || { c_err "npm/node is not installed"; exit 1; }

if ! command -v lsof >/dev/null; then
  c_warn "lsof not found — port checks / --stop may be limited"
fi

if [[ -f "$BACKEND_PID_FILE" ]] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  c_warn "Backend already running (pid $(cat "$BACKEND_PID_FILE")) — restarting"
  stop_services
fi

c_endpoint "Installing backend dependencies (uv sync)..."
( cd "$BACKEND_DIR" && uv sync )

c_endpoint "Installing frontend dependencies (npm install)..."
if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  ( cd "$FRONTEND_DIR" && npm install )
else
  c_ok "  node_modules present, skipping npm install"
fi

# --- start backend ----------------------------------------------------------
c_endpoint "Starting backend on $BACKEND_URL..."
(
  cd "$BACKEND_DIR"
  uv run uvicorn app:app --host 127.0.0.1 --port "$BACKEND_PORT"
) >"$BACKEND_LOG" 2>&1 &
echo $! > "$BACKEND_PID_FILE"

# wait for backend health
for i in $(seq 1 30); do
  if curl -sf "$BACKEND_URL/api/health" >/dev/null 2>&1; then
    c_ok "  backend is up"
    break
  fi
  if ! kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
    c_err "Backend failed to start — last log lines:"
    tail -n 20 "$BACKEND_LOG" >&2 || true
    exit 1
  fi
  sleep 1
  [[ $i -eq 30 ]] && { c_err "Backend did not become healthy in time"; tail -n 20 "$BACKEND_LOG" >&2; exit 1; }
done

# --- start frontend ---------------------------------------------------------
c_endpoint "Starting frontend on $FRONTEND_URL..."
(
  cd "$FRONTEND_DIR"
  npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"
) >"$FRONTEND_LOG" 2>&1 &
echo $! > "$FRONTEND_PID_FILE"

for i in $(seq 1 30); do
  if curl -sf "$FRONTEND_URL" >/dev/null 2>&1; then
    c_ok "  frontend is up"
    break
  fi
  if ! kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
    c_err "Frontend failed to start — last log lines:"
    tail -n 20 "$FRONTEND_LOG" >&2 || true
    exit 1
  fi
  sleep 1
  [[ $i -eq 30 ]] && { c_err "Frontend did not become ready in time"; tail -n 20 "$FRONTEND_LOG" >&2; exit 1; }
done

echo
c_ok "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
c_ok " AILearn is running"
c_ok "   Frontend : $FRONTEND_URL"
c_ok "   Backend  : $BACKEND_URL/api/health"
c_ok "   Logs     : $BACKEND_LOG | $FRONTEND_LOG"
c_ok "   Stop     : $0 --stop   (or press Ctrl-C)"
c_ok "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# open in browser on macOS
if command -v open >/dev/null 2>&1; then
  open "$FRONTEND_URL"
fi

# foreground-wait: keep script alive so Ctrl-C stops both
wait
