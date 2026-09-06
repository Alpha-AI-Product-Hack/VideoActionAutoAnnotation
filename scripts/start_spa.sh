#!/usr/bin/env bash
# Start backend + frontend together for the merged SPA.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8443}"
export PIPELINE_ENCODER="${PIPELINE_ENCODER:-auto}"
export PIPELINE_SOURCES="${PIPELINE_SOURCES:-motion}"
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-$ROOT/src/proposals_pipeline/configs/mvp-motion.json}"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required (Node 22+). Install Node, then retry ./scripts/start_spa.sh" >&2
  exit 1
fi

backend_pid=""
frontend_pid=""

cleanup() {
  if [ -n "$frontend_pid" ] && kill -0 "$frontend_pid" 2>/dev/null; then
    kill "$frontend_pid" 2>/dev/null || true
  fi
  if [ -n "$backend_pid" ] && kill -0 "$backend_pid" 2>/dev/null; then
    kill "$backend_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

HOST=127.0.0.1 PORT="$BACKEND_PORT" "$ROOT/scripts/start_backend.sh" &
backend_pid=$!

echo "[spa] waiting for backend on :${BACKEND_PORT}"
ready=0
for _ in $(seq 1 40); do
  if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    echo "backend exited before becoming ready" >&2
    exit 1
  fi
  sleep 0.5
done
if [ "$ready" -ne 1 ]; then
  echo "backend did not respond on :${BACKEND_PORT}" >&2
  exit 1
fi

PORT="$FRONTEND_PORT" API_PROXY_TARGET="http://127.0.0.1:${BACKEND_PORT}" "$ROOT/scripts/start_frontend.sh" &
frontend_pid=$!

echo
echo "SPA:      http://127.0.0.1:${FRONTEND_PORT}"
echo "API:      http://127.0.0.1:${BACKEND_PORT}/api/health"
echo "Encoder:  ${PIPELINE_ENCODER}   sources: ${PIPELINE_SOURCES}"
echo "Ctrl+C stops both processes."
echo

wait "$backend_pid" "$frontend_pid"
