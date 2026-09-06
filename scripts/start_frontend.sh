#!/usr/bin/env bash
# Start the React SPA. Proxies /api to the backend (default 127.0.0.1:8000).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/frontend"

export PATH="${HOME}/.local/bin:${PATH}"
export PORT="${PORT:-8443}"
export API_PROXY_TARGET="${API_PROXY_TARGET:-http://127.0.0.1:8000}"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required (Node 22+). Install it, then retry:" >&2
  echo "  # this repo also looks in \$HOME/.local/bin" >&2
  echo "  # Node 22 tarball: https://nodejs.org/dist/v22.19.0/" >&2
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "[frontend] installing npm dependencies"
  npm install
fi

echo "[frontend] http://127.0.0.1:${PORT}  (API proxy ${API_PROXY_TARGET})"
exec npm run dev
