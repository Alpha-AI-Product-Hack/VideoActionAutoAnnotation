#!/usr/bin/env bash
# Start the FastAPI backend that runs the merged annotation pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${ROOT}/src/proposals_pipeline${PYTHONPATH:+:$PYTHONPATH}"
export PIPELINE_ENCODER="${PIPELINE_ENCODER:-auto}"
export PIPELINE_SOURCES="${PIPELINE_SOURCES:-motion}"
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-$ROOT/src/proposals_pipeline/configs/mvp-motion.json}"
export PIPELINE_RUNTIME_DIR="${PIPELINE_RUNTIME_DIR:-$ROOT/data/runtime}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

"$PY" - <<'PY'
import importlib.util, sys
missing = [m for m in ("fastapi", "uvicorn") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(
        "Missing %s. From the repo root run:\n  uv sync --extra api --extra video --extra test\n  # or: uv pip install -e '.[api,video]'"
        % ", ".join(missing)
    )
PY

if command -v ss >/dev/null 2>&1 && ss -tln | grep -qE ":${PORT}\\s"; then
  echo "Port ${PORT} is already in use. Stop the leftover process:" >&2
  echo "  fuser -k ${PORT}/tcp" >&2
  echo "  # or: pkill -f 'uvicorn backend.main:app'" >&2
  exit 1
fi

mkdir -p "$PIPELINE_RUNTIME_DIR"
echo "[backend] encoder=${PIPELINE_ENCODER} sources=${PIPELINE_SOURCES} http://${HOST}:${PORT}"
exec "$PY" -m uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload --reload-dir backend --reload-dir src
