#!/usr/bin/env bash
# One-shot environment setup: virtualenv, torch/torchvision for your CUDA,
# Python requirements, model access checks and the DDM-Net checkpoint.
#
#   bash setup.sh                 # create .venv and install everything
#   bash setup.sh --check         # only verify an existing environment
#   CUDA=cu126 bash setup.sh      # other torch wheel index (default cu128)
#   VENV=/path/to/venv bash setup.sh
#
# facebook/sam3 is gated: accept its license on the Hub and run
# `huggingface-cli login` (or set HF_TOKEN) before the check passes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="${VENV:-.venv}"
CUDA="${CUDA:-cu128}"
PY="$VENV/bin/python"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

if [ "$CHECK_ONLY" = 0 ]; then
  if [ ! -x "$PY" ]; then
    echo "[setup] creating $VENV"
    python3 -m venv "$VENV"
  fi
  "$PY" -m pip install -q --upgrade pip
  if ! "$PY" -c "import torch, torchvision" >/dev/null 2>&1; then
    echo "[setup] installing torch + torchvision ($CUDA)"
    "$PY" -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
  fi
  echo "[setup] installing requirements"
  "$PY" -m pip install -q -r requirements.txt
fi

echo "[check] python: $("$PY" --version)"
"$PY" - <<'PYEOF'
import importlib.util, sys
missing = [m for m in ("torch", "torchvision", "transformers", "scipy", "numpy", "cv2", "matplotlib", "tqdm", "einops", "gdown", "pytest") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"[check] missing packages: {missing}")
import torch, transformers
print(f"[check] torch {torch.__version__}, transformers {transformers.__version__}, cuda={torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -- extraction needs a GPU"))
try:
    from transformers import Sam3Model, Sam3Processor  # noqa: F401
except ImportError as e:
    sys.exit(f"[check] transformers lacks SAM3 support ({e}); need transformers>=4.57")
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
for repo in ("facebook/sam3", "Dev-Jahn/vjepa2.1-vitl-fpc64-384"):
    try:
        HfApi().model_info(repo)
        print(f"[check] hub access ok: {repo}")
    except GatedRepoError:
        print(f"[check] {repo} is gated: accept the license on the Hub and run `huggingface-cli login`")
    except HfHubHTTPError as e:
        print(f"[check] could not reach {repo}: {e}")
PYEOF

if [ "$CHECK_ONLY" = 0 ]; then
  echo "[setup] DDM-Net checkpoint (~1.75 GB, Google Drive)"
  "$PY" -c "from action_boundaries.gebd import ensure_checkpoint; print('[setup]', ensure_checkpoint())"
fi

echo "[check] running unit tests"
"$PY" -m pytest -q tests/
echo "[setup] done. Data goes under data/assembly101 (see download_assembly101.py) and data/epic_kitchens."
