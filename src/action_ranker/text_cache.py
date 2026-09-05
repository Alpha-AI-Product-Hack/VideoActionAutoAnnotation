from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from action_ranker.taxonomies import REPO_ROOT

CACHE_ROOT = REPO_ROOT / "artifacts" / "text_cache"


def cache_paths(encoder_id: str, dictionary_id: str, prompt_id: str) -> tuple[Path, Path]:
    folder = CACHE_ROOT / _safe(encoder_id)
    stem = f"{_safe(dictionary_id)}__{_safe(prompt_id)}"
    return folder / f"{stem}.npz", folder / f"{stem}.json"


def load_or_build_text_cache(
    encoder_id: str,
    dictionary_id: str,
    prompt_id: str,
    labels: list[str],
    encode_fn,
) -> np.ndarray:
    npz_path, sidecar_path = cache_paths(encoder_id, dictionary_id, prompt_id)
    if npz_path.is_file() and sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        cached_labels = list(sidecar.get("labels") or [])
        if (
            sidecar.get("encoder_id") == encoder_id
            and sidecar.get("dictionary_id") == dictionary_id
            and sidecar.get("prompt_id") == prompt_id
            and sidecar.get("num_labels") == len(labels)
            and cached_labels == labels
        ):
            payload = np.load(npz_path, allow_pickle=False)
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if embeddings.shape[0] == len(labels) and sidecar.get("dim") == embeddings.shape[1]:
                return embeddings
    embeddings = np.asarray(encode_fn(labels), dtype=np.float32)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, embeddings=embeddings)
    sidecar = {
        "encoder_id": encoder_id,
        "dictionary_id": dictionary_id,
        "prompt_id": prompt_id,
        "num_labels": len(labels),
        "dim": int(embeddings.shape[1]),
        "labels": labels,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return embeddings


def _safe(token: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in token)
