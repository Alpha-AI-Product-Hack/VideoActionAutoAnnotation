"""ABD proposals on V-JEPA 2.1 embeddings, optionally fused with DDM-Net
generic-boundary scores.

Extraction encodes 1 s windows every 0.125 s in 120 s chunks (cached in the
shared `EmbeddingStore`) and scores every 0.125 s with DDM-Net. Proposing
builds the cosine-valley curve `d_z = -zscore(cos(f_t, f_{t+delta}))`,
optionally combines it with the GEBD probability `b_t` as
`max(b_t, w * d_z)`, z-scores the result and picks prominent peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from action_boundaries.boundary_detection import cosine_valley_curve, zscore
from action_boundaries.constants import DEFAULT_CHECKPOINT
from action_boundaries.embedding_store import EmbeddingKey, EmbeddingStore
from boundary_pipeline.datasets import VideoItem
from boundary_pipeline.proposals import Proposal

NAME = "abd"
STORE_DIR = Path(".cache/embedding_store")
GEBD_TAG = "ddm-net:1k66v9VuFgah3Wx6eKmWpj8_b5MXxlwt7"
CHUNK_S = 120.0
WINDOW_S = 1.0
STRIDE_S = 0.125
FRAMES_PER_WINDOW = 8
GEBD_DS_S = 0.1
GEBD_FRAMES_PER_SIDE = 5

SPACE = {
    "mode": ["abd", "gebd", "fused"],
    "gebd_weight": [0.3, 0.6, 1.0, 2.0],
    "delta": [1, 2, 3, 4],
    "sigma": [0.5, 1.0, 2.0],
    "prominence": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0],
    "min_distance_s": [0.25, 0.5, 1.0, 2.0],
}
DEFAULTS = {"mode": "fused", "gebd_weight": 0.6, "delta": 3, "sigma": 1.0, "prominence": 0.5, "min_distance_s": 0.5}


@dataclass(eq=False)
class AbdSignal:
    stride_s: float
    embeddings: np.ndarray      # (T, D), window i starts at i * stride_s
    gebd_times_s: np.ndarray    # (G,)
    gebd_b: np.ndarray          # (G,)


def chunks(item: VideoItem) -> list[tuple[float, float]]:
    n = int(np.ceil(item.duration_s / CHUNK_S))
    out = []
    for i in range(n):
        start = i * CHUNK_S
        dur = min(CHUNK_S, item.duration_s - start)
        if dur >= WINDOW_S + 4 * STRIDE_S:
            out.append((start, dur))
    return out


def _keys(item: VideoItem, start: float, dur: float) -> tuple[EmbeddingKey, EmbeddingKey]:
    vkey = EmbeddingKey(str(item.video_path), DEFAULT_CHECKPOINT, start, dur, WINDOW_S, STRIDE_S, FRAMES_PER_WINDOW)
    gkey = EmbeddingKey(str(item.video_path), GEBD_TAG, start, dur, GEBD_DS_S, STRIDE_S, 0)
    return vkey, gkey


def extract(item: VideoItem, encoder, scorer, store: EmbeddingStore | None = None, batch_size: int = 16, progress: bool = True) -> None:
    from action_boundaries.gebd import compute_gebd_scores
    from action_boundaries.vjepa_encoder import extract_window_embeddings

    store = store or EmbeddingStore(STORE_DIR)
    for start, dur in chunks(item):
        vkey, gkey = _keys(item, start, dur)
        if progress:
            print(f"  [{item.video_id}] chunk [{start:.0f}s, {start + dur:.0f}s)", flush=True)
        if store.get(vkey) is None:
            emb = extract_window_embeddings(
                str(item.video_path), encoder, start_s=start, duration_s=dur, window_s=WINDOW_S,
                stride_s=STRIDE_S, frames_per_window=FRAMES_PER_WINDOW, batch_size=batch_size, progress=False,
            )
            store.put(vkey, emb)
        if store.get(gkey) is None:
            res = compute_gebd_scores(
                str(item.video_path), scorer, start_s=start, duration_s=dur, stride_s=STRIDE_S,
                ds_seconds=GEBD_DS_S, batch_size=batch_size, progress=False,
            )
            store.put(gkey, res.b_t.reshape(-1, 1))


def _gebd_leading_drop(fps: float) -> int:
    """`compute_gebd_scores` drops the leading samples whose frame block would start before frame 0."""
    ds = max(1, round(GEBD_DS_S * fps))
    i = 0
    while round(i * STRIDE_S * fps) - GEBD_FRAMES_PER_SIDE * ds < 0:
        i += 1
    return i


def load(item: VideoItem, store: EmbeddingStore | None = None) -> AbdSignal | None:
    store = store or EmbeddingStore(STORE_DIR)
    chunk_list = chunks(item)
    embs, gtimes, gvals = [], [], []
    k0 = _gebd_leading_drop(item.fps)
    for ci, (start, dur) in enumerate(chunk_list):
        vkey, gkey = _keys(item, start, dur)
        emb, b = store.get(vkey), store.get(gkey)
        if emb is None or b is None:
            return None
        expected = int(np.floor(dur / STRIDE_S)) + 1
        if ci < len(chunk_list) - 1 and len(emb) == expected:
            emb = emb[:-1]  # window at the chunk end is repeated as the next chunk's first window
        embs.append(emb)
        gtimes.append(start + (k0 + np.arange(len(b))) * STRIDE_S)
        gvals.append(b[:, 0])
    return AbdSignal(STRIDE_S, np.concatenate(embs), np.concatenate(gtimes), np.concatenate(gvals))


@lru_cache(maxsize=512)
def _d_z(sig: AbdSignal, delta: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    return cosine_valley_curve(sig.embeddings, sig.stride_s, delta, sigma)


def curve(sig: AbdSignal, params: dict) -> tuple[np.ndarray, np.ndarray]:
    times, d_z = _d_z(sig, params["delta"], params["sigma"])
    if params["mode"] == "abd":
        return times, d_z
    b = np.interp(times, sig.gebd_times_s, sig.gebd_b) if len(sig.gebd_times_s) else np.zeros_like(times)
    fused = b if params["mode"] == "gebd" else np.maximum(b, params["gebd_weight"] * d_z)
    return times, zscore(fused)


def propose(sig: AbdSignal, params: dict) -> list[Proposal]:
    if len(sig.embeddings) <= params["delta"]:
        return []
    times, z = curve(sig, params)
    idx, props = find_peaks(z, prominence=params["prominence"], distance=max(1, round(params["min_distance_s"] / sig.stride_s)))
    return [Proposal(float(times[i]), float(p), NAME, params["mode"]) for i, p in zip(idx, props["prominences"])]
