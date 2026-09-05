from __future__ import annotations

import numpy as np

from action_ranker.encoder_protocol import FrozenEncoder


def encode_clip_batch(frames: np.ndarray, encoder: FrozenEncoder) -> np.ndarray:
    array = np.asarray(frames)
    if array.size == 0 or array.shape[0] == 0:
        raise ValueError("Empty clip batch is not allowed")
    if array.ndim != 5:
        raise ValueError("frames must be float32 [B, T, C, H, W]")
    embeddings = encoder.encode_clips(array.astype(np.float32, copy=False))
    out = np.asarray(embeddings, dtype=np.float32)
    if out.ndim != 2 or out.shape[0] != array.shape[0]:
        raise ValueError("encoder.encode_clips must return [B, D]")
    return out
