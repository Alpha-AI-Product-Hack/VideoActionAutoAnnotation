from __future__ import annotations

import hashlib

import numpy as np


class StubEncoder:
    """Deterministic fake embeddings. No checkpoints, no training."""

    encoder_id = "stub-d64"
    dim = 64
    num_frames = 8

    def encode_clips(self, frames: np.ndarray) -> np.ndarray:
        if frames.size == 0 or frames.shape[0] == 0:
            raise ValueError("Empty clip batch is not allowed")
        batch = []
        for item in frames:
            digest = hashlib.sha256(np.ascontiguousarray(item).tobytes()).digest()
            batch.append(_digest_to_vec(digest, self.dim))
        return np.stack(batch, axis=0).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Empty text batch is not allowed")
        rows = [_digest_to_vec(hashlib.sha256(t.encode("utf-8")).digest(), self.dim) for t in texts]
        return np.stack(rows, axis=0).astype(np.float32)


def _digest_to_vec(digest: bytes, dim: int) -> np.ndarray:
    raw = np.frombuffer((digest * ((dim // 32) + 1))[:dim], dtype=np.uint8).astype(np.float32)
    raw = raw - raw.mean()
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm
    return raw
