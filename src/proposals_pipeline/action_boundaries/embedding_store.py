"""Persistent cache of per-window arrays keyed by (video fingerprint,
checkpoint, slice, windowing config): a SQLite index plus `.npy` files.
Cache keys include the video's mtime and size, so a replaced video file
invalidates its entries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingKey:
    video_path: str
    checkpoint: str
    start_s: float
    duration_s: float
    window_s: float
    stride_s: float
    frames_per_window: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key_hash TEXT PRIMARY KEY,
    video_path TEXT NOT NULL,
    video_mtime REAL NOT NULL,
    video_size INTEGER NOT NULL,
    checkpoint TEXT NOT NULL,
    start_s REAL NOT NULL,
    duration_s REAL NOT NULL,
    window_s REAL NOT NULL,
    stride_s REAL NOT NULL,
    frames_per_window INTEGER NOT NULL,
    num_windows INTEGER NOT NULL,
    hidden_size INTEGER NOT NULL,
    npy_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class EmbeddingStore:
    def __init__(self, root: str | Path = ".cache/embedding_store"):
        self.root = Path(root)
        self.array_dir = self.root / "arrays"
        self.array_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite3"
        with self._connect() as con:
            con.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _fingerprint(video_path: str) -> tuple[str, float, int]:
        resolved = str(Path(video_path).resolve())
        stat = Path(resolved).stat()
        return resolved, stat.st_mtime, stat.st_size

    def _hash(self, key: EmbeddingKey, mtime: float, size: int) -> str:
        blob = json.dumps({**asdict(key), "video_mtime": mtime, "video_size": size}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    def resolve_key(self, key: EmbeddingKey) -> tuple[EmbeddingKey, str]:
        resolved, mtime, size = self._fingerprint(key.video_path)
        key = EmbeddingKey(**{**asdict(key), "video_path": resolved})
        return key, self._hash(key, mtime, size)

    def get(self, key: EmbeddingKey) -> np.ndarray | None:
        _, key_hash = self.resolve_key(key)
        with self._connect() as con:
            row = con.execute("SELECT npy_path FROM embeddings WHERE key_hash = ?", (key_hash,)).fetchone()
        if row is None or not Path(row[0]).exists():
            return None
        return np.load(row[0])

    def put(self, key: EmbeddingKey, embeddings: np.ndarray) -> Path:
        resolved_key, key_hash = self.resolve_key(key)
        _, mtime, size = self._fingerprint(resolved_key.video_path)
        npy_path = self.array_dir / f"{key_hash}.npy"
        np.save(npy_path, embeddings.astype(np.float32, copy=False))
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key_hash, resolved_key.video_path, mtime, size, resolved_key.checkpoint,
                    resolved_key.start_s, resolved_key.duration_s, resolved_key.window_s, resolved_key.stride_s,
                    resolved_key.frames_per_window, embeddings.shape[0], embeddings.shape[1], str(npy_path),
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
        return npy_path

    def stats(self) -> dict:
        with self._connect() as con:
            n, total_windows = con.execute("SELECT COUNT(*), COALESCE(SUM(num_windows), 0) FROM embeddings").fetchone()
        total_bytes = sum(f.stat().st_size for f in self.array_dir.glob("*.npy"))
        return {"entries": n, "total_windows": total_windows, "total_mb": total_bytes / 1024**2, "root": str(self.root)}

    def vacuum(self) -> int:
        """Delete `.npy` files with no index row; returns the count removed."""
        with self._connect() as con:
            known = {row[0] for row in con.execute("SELECT npy_path FROM embeddings")}
        removed = 0
        for f in self.array_dir.glob("*.npy"):
            if str(f) not in known:
                f.unlink()
                removed += 1
        return removed
