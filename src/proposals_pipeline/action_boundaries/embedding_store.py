"""Persistent cache of per-window arrays keyed by (video identity,
checkpoint, slice, windowing config): a SQLite index plus `.npy` files.

A video's identity is its last two path components plus mtime and size,
so moving the data tree or the repository keeps the cache valid while a
replaced video file invalidates its entries. Array paths are stored
relative to the store root for the same reason.
"""

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


def video_identity(video_path: str) -> str:
    return "/".join(Path(video_path).resolve().parts[-2:])


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
    def _hash(key: EmbeddingKey, mtime: float, size: int) -> str:
        payload = {**asdict(key), "video_path": video_identity(key.video_path), "video_mtime": round(mtime, 3), "video_size": size}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

    def resolve_key(self, key: EmbeddingKey) -> tuple[EmbeddingKey, str]:
        stat = Path(key.video_path).stat()
        return key, self._hash(key, stat.st_mtime, stat.st_size)

    def _array_path(self, npy_path: str) -> Path:
        p = Path(npy_path)
        return p if p.is_absolute() else self.array_dir / p.name

    def get(self, key: EmbeddingKey) -> np.ndarray | None:
        _, key_hash = self.resolve_key(key)
        with self._connect() as con:
            row = con.execute("SELECT npy_path FROM embeddings WHERE key_hash = ?", (key_hash,)).fetchone()
        if row is None:
            return None
        path = self._array_path(row[0])
        return np.load(path) if path.exists() else None

    def put(self, key: EmbeddingKey, embeddings: np.ndarray) -> Path:
        _, key_hash = self.resolve_key(key)
        stat = Path(key.video_path).stat()
        npy_path = self.array_dir / f"{key_hash}.npy"
        np.save(npy_path, embeddings.astype(np.float32, copy=False))
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key_hash, str(Path(key.video_path).resolve()), stat.st_mtime, stat.st_size, key.checkpoint,
                    key.start_s, key.duration_s, key.window_s, key.stride_s, key.frames_per_window,
                    embeddings.shape[0], embeddings.shape[1], npy_path.name, time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
        return npy_path

    def migrate(self) -> int:
        """Re-key rows written under an older hashing scheme (absolute paths);
        returns the number of rows rewritten."""
        with self._connect() as con:
            rows = con.execute("SELECT * FROM embeddings").fetchall()
            cols = [d[1] for d in con.execute("PRAGMA table_info(embeddings)")]
            changed = 0
            for row in rows:
                r = dict(zip(cols, row))
                key = EmbeddingKey(r["video_path"], r["checkpoint"], r["start_s"], r["duration_s"], r["window_s"], r["stride_s"], r["frames_per_window"])
                new_hash = self._hash(key, r["video_mtime"], r["video_size"])
                old_file = self._array_path(r["npy_path"])
                if new_hash == r["key_hash"] and old_file.name == r["npy_path"]:
                    continue
                new_file = self.array_dir / f"{new_hash}.npy"
                if old_file.exists() and old_file != new_file:
                    old_file.replace(new_file)
                con.execute("DELETE FROM embeddings WHERE key_hash = ?", (r["key_hash"],))
                con.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (new_hash, r["video_path"], r["video_mtime"], r["video_size"], r["checkpoint"], r["start_s"], r["duration_s"],
                     r["window_s"], r["stride_s"], r["frames_per_window"], r["num_windows"], r["hidden_size"], new_file.name, r["created_at"]),
                )
                changed += 1
        return changed

    def stats(self) -> dict:
        with self._connect() as con:
            n, total_windows = con.execute("SELECT COUNT(*), COALESCE(SUM(num_windows), 0) FROM embeddings").fetchone()
        total_bytes = sum(f.stat().st_size for f in self.array_dir.glob("*.npy"))
        return {"entries": n, "total_windows": total_windows, "total_mb": total_bytes / 1024**2, "root": str(self.root)}

    def vacuum(self) -> int:
        """Delete `.npy` files with no index row; returns the count removed."""
        with self._connect() as con:
            known = {self._array_path(row[0]).name for row in con.execute("SELECT npy_path FROM embeddings")}
        removed = 0
        for f in self.array_dir.glob("*.npy"):
            if f.name not in known:
                f.unlink()
                removed += 1
        return removed
