"""Optical-flow motion proposals.

The camera-motion signal (`action_boundaries.camera_motion`) gives, per
sampled frame pair, the whole-frame flow magnitude `m` and the residual
foreground motion `r` that a background homography cannot explain. Either
signal, or the magnitude of its derivative, is log-compressed, smoothed,
z-scored and peak-picked; peaks mark motion bursts or motion changes,
valleys mark pauses between actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from boundary_pipeline.datasets import VideoItem
from boundary_pipeline.proposals import Proposal

NAME = "motion"
CACHE_DIR = Path(".cache/motion")
SAMPLE_FPS = 8.0

SPACE = {
    "signal": ["r", "m", "dr", "dm"],
    "mode": ["peak", "valley"],
    "sigma": [1, 2, 4, 8],
    "prominence": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    "min_distance_s": [0.25, 0.5, 1.0, 2.0],
}
DEFAULTS = {"signal": "r", "mode": "peak", "sigma": 2, "prominence": 1.0, "min_distance_s": 0.5}


@dataclass
class MotionSignal:
    times_s: np.ndarray
    m: np.ndarray
    r: np.ndarray
    ego_translation: np.ndarray
    inlier_ratio: np.ndarray


def cache_path(item: VideoItem, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{item.slug}.npz"


def extract(item: VideoItem, sample_fps: float = SAMPLE_FPS, max_dim: int = 480, cache_dir: Path = CACHE_DIR) -> Path:
    from action_boundaries.camera_motion import compute_camera_motion_signal

    res = compute_camera_motion_signal(str(item.video_path), start_s=0.0, duration_s=item.duration_s, sample_fps=sample_fps, max_dim=max_dim)
    out = cache_path(item, cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, times_s=res.times_s, m=res.m_t, r=res.r_t,
        ego_translation=res.ego_translation_px, inlier_ratio=res.inlier_ratio, sample_fps=sample_fps,
    )
    return out


def load(item: VideoItem, cache_dir: Path = CACHE_DIR) -> MotionSignal | None:
    path = cache_path(item, cache_dir)
    if not path.is_file():
        return None
    z = np.load(path)
    return MotionSignal(z["times_s"], z["m"], z["r"], z["ego_translation"], z["inlier_ratio"])


def curve(sig: MotionSignal, params: dict) -> np.ndarray:
    base = np.log1p(sig.r if params["signal"] in ("r", "dr") else sig.m)
    smooth = gaussian_filter1d(base, params["sigma"]) if params["sigma"] > 0 else base
    if params["signal"] in ("dr", "dm"):
        smooth = np.abs(np.gradient(smooth))
    std = smooth.std()
    z = (smooth - smooth.mean()) / std if std > 0 else np.zeros_like(smooth)
    return z if params["mode"] == "peak" else -z


def propose(sig: MotionSignal, params: dict) -> list[Proposal]:
    if len(sig.times_s) < 3:
        return []
    z = curve(sig, params)
    stride = float(np.median(np.diff(sig.times_s)))
    idx, props = find_peaks(z, prominence=params["prominence"], distance=max(1, round(params["min_distance_s"] / stride)))
    return [Proposal(float(sig.times_s[i]), float(p), NAME, params["signal"] + "-" + params["mode"]) for i, p in zip(idx, props["prominences"])]
