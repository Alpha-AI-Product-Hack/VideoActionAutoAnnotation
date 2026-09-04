"""Training-free action boundary detection (ABD, Du et al. CVPR 2022), boundary
step only: local minima of the smoothed, z-scored similarity between
embeddings `delta` strides apart."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else np.zeros_like(x)


def cosine_valley_curve(embeddings: np.ndarray, stride_s: float, delta: int = 3, gaussian_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """`(times_s, d_z)` with `d_z = -zscore(smooth(cos(f_t, f_{t+delta})))`,
    timestamped at the midpoint of each pair; higher means more boundary-like."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (T, D), got shape {embeddings.shape}")
    if delta < 1:
        raise ValueError("delta must be >= 1 stride")
    if embeddings.shape[0] <= delta:
        raise ValueError(f"need more than `delta` embeddings (got T={embeddings.shape[0]}, delta={delta})")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    s = np.sum(unit[:-delta] * unit[delta:], axis=1)
    s = gaussian_filter1d(s, sigma=gaussian_sigma) if gaussian_sigma > 0 else s
    times = (np.arange(len(s)) + delta / 2.0) * stride_s
    return times, -zscore(s)


@dataclass
class BoundaryResult:
    boundary_times: np.ndarray
    boundary_scores: np.ndarray   # peak prominence
    boundary_indices: np.ndarray  # index into times_s / d_z
    times_s: np.ndarray
    d_z: np.ndarray
    delta: int
    stride_s: float
    params: dict = field(default_factory=dict)


def detect_boundaries(
    embeddings: np.ndarray,
    stride_s: float,
    delta: int = 3,
    gaussian_sigma: float = 1.0,
    prominence: float = 1.0,
    min_distance_s: float = 0.5,
) -> BoundaryResult:
    """Peaks of the cosine-valley curve with at least `prominence` (in
    std-dev units) and `min_distance_s` between them."""
    times, d_z = cosine_valley_curve(embeddings, stride_s, delta, gaussian_sigma)
    distance = max(1, round(min_distance_s / stride_s))
    idx, props = find_peaks(d_z, prominence=prominence, distance=distance)
    return BoundaryResult(
        times[idx], props["prominences"], idx, times, d_z, delta, stride_s,
        dict(gaussian_sigma=gaussian_sigma, prominence=prominence, min_distance_s=min_distance_s),
    )
