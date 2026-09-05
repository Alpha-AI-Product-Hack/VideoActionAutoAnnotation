"""Boundary proposals and temporal non-maximum suppression."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Proposal:
    time_s: float
    score: float
    source: str
    kind: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_scores(proposals: list[Proposal]) -> np.ndarray:
    """Min-max normalize scores to [0, 1] (all 1.0 when scores are equal)."""
    if not proposals:
        return np.zeros(0)
    s = np.asarray([p.score for p in proposals], dtype=float)
    lo, hi = s.min(), s.max()
    return np.ones_like(s) if hi <= lo else (s - lo) / (hi - lo)


def nms_1d(
    proposals: list[Proposal],
    scores: np.ndarray,
    window_s: float,
) -> tuple[list[Proposal], np.ndarray]:
    """Greedy temporal NMS. Proposals are visited by descending `scores`;
    one is kept if no kept proposal lies within `window_s`. Each kept
    proposal's fused score is its own score plus, per other source, the best
    score among the proposals it suppressed. Returns `(kept, fused_scores)`
    sorted by time."""
    if not proposals:
        return [], np.zeros(0)
    times = np.asarray([p.time_s for p in proposals], dtype=float)
    order = np.argsort(-scores, kind="stable")
    kept_idx: list[int] = []
    kept_times: list[float] = []
    support: list[dict[str, float]] = []
    for i in order:
        if kept_times:
            d = np.abs(np.asarray(kept_times) - times[i])
            j = int(np.argmin(d))
            if d[j] <= window_s:
                src = proposals[i].source
                if src != proposals[kept_idx[j]].source:
                    support[j][src] = max(support[j].get(src, 0.0), float(scores[i]))
                continue
        kept_idx.append(int(i))
        kept_times.append(float(times[i]))
        support.append({})
    fused = np.asarray([scores[i] + sum(s.values()) for i, s in zip(kept_idx, support)])
    order_t = np.argsort(kept_times)
    return [proposals[kept_idx[k]] for k in order_t], fused[order_t]
