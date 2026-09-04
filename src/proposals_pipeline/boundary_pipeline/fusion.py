"""Combine per-source proposals with weighted temporal NMS."""

from __future__ import annotations

import numpy as np

from boundary_pipeline.proposals import Proposal, nms_1d, normalize_scores

SPACE = {
    "weight": [0.25, 0.5, 1.0, 2.0],
    "nms_window_s": [0.25, 0.5, 0.75, 1.0, 1.5],
    "min_fused_score": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
}
DEFAULTS = {"weights": {"abd": 1.0, "motion": 1.0, "sam3": 1.0}, "nms_window_s": 0.5, "min_fused_score": 0.0}


def fuse(per_source: dict[str, list[Proposal]], params: dict) -> list[Proposal]:
    """Weighted NMS over all sources. Scores are min-max normalized per
    source and video and multiplied by the source weight, then greedily
    suppressed within `nms_window_s`; a kept proposal's fused score adds the
    best suppressed score of every other source, and proposals below
    `min_fused_score` are dropped. Every source always contributes
    candidates: tuning never sets a weight to 0 (that is reserved for
    leave-one-out ablations)."""
    proposals: list[Proposal] = []
    scores: list[np.ndarray] = []
    for name, props in per_source.items():
        w = params["weights"].get(name, 0.0)
        if w <= 0 or not props:
            continue
        proposals.extend(props)
        scores.append(w * normalize_scores(props))
    if not proposals:
        return []
    kept, fused = nms_1d(proposals, np.concatenate(scores), params["nms_window_s"])
    return [Proposal(p.time_s, float(s), p.source, p.kind) for p, s in zip(kept, fused) if s >= params["min_fused_score"]]
