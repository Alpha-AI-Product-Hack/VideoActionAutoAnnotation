"""Timeline plot: ground truth, per-source proposals and the fused result."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boundary_pipeline.datasets import GroundTruth
from boundary_pipeline.evaluation import match
from boundary_pipeline.proposals import Proposal

COLORS = {"abd": "#4C78A8", "motion": "#F58518", "sam3": "#54A24B", "fused": "#B279A2"}


def plot_timeline(
    out_path: str | Path,
    per_source: dict[str, list[Proposal]],
    fused: list[Proposal],
    gt: GroundTruth | None = None,
    tolerance_s: float = 1.0,
    window: tuple[float, float] | None = None,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(per_source) + ["fused"]
    lo, hi = window or (0.0, max([p.time_s for ps in per_source.values() for p in ps] + [p.time_s for p in fused] + [1.0]))
    fig, ax = plt.subplots(figsize=(max(10, (hi - lo) / 4), 1.0 + 0.6 * len(rows)))
    matched_gt: set[int] = set()
    fused_t = np.asarray([p.time_s for p in fused])
    if gt is not None:
        pairs = match(fused_t, gt.boundaries_s, tolerance_s)
        matched_det = {i for i, _ in pairs}
        matched_gt = {j for _, j in pairs}
        for j, t in enumerate(gt.boundaries_s):
            if lo <= t <= hi:
                ax.axvline(t, color="black" if j in matched_gt else "red", lw=0.8, alpha=0.5, zorder=0)
    else:
        matched_det = set()
    for k, name in enumerate(rows):
        y = len(rows) - 1 - k
        props = fused if name == "fused" else per_source[name]
        for i, p in enumerate(props):
            if not lo <= p.time_s <= hi:
                continue
            color = COLORS.get(name, "gray")
            if name == "fused" and gt is not None:
                color = "green" if i in matched_det else "red"
            ax.vlines(p.time_s, y + 0.1, y + 0.9, color=color, lw=1.2)
    ax.set_yticks([len(rows) - 1 - k + 0.5 for k in range(len(rows))])
    ax.set_yticklabels(rows)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, len(rows))
    ax.set_xlabel("time (s)")
    subtitle = f" (GT: black=matched @{tolerance_s}s, red=missed; fused: green=hit, red=false)" if gt is not None else ""
    ax.set_title(title + subtitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
