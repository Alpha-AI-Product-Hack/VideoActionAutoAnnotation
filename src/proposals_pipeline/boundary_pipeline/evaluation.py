"""Boundary matching and precision/recall/F1.

Detections and ground-truth boundaries are matched one-to-one within a
tolerance by the Hungarian algorithm, run per connected component of the
"within tolerance" graph so long videos stay cheap. Only detections inside
the ground truth's annotated spans are scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from boundary_pipeline.datasets import GroundTruth

TOLERANCES_S = (0.5, 1.0, 2.0)


@dataclass
class Counts:
    tp: int = 0
    n_det: int = 0
    n_gt: int = 0
    abs_err: list[float] = field(default_factory=list)

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.n_det + other.n_det, self.n_gt + other.n_gt, self.abs_err + other.abs_err)

    @property
    def precision(self) -> float:
        return self.tp / self.n_det if self.n_det else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def summary(self) -> dict:
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "tp": self.tp, "n_det": self.n_det, "n_gt": self.n_gt,
            "over_segmentation": self.n_det / self.n_gt if self.n_gt else None,
            "mean_abs_error_s": float(np.mean(self.abs_err)) if self.abs_err else None,
        }


def match(det: np.ndarray, gt: np.ndarray, tolerance_s: float) -> list[tuple[int, int]]:
    """Optimal one-to-one matching of sorted `det` to sorted `gt` within `tolerance_s`."""
    det, gt = np.asarray(det, dtype=float), np.asarray(gt, dtype=float)
    if len(det) == 0 or len(gt) == 0:
        return []
    order_d, order_g = np.argsort(det), np.argsort(gt)
    det_s, gt_s = det[order_d], gt[order_g]
    merged = np.concatenate([det_s, gt_s])
    is_det = np.concatenate([np.ones(len(det_s), bool), np.zeros(len(gt_s), bool)])
    local = np.concatenate([np.arange(len(det_s)), np.arange(len(gt_s))])
    order = np.argsort(merged, kind="stable")
    merged, is_det, local = merged[order], is_det[order], local[order]
    breaks = np.flatnonzero(np.diff(merged) > tolerance_s) + 1
    pairs: list[tuple[int, int]] = []
    for lo, hi in zip(np.concatenate([[0], breaks]), np.concatenate([breaks, [len(merged)]])):
        d_idx = local[lo:hi][is_det[lo:hi]]
        g_idx = local[lo:hi][~is_det[lo:hi]]
        if len(d_idx) == 0 or len(g_idx) == 0:
            continue
        cost = np.abs(det_s[d_idx][:, None] - gt_s[g_idx][None, :])
        big = tolerance_s * 10 + 1.0
        rows, cols = linear_sum_assignment(np.where(cost > tolerance_s, big, cost))
        for r, c in zip(rows, cols):
            if cost[r, c] <= tolerance_s:
                pairs.append((int(order_d[d_idx[r]]), int(order_g[g_idx[c]])))
    return pairs


def evaluate(det_times: np.ndarray, gt: GroundTruth, tolerances_s=TOLERANCES_S) -> dict[float, Counts]:
    det = np.asarray(det_times, dtype=float)
    det = det[gt.in_spans(det)]
    out = {}
    for tol in tolerances_s:
        pairs = match(det, gt.boundaries_s, tol)
        errs = [abs(det[i] - gt.boundaries_s[j]) for i, j in pairs]
        out[tol] = Counts(len(pairs), len(det), len(gt.boundaries_s), errs)
    return out


def aggregate(per_video: list[dict[float, Counts]]) -> dict[float, Counts]:
    total: dict[float, Counts] = {}
    for counts in per_video:
        for tol, c in counts.items():
            total[tol] = total.get(tol, Counts()) + c
    return total


def format_table(counts: dict[float, Counts]) -> str:
    lines = ["tol    P      R      F1     det/gt"]
    for tol, c in sorted(counts.items()):
        os_ = c.n_det / c.n_gt if c.n_gt else float("nan")
        lines.append(f"{tol:.1f}s  {c.precision:.3f}  {c.recall:.3f}  {c.f1:.3f}  {os_:.2f}")
    return "\n".join(lines)
