"""Greedy IoU linking of per-frame object masks into tracklets."""

from __future__ import annotations

import numpy as np

from sam3_pipeline.mask_ops import mask_iou


class GreedyIoULinker:
    """Assigns a persistent track id to each mask by greedy IoU matching
    against the previous frame's masks. State carries across calls, so a
    long video can be linked chunk by chunk."""

    def __init__(self, iou_thresh: float = 0.3):
        self.iou_thresh = iou_thresh
        self.active: dict[int, np.ndarray] = {}
        self.next_id = 0
        self.track_len: dict[int, int] = {}

    def step(self, masks: list[np.ndarray]) -> list[int]:
        assignment: list[int] = []
        used: set[int] = set()
        for mask in masks:
            best, best_iou = None, 0.0
            for tid, prev in self.active.items():
                if tid in used:
                    continue
                iou = mask_iou(mask, prev)
                if iou >= self.iou_thresh and iou > best_iou:
                    best, best_iou = tid, iou
            if best is None:
                best = self.next_id
                self.next_id += 1
            assignment.append(best)
            used.add(best)
            self.track_len[best] = self.track_len.get(best, 0) + 1
        self.active = {tid: m for tid, m in zip(assignment, masks)}
        return assignment
