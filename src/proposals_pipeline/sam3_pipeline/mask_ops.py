"""Boolean-mask helpers."""

from __future__ import annotations

import cv2
import numpy as np


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def overlap_frac(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection area as a fraction of `mask_a`'s area."""
    area_a = mask_a.sum()
    if area_a == 0:
        return 0.0
    return np.logical_and(mask_a, mask_b).sum() / area_a


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return np.logical_and(mask_a, mask_b).sum() / union
