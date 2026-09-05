"""Camera-motion signal from one pass of frame-to-frame tracking (pure OpenCV).

Per sampled frame pair: `m_t`, the median dense DIS optical-flow magnitude
(how much the whole frame moved), and `r_t`, the median residual of a
RANSAC homography over its outlier points (motion the background model
cannot explain, i.e. hands and objects). A pure head turn gives high `m_t`
and low `r_t`; a reach gives high `r_t`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

HandMaskFn = Callable[[np.ndarray], "np.ndarray | None"]


@dataclass
class CameraMotionResult:
    times_s: np.ndarray
    m_t: np.ndarray
    r_t: np.ndarray
    ego_translation_px: np.ndarray
    ego_rotation_rad: np.ndarray
    inlier_ratio: np.ndarray
    num_tracked: np.ndarray
    working_scale: float
    sample_fps: float
    params: dict = field(default_factory=dict)


def _resize_for_flow(frame_bgr: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return gray, scale


def _camera_motion_between(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    hand_mask: np.ndarray | None = None,
    max_corners: int = 600,
    corner_quality: float = 0.01,
    corner_min_distance: int = 7,
    ransac_reproj_thresh: float = 3.0,
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
) -> dict:
    """One frame pair's `(m_t, r_t, ego-motion)`. `hand_mask` (if given) is
    excluded from the dense median only; RANSAC needs the hand points to
    separate them as outliers."""
    flow = cv2.DISOpticalFlow_create(dis_preset).calc(prev_gray, curr_gray, None)
    mag = np.linalg.norm(flow, axis=2)
    if hand_mask is not None and (~hand_mask).any():
        m_t = float(np.median(mag[~hand_mask]))
    else:
        m_t = float(np.median(mag))
    empty = dict(m_t=m_t, r_t=0.0, ego_translation_px=0.0, ego_rotation_rad=0.0, inlier_ratio=0.0, num_tracked=0)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=corner_quality, minDistance=corner_min_distance)
    if prev_pts is None or len(prev_pts) < 8:
        return empty
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    status = status.reshape(-1).astype(bool)
    prev_valid, curr_valid = prev_pts[status].reshape(-1, 2), curr_pts[status].reshape(-1, 2)
    num_tracked = len(prev_valid)
    if num_tracked < 8:
        return {**empty, "num_tracked": num_tracked}
    H, inlier_mask = cv2.findHomography(prev_valid, curr_valid, cv2.RANSAC, ransac_reproj_thresh)
    if H is None:
        return {**empty, "num_tracked": num_tracked}

    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    warped = cv2.perspectiveTransform(prev_valid.reshape(-1, 1, 2), H).reshape(-1, 2)
    residual = np.linalg.norm(curr_valid - warped, axis=1)[~inlier_mask]
    min_outliers = max(5, round(0.02 * num_tracked))  # a few isolated KLT mismatches are not foreground motion
    return dict(
        m_t=m_t,
        r_t=float(np.median(residual)) if residual.size >= min_outliers else 0.0,
        ego_translation_px=float(np.hypot(H[0, 2], H[1, 2])),
        ego_rotation_rad=float(np.arctan2(H[1, 0], H[0, 0])),
        inlier_ratio=float(inlier_mask.mean()),
        num_tracked=num_tracked,
    )


def compute_camera_motion_signal(
    video_path: str,
    start_s: float = 0.0,
    duration_s: float = 60.0,
    sample_fps: float = 4.0,
    hand_mask_fn: HandMaskFn | None = None,
    max_dim: int = 480,
    max_corners: int = 600,
    ransac_reproj_thresh: float = 3.0,
) -> CameraMotionResult:
    """Sample `[start_s, start_s + duration_s)` at `sample_fps` and compute
    the signal between consecutive samples. `times_s` is absolute (video time)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")
    cols = {k: [] for k in ("times_s", "m_t", "r_t", "ego_translation_px", "ego_rotation_rad", "inlier_ratio", "num_tracked")}
    working_scale = 1.0
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            raise IOError(f"video reports invalid fps ({native_fps}): {video_path}")
        start_frame = round(start_s * native_fps)
        frame_step = max(1, round(native_fps / sample_fps))
        n_samples = max(2, round(duration_s * sample_fps) + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev_gray, frame_idx, collected = None, start_frame, 0
        while collected < n_samples:
            ok, frame = cap.read()
            if not ok:
                break
            gray, working_scale = _resize_for_flow(frame, max_dim)
            mask = None
            if hand_mask_fn is not None:
                raw = hand_mask_fn(frame)
                if raw is not None:
                    mask = cv2.resize(raw.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if prev_gray is not None:
                stats = _camera_motion_between(prev_gray, gray, hand_mask=mask, max_corners=max_corners, ransac_reproj_thresh=ransac_reproj_thresh)
                cols["times_s"].append((frame_idx - start_frame) / native_fps + start_s)
                for k in stats:
                    cols[k].append(stats[k])
            prev_gray = gray
            collected += 1
            frame_idx += frame_step
            for _ in range(frame_step - 1):
                if collected >= n_samples or not cap.grab():
                    break
    finally:
        cap.release()
    return CameraMotionResult(
        **{k: np.asarray(v, dtype=np.int64 if k == "num_tracked" else np.float64) for k, v in cols.items()},
        working_scale=float(working_scale),
        sample_fps=sample_fps,
        params=dict(max_dim=max_dim, max_corners=max_corners, ransac_reproj_thresh=ransac_reproj_thresh),
    )
