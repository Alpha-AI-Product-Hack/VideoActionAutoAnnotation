import cv2
import numpy as np

from action_boundaries.camera_motion import _camera_motion_between


def _textured_frame(size=400, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 128, dtype=np.uint8)
    for _ in range(120):
        x, y = rng.integers(0, size, size=2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(4, 14)), int(rng.integers(0, 255)), -1)
    return img


def _pan(img, dx, dy):
    h, w = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=128)


def _pan_with_local_patch(img, dx, dy, patch_dx, patch_dy, patch_box=(150, 150, 240, 240)):
    out = _pan(img, dx, dy)
    x0, y0, x1, y1 = patch_box
    shifted = _pan(img, dx + patch_dx, dy + patch_dy)
    out[y0:y1, x0:x1] = shifted[y0:y1, x0:x1]
    return out


def test_pure_pan_has_high_motion_but_low_residual():
    bg = _textured_frame()
    stats = _camera_motion_between(bg, _pan(bg, dx=12, dy=6))
    assert stats["num_tracked"] >= 8
    assert abs(stats["m_t"] - np.hypot(12, 6)) < 3.0
    assert stats["inlier_ratio"] > 0.8
    assert abs(stats["ego_translation_px"] - np.hypot(12, 6)) < 3.0
    assert stats["r_t"] < 1.5


def test_pan_plus_local_motion_has_higher_residual_than_pure_pan():
    bg = _textured_frame()
    pure = _camera_motion_between(bg, _pan(bg, dx=12, dy=6))
    reach = _camera_motion_between(bg, _pan_with_local_patch(bg, dx=12, dy=6, patch_dx=25, patch_dy=-20))
    assert abs(pure["m_t"] - reach["m_t"]) < 4.0
    assert reach["r_t"] > pure["r_t"] + 2.0


def test_static_frames_have_near_zero_motion():
    bg = _textured_frame()
    stats = _camera_motion_between(bg, bg.copy())
    assert stats["m_t"] < 0.5 and stats["r_t"] < 0.5 and stats["ego_translation_px"] < 0.5
