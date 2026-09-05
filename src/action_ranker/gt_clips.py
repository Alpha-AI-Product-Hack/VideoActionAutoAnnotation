from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_NUM_FRAMES = 8
DEFAULT_SIZE = 224


class ClipDecodeError(Exception):
    pass


def sample_gt_clip(
    media_path: str | Path,
    start_sec: float,
    end_sec: float,
    num_frames: int = DEFAULT_NUM_FRAMES,
    height: int = DEFAULT_SIZE,
    width: int = DEFAULT_SIZE,
) -> np.ndarray | None:
    """Return float32 [T, C, H, W] in [0, 1], or None if the interval is invalid."""
    if not _valid_interval(start_sec, end_sec):
        return None
    path = Path(media_path)
    if not path.is_file():
        raise ClipDecodeError(f"Unreadable clip: missing file {path}")
    frames = _read_interval_frames(path, start_sec, end_sec, num_frames, height, width)
    if frames is None:
        raise ClipDecodeError(f"Unreadable clip: failed to decode {path}")
    return frames


def synthetic_clip(num_frames: int = DEFAULT_NUM_FRAMES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((num_frames, 3, DEFAULT_SIZE, DEFAULT_SIZE), dtype=np.float32)


def _valid_interval(start_sec: float, end_sec: float) -> bool:
    if not np.isfinite(start_sec) or not np.isfinite(end_sec):
        return False
    return float(end_sec) > float(start_sec)


def _read_interval_frames(
    path: Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray | None:
    av_frames = _read_interval_frames_av(path, start_sec, end_sec, num_frames, height, width)
    if av_frames is not None:
        return av_frames
    try:
        import torchvision.io as tvio
    except ImportError:
        tvio = None
    if tvio is not None and hasattr(tvio, "read_video"):
        try:
            video, _, info = tvio.read_video(str(path), start_pts=start_sec, end_pts=end_sec, pts_unit="sec")
            if video.numel() == 0:
                return None
            arr = video.numpy()  # T, H, W, C uint8
            return _resample_thwc(arr, num_frames, height, width)
        except Exception as exc:  # noqa: BLE001
            raise ClipDecodeError(str(exc)) from exc
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ClipDecodeError(
            "Install PyAV (`av`), torchvision with read_video, or decord to decode videos, or use --synthetic"
        ) from exc
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps() or 30.0)
    start_f = max(int(start_sec * fps), 0)
    end_f = min(int(end_sec * fps), len(vr) - 1)
    if end_f <= start_f:
        return None
    idxs = np.linspace(start_f, end_f, num=num_frames, dtype=np.int32)
    batch = vr.get_batch(idxs).asnumpy()
    return _resample_thwc(batch, num_frames, height, width)


def _read_interval_frames_av(
    path: Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray | None:
    try:
        import av
    except ImportError:
        return None
    targets = np.linspace(float(start_sec), float(end_sec), num_frames)
    out = np.zeros((num_frames, 3, height, width), dtype=np.float32)
    filled = np.zeros(num_frames, dtype=bool)
    target_index = 0
    last_rgb: np.ndarray | None = None
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if target_index >= num_frames:
                break
            ts = _frame_time(frame, stream)
            if ts < targets[target_index]:
                continue
            last_rgb = frame.to_ndarray(format="rgb24")
            while target_index < num_frames and ts >= targets[target_index]:
                out[target_index] = _resize_rgb_to_chw(last_rgb, height, width)
                filled[target_index] = True
                target_index += 1
        container.close()
    except Exception as exc:  # noqa: BLE001
        raise ClipDecodeError(str(exc)) from exc
    if target_index < num_frames and last_rgb is not None:
        last = _resize_rgb_to_chw(last_rgb, height, width)
        while target_index < num_frames:
            out[target_index] = last
            filled[target_index] = True
            target_index += 1
    if not bool(filled.all()):
        return None
    return out


def _frame_time(frame, stream) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None:
        return 0.0
    return float(frame.pts * stream.time_base)


def _resize_rgb_to_chw(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    ys = np.linspace(0, src_h - 1, height).astype(np.int32)
    xs = np.linspace(0, src_w - 1, width).astype(np.int32)
    resized = rgb[ys][:, xs]
    if resized.shape[-1] == 3:
        chw = np.transpose(resized, (2, 0, 1))
    else:
        chw = np.repeat(resized[..., None], 3, axis=2).transpose(2, 0, 1)
    return chw.astype(np.float32) / 255.0


def _resample_thwc(thwc: np.ndarray, num_frames: int, height: int, width: int) -> np.ndarray:
    if thwc.ndim != 4:
        raise ClipDecodeError("decoded video must be [T, H, W, C]")
    t = thwc.shape[0]
    idxs = np.linspace(0, t - 1, num=num_frames, dtype=np.int32)
    picked = thwc[idxs]
    # nearest-neighbor resize without extra deps
    out = np.zeros((num_frames, 3, height, width), dtype=np.float32)
    src_h, src_w = picked.shape[1], picked.shape[2]
    ys = (np.linspace(0, src_h - 1, height)).astype(np.int32)
    xs = (np.linspace(0, src_w - 1, width)).astype(np.int32)
    for i in range(num_frames):
        frame = picked[i]
        resized = frame[ys][:, xs]
        if resized.shape[-1] == 3:
            chw = np.transpose(resized, (2, 0, 1))
        else:
            chw = np.repeat(resized[..., None], 3, axis=2).transpose(2, 0, 1)
        out[i] = chw.astype(np.float32) / 255.0
    return out
