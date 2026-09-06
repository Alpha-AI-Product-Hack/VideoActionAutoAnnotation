from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.media import probe_video

ProgressFn = Callable[[int, str], None]

PROPOSALS_ROOT = Path(__file__).resolve().parents[1] / "proposals_pipeline"
DEFAULT_MOTION_PARAMS = {
    "signal": "r",
    "mode": "peak",
    "sigma": 2,
    "prominence": 1.0,
    "min_distance_s": 0.5,
}
DEFAULT_SAM3_PARAMS = {
    "dilate_px": 5,
    "min_obj_score": 0.5,
    "min_overlap_frac": 0.01,
    "min_hold_frames": 3,
    "min_track_len": 3,
    "use_switch": 1,
}
DEFAULT_ABD_PARAMS = {
    "mode": "fused",
    "gebd_weight": 0.6,
    "delta": 3,
    "sigma": 1.0,
    "prominence": 0.5,
    "min_distance_s": 0.5,
}


def ensure_proposals_on_path() -> Path:
    root = str(PROPOSALS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROPOSALS_ROOT


def boundaries_to_intervals(
    boundary_times: list[float],
    duration_s: float,
    min_duration_s: float = 0.5,
) -> list[tuple[float, float]]:
    """Turn change-point times into half-open `[start, end]` intervals."""
    duration_s = float(duration_s)
    if duration_s <= 0:
        return []
    min_duration_s = max(0.05, float(min_duration_s))
    cuts = [0.0]
    for t in sorted(float(x) for x in boundary_times):
        if t <= min_duration_s * 0.25 or t >= duration_s - min_duration_s * 0.25:
            continue
        if t - cuts[-1] >= min_duration_s * 0.5:
            cuts.append(t)
    if duration_s - cuts[-1] >= min_duration_s * 0.25:
        cuts.append(duration_s)
    else:
        cuts[-1] = duration_s

    raw: list[tuple[float, float]] = []
    for start, end in zip(cuts, cuts[1:]):
        if end <= start:
            continue
        raw.append((start, end))
    if not raw:
        return [(0.0, duration_s)]

    merged: list[tuple[float, float]] = []
    for start, end in raw:
        if end - start >= min_duration_s or not merged:
            merged.append((start, end))
        else:
            prev_start, _prev_end = merged[-1]
            merged[-1] = (prev_start, end)
    if len(merged) == 1 and merged[0][1] - merged[0][0] < min_duration_s:
        return [(0.0, duration_s)]
    return merged


def propose_boundaries(
    video_path: Path,
    work_dir: Path,
    sources: list[str] | None = None,
    config_path: Path | None = None,
    progress: ProgressFn | None = None,
) -> tuple[list[float], str, list[str]]:
    """Return `(times_s, segmenter_id, warnings)`."""
    warnings: list[str] = []
    requested = [s.strip() for s in (sources or ["motion"]) if s.strip()]
    if not requested:
        requested = ["motion"]

    times, used = _try_boundary_pipeline(video_path, work_dir, requested, config_path, progress, warnings)
    if times:
        return times, used, warnings

    progress and progress(45, "fallback frame-diff segmentation")
    fallback = frame_diff_boundaries(video_path)
    if fallback:
        warnings.append("boundary_pipeline unavailable or empty; used frame-diff fallback")
        return fallback, "frame_diff", warnings

    fps, duration_s = probe_video(video_path)
    warnings.append("no boundaries detected; using uniform windows")
    return uniform_boundaries(duration_s), "uniform", warnings


def uniform_boundaries(duration_s: float, window_s: float = 4.0) -> list[float]:
    if duration_s <= window_s * 1.5:
        return []
    times = []
    t = window_s
    while t < duration_s - 0.4:
        times.append(float(t))
        t += window_s
    return times


def frame_diff_boundaries(
    video_path: Path,
    sample_fps: float = 4.0,
    min_distance_s: float = 0.75,
) -> list[float]:
    samples = _sample_gray_frames(video_path, sample_fps)
    if samples is None:
        return []
    times, frames = samples
    if len(frames) < 4:
        return []
    diffs = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2))
    if diffs.size == 0:
        return []
    smooth = diffs.copy()
    if len(smooth) >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        smooth = np.convolve(diffs, kernel, mode="same")
    median = float(np.median(smooth))
    std = float(np.std(smooth))
    threshold = median + 1.25 * std
    min_dist = max(1, int(round(min_distance_s * sample_fps)))
    peaks: list[float] = []
    last_idx = -min_dist
    for i, value in enumerate(smooth):
        if value < threshold:
            continue
        if i - last_idx < min_dist:
            if peaks and value > smooth[last_idx]:
                peaks[-1] = float(times[i + 1])
                last_idx = i
            continue
        peaks.append(float(times[i + 1]))
        last_idx = i
    return peaks


def _try_boundary_pipeline(
    video_path: Path,
    work_dir: Path,
    sources: list[str],
    config_path: Path | None,
    progress: ProgressFn | None,
    warnings: list[str],
) -> tuple[list[float], str]:
    try:
        ensure_proposals_on_path()
        from boundary_pipeline.datasets import VideoItem, _video_meta
        from boundary_pipeline.fusion import fuse
        from boundary_pipeline.sources import SOURCE_NAMES, get_source
        from boundary_pipeline.tuning import Config
    except Exception as exc:
        warnings.append(f"could not import boundary_pipeline: {exc}")
        return [], ""

    allowed = [name for name in sources if name in SOURCE_NAMES]
    skipped = [name for name in sources if name not in SOURCE_NAMES]
    if skipped:
        warnings.append(f"unknown proposal sources skipped: {skipped}")
    if not allowed:
        allowed = ["motion"]

    config = _load_or_build_config(config_path, allowed, warnings)
    active = [name for name in allowed if name in config.sources]
    if not active:
        warnings.append("config has no requested sources")
        return [], ""

    cache_root = work_dir / "proposal_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        fps, duration = _video_meta(Path(video_path))
    except Exception as exc:
        warnings.append(f"boundary_pipeline probe failed: {exc}")
        return [], ""

    item = VideoItem(f"run/{Path(video_path).stem}", "custom", Path(video_path), fps, duration)
    per_source: dict[str, Any] = {}
    used: list[str] = []
    for name in active:
        src = get_source(name)
        params = config.sources[name]
        progress and progress(35, f"extract {name} signal")
        try:
            signal = _extract_source(src, item, name, params, cache_root)
            if signal is None:
                warnings.append(f"{name} produced no signal")
                continue
            per_source[name] = src.propose(signal, params)
            used.append(name)
        except Exception as exc:
            warnings.append(f"{name} failed: {exc}")

    if not per_source:
        return [], ""
    fused = fuse(per_source, config.fusion)
    times = [float(p.time_s) for p in fused]
    (work_dir / "boundaries.json").write_text(
        json.dumps(
            {
                "video": str(video_path),
                "sources": used,
                "boundaries": [p.to_dict() for p in fused],
                "per_source": {k: [p.to_dict() for p in v] for k, v in per_source.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return times, "+".join(used) if used else "boundary_pipeline"


def _load_or_build_config(config_path: Path | None, sources: list[str], warnings: list[str]):
    from boundary_pipeline.tuning import Config

    if config_path and Path(config_path).is_file():
        doc = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config = Config.from_dict(doc)
        keep = {name: params for name, params in config.sources.items() if name in sources}
        weights = dict(config.fusion.get("weights") or {})
        for name in list(weights):
            if name not in keep:
                weights[name] = 0.0
        for name in keep:
            weights.setdefault(name, 1.0)
        config.sources = keep
        config.fusion = {**config.fusion, "weights": weights}
        return config

    source_params = {
        "motion": dict(DEFAULT_MOTION_PARAMS),
        "sam3": dict(DEFAULT_SAM3_PARAMS),
        "abd": dict(DEFAULT_ABD_PARAMS),
    }
    keep = {name: source_params[name] for name in sources if name in source_params}
    weights = {name: 1.0 if name in keep else 0.0 for name in ("motion", "sam3", "abd")}
    return Config(keep, {"weights": weights, "nms_window_s": 0.5, "min_fused_score": 0.0})


def _extract_source(src, item, name: str, params: dict, cache_root: Path):
    cache_dir = cache_root / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if name == "motion":
        loaded = src.load(item, cache_dir=cache_dir)
        if loaded is None:
            src.extract(item, cache_dir=cache_dir)
            loaded = src.load(item, cache_dir=cache_dir)
        return loaded
    if name == "sam3":
        loaded = src.load(item, target_fps=4.0)
        if loaded is None:
            import torch
            from sam3_pipeline.sam3_tracker import Sam3FrameDetector

            detector = Sam3FrameDetector(dtype=torch.bfloat16)
            src.extract(item, detector, target_fps=4.0)
            del detector
            loaded = src.load(item, target_fps=4.0)
        return loaded
    loaded = src.load(item)
    if loaded is None:
        from action_boundaries.gebd import DDMNetScorer
        from action_boundaries.vjepa_encoder import VJepa21Encoder

        src.extract(item, VJepa21Encoder(), DDMNetScorer())
        loaded = src.load(item)
    return loaded


def _sample_gray_frames(video_path: Path, sample_fps: float) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import cv2
    except ImportError:
        return _sample_gray_frames_av(video_path, sample_fps)
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or sample_fps
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = n / fps if n else 0.0
        if duration <= 0:
            return None
        targets = np.arange(0.0, duration, 1.0 / sample_fps)
        frames = []
        times = []
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            frames.append(gray.astype(np.float32) / 255.0)
            times.append(t)
    finally:
        cap.release()
    if len(frames) < 4:
        return None
    return np.asarray(times, dtype=np.float32), np.stack(frames, axis=0)


def _sample_gray_frames_av(video_path: Path, sample_fps: float) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import av
        from PIL import Image
    except ImportError:
        try:
            import av
        except ImportError:
            return None
        Image = None
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        step = 1.0 / sample_fps
        next_t = 0.0
        frames = []
        times = []
        for frame in container.decode(stream):
            ts = float(frame.time or 0.0)
            if ts + 1e-3 < next_t:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if Image is not None:
                gray = np.asarray(Image.fromarray(rgb).convert("L").resize((64, 64)), dtype=np.float32) / 255.0
            else:
                gray = rgb.mean(axis=2)
                gray = _resize_nearest(gray, 64, 64)
            frames.append(gray)
            times.append(ts)
            next_t += step
        container.close()
    except Exception:
        return None
    if len(frames) < 4:
        return None
    return np.asarray(times, dtype=np.float32), np.stack(frames, axis=0)


def _resize_nearest(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = (np.linspace(0, gray.shape[0] - 1, height)).astype(int)
    xs = (np.linspace(0, gray.shape[1] - 1, width)).astype(int)
    return gray[ys][:, xs].astype(np.float32) / (255.0 if gray.max() > 1.5 else 1.0)
