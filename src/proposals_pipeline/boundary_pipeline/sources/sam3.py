"""SAM3 hand-object contact proposals.

Extraction runs the batched per-frame SAM3 detector at `target_fps` and
caches, per sampled frame, the hand count and, per hand-adjacent
object, its score, area, tracklet id and its overlap with the hands at
three dilation radii. Proposing is then a cheap re-derivation of contact
on/off and held-object switch events under tunable thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from boundary_pipeline.datasets import VideoItem
from boundary_pipeline.proposals import Proposal
from sam3_pipeline.mask_ops import dilate_mask, overlap_frac

NAME = "sam3"
DILATIONS = (0, 5, 15)
CACHE_DIR = Path(".cache/sam3")
TARGET_FPS = 4.0

SPACE = {
    "dilate_px": list(DILATIONS),
    "min_obj_score": [0.3, 0.5, 0.7],
    "min_overlap_frac": [0.0, 0.01, 0.03, 0.1, 0.2],
    "min_hold_frames": [1, 2, 3, 5, 8],
    "min_track_len": [1, 2, 3, 5],
    "use_switch": [0, 1],
}
DEFAULTS = {"dilate_px": 5, "min_obj_score": 0.5, "min_overlap_frac": 0.01, "min_hold_frames": 3, "min_track_len": 3, "use_switch": 1}


@dataclass
class Sam3Signal:
    times_s: np.ndarray        # (F,)
    n_hands: np.ndarray        # (F,)
    obj_frame: np.ndarray      # (N,) index into times_s
    obj_score: np.ndarray      # (N,)
    obj_track: np.ndarray      # (N,)
    obj_overlap: np.ndarray    # (N, len(DILATIONS)) fraction of the dilated hand covered by the object
    track_len: np.ndarray      # (N,) length in frames of the object's tracklet


def cache_path(item: VideoItem, cache_dir: Path = CACHE_DIR, target_fps: float = TARGET_FPS) -> Path:
    return Path(cache_dir) / f"{item.slug}_{target_fps:g}fps.npz"


def iter_sampled_frames(video_path: Path, target_fps: float, max_dim: int | None, chunk: int):
    """Yield `(times_s, frames_rgb)` chunks, decoding sequentially."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, round(fps / target_fps))
        times, frames = [], []
        idx = 0
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_dim and max(frame.shape[:2]) > max_dim:
                    scale = max_dim / max(frame.shape[:2])
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                times.append(idx / fps)
                if len(frames) == chunk:
                    yield times, frames
                    times, frames = [], []
            elif not cap.grab():
                break
            idx += 1
        if frames:
            yield times, frames
    finally:
        cap.release()


def extract(
    item: VideoItem,
    detector,
    target_fps: float = TARGET_FPS,
    max_dim: int = 1008,
    chunk: int = 240,
    cache_dir: Path = CACHE_DIR,
    progress: bool = True,
) -> Path:
    from sam3_pipeline.object_tracks import GreedyIoULinker

    linker = GreedyIoULinker(iou_thresh=0.3)
    times, n_hands = [], []
    obj_frame, obj_score, obj_track, obj_overlap = [], [], [], []
    iterator = iter_sampled_frames(item.video_path, target_fps, max_dim, chunk)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=int(np.ceil(item.duration_s * target_fps / chunk)), desc=item.video_id)
    for chunk_times, frames in iterator:
        for t, fm in zip(chunk_times, detector.detect(frames)):
            fidx = len(times)
            times.append(t)
            n_hands.append(len(fm.hand_masks))
            dilated = {d: [dilate_mask(h, d) for h in fm.hand_masks] for d in DILATIONS}
            for tid, mask, score in zip(linker.step(fm.object_masks), fm.object_masks, fm.object_scores):
                obj_frame.append(fidx)
                obj_score.append(score)
                obj_track.append(tid)
                obj_overlap.append([max((overlap_frac(h, mask) for h in dilated[d]), default=0.0) for d in DILATIONS])
    obj_track = np.asarray(obj_track, dtype=np.int64)
    track_len = np.asarray([linker.track_len[t] for t in obj_track], dtype=np.int64)
    out = cache_path(item, cache_dir, target_fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times_s=np.asarray(times), n_hands=np.asarray(n_hands, dtype=np.int64),
        obj_frame=np.asarray(obj_frame, dtype=np.int64), obj_score=np.asarray(obj_score, dtype=np.float32),
        obj_track=obj_track, obj_overlap=np.asarray(obj_overlap, dtype=np.float32).reshape(-1, len(DILATIONS)),
        track_len=track_len,
        meta=json.dumps({"target_fps": target_fps, "max_dim": max_dim, "dilations": DILATIONS}),
    )
    return out


def load(item: VideoItem, cache_dir: Path = CACHE_DIR, target_fps: float = TARGET_FPS) -> Sam3Signal | None:
    path = cache_path(item, cache_dir, target_fps)
    if not path.is_file():
        return None
    z = np.load(path)
    return Sam3Signal(z["times_s"], z["n_hands"], z["obj_frame"], z["obj_score"], z["obj_track"], z["obj_overlap"], z["track_len"])


def _hysteresis(states: np.ndarray, min_hold: int) -> np.ndarray:
    """Accept a state change only once the new state has held `min_hold` frames."""
    if min_hold <= 1 or len(states) == 0:
        return states.copy()
    out = states.copy()
    run = 1
    for i in range(1, len(states)):
        run = run + 1 if states[i] == states[i - 1] else 1
        out[i] = states[i] if run >= min_hold else out[i - 1]
    return out


def contact_timeline(sig: Sam3Signal, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame `(in_contact, held_track)` under `params` (held_track = -1 when idle)."""
    d = DILATIONS.index(params["dilate_px"])
    ok = (
        (sig.obj_score >= params["min_obj_score"])
        & (sig.obj_overlap[:, d] >= params["min_overlap_frac"])
        & (sig.track_len >= params["min_track_len"])
    )
    if params["min_overlap_frac"] <= 0:
        ok &= sig.obj_overlap[:, d] > 0
    n = len(sig.times_s)
    best = np.full(n, -1.0)
    held = np.full(n, -1, dtype=np.int64)
    for f, t, o in zip(sig.obj_frame[ok], sig.obj_track[ok], sig.obj_overlap[ok, d]):
        if o > best[f]:
            best[f], held[f] = o, t
    raw = best >= 0
    contact = _hysteresis(raw, params["min_hold_frames"])
    held = np.where(contact, held, -1)
    if params["min_hold_frames"] > 1:
        held = _fill_held(held, contact)
    return contact, held


def _fill_held(held: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """After hysteresis a contact frame can lack a held id (its raw frame was idle); carry the previous id forward."""
    out = held.copy()
    for i in range(1, len(out)):
        if contact[i] and out[i] < 0:
            out[i] = out[i - 1]
    return out


def propose(sig: Sam3Signal, params: dict) -> list[Proposal]:
    contact, held = contact_timeline(sig, params)
    t = sig.times_s
    proposals: list[Proposal] = []
    changes = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    runs = np.diff(np.concatenate([[0], changes, [len(contact)]]))
    for k, i in enumerate(changes):
        persistence = min(runs[k], runs[k + 1]) * (t[1] - t[0] if len(t) > 1 else 1.0)
        proposals.append(Proposal(float((t[i - 1] + t[i]) / 2), float(persistence), NAME, "on" if contact[i] else "off"))
    if params["use_switch"]:
        both = contact[1:] & contact[:-1] & (held[1:] != held[:-1]) & (held[1:] >= 0) & (held[:-1] >= 0)
        for i in np.flatnonzero(both) + 1:
            if any(abs(p.time_s - (t[i - 1] + t[i]) / 2) < 1e-6 for p in proposals):
                continue
            proposals.append(Proposal(float((t[i - 1] + t[i]) / 2), 1.0, NAME, "switch"))
    proposals.sort(key=lambda p: p.time_s)
    return proposals
