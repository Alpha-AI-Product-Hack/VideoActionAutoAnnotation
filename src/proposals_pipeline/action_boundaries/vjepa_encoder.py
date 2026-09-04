"""Dense sliding-window clip embeddings from V-JEPA 2.1.

Uses the community HF conversion `Dev-Jahn/vjepa2.1-vitl-fpc64-384` (custom
modeling code, hence `trust_remote_code=True`; reviewed before use). Only
the requested slice is decoded, frame by frame, resized to the crop size on
the way in. Token grids are mean-pooled into one vector per window.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from transformers import AutoModel

from .constants import DEFAULT_CHECKPOINT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MAX_BUFFER_BYTES = 8 * 1024**3


def _resize_and_center_crop(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = size / min(h, w)
    resized = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    top, left = (resized.shape[0] - size) // 2, (resized.shape[1] - size) // 2
    return cv2.cvtColor(resized[top:top + size, left:left + size], cv2.COLOR_BGR2RGB)


@dataclass
class DecodedSlice:
    frames: np.ndarray  # (N, size, size, 3) uint8 RGB
    fps: float
    start_s: float


def decode_slice(video_path: str, start_s: float, duration_s: float, crop_size: int, tail_pad_s: float = 1.0, force: bool = False) -> DecodedSlice:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or fps <= 0:
            raise IOError(f"video reports invalid fps ({fps}): {video_path}")
        n_needed = int(round((duration_s + tail_pad_s) * fps))
        est_bytes = n_needed * crop_size * crop_size * 3
        if est_bytes > _MAX_BUFFER_BYTES and not force:
            raise MemoryError(f"decoding {duration_s:.1f}s would buffer ~{est_bytes / 1024**3:.1f} GB; use a shorter duration or force=True")
        start_frame = min(max(int(round(start_s * fps)), 0), max(n_total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(n_needed):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(_resize_and_center_crop(frame, crop_size))
        if not frames:
            raise IOError(f"read 0 frames starting at {start_s:.2f}s from {video_path}")
        return DecodedSlice(frames=np.stack(frames), fps=fps, start_s=start_frame / fps)
    finally:
        cap.release()


class VJepa21Encoder:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None, dtype: torch.dtype = torch.float32, trust_remote_code: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype  # this checkpoint's RoPE promotes q/k to float32, so reduced precision fails in SDPA
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=trust_remote_code, dtype=self.dtype).to(self.device).eval()
        self.crop_size = self.model.config.crop_size
        self.tubelet_size = self.model.config.tubelet_size
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def embed_clips(self, clips: np.ndarray) -> np.ndarray:
        """`clips`: (B, T, H, W, 3) uint8 RGB, T a multiple of the tubelet size -> (B, hidden)."""
        if clips.shape[1] % self.tubelet_size != 0:
            raise ValueError(f"clip length {clips.shape[1]} must be a multiple of tubelet_size={self.tubelet_size}")
        x = torch.from_numpy(clips).to(self.device).permute(0, 4, 1, 2, 3).to(self.dtype) / 255.0
        x = (x - self._mean) / self._std
        return self.model.get_vision_features(x).mean(dim=1).float().cpu().numpy()


def extract_window_embeddings(
    video_path: str,
    encoder: VJepa21Encoder,
    start_s: float = 0.0,
    duration_s: float = 60.0,
    window_s: float = 1.0,
    stride_s: float = 0.125,
    frames_per_window: int = 8,
    batch_size: int = 16,
    force: bool = False,
    progress: bool = True,
) -> np.ndarray:
    """`(num_windows, hidden)` embeddings of `window_s` windows every `stride_s` over the slice."""
    if frames_per_window % encoder.tubelet_size != 0:
        raise ValueError(f"frames_per_window={frames_per_window} must be a multiple of tubelet_size={encoder.tubelet_size}")
    decoded = decode_slice(video_path, start_s=start_s, duration_s=duration_s + window_s, crop_size=encoder.crop_size, force=force)
    local_offsets = np.round(np.linspace(0, window_s, frames_per_window, endpoint=False) * decoded.fps).astype(int)
    n_windows = int(np.floor(duration_s / stride_s)) + 1
    starts = np.round(np.arange(n_windows) * stride_s * decoded.fps).astype(int)
    starts = starts[(starts + local_offsets[-1]) < len(decoded.frames)]

    iterator = range(0, len(starts), batch_size)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=(len(starts) + batch_size - 1) // batch_size, desc="V-JEPA2.1 windows")
    embeds = [encoder.embed_clips(np.stack([decoded.frames[s + local_offsets] for s in starts[i:i + batch_size]])) for i in iterator]
    return np.concatenate(embeds, axis=0)
