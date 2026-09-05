"""Frame-level generic boundary probability `b_t` from DDM-Net (MCG-NJU/DDM,
CVPR 2022), vendored under `third_party/ddm_net/` (MIT).

Input contract matches DDM-Net's training: for each candidate time, the 10
native frames at `+-{1..5} * ds` around it, squashed to 224x224 with
nearest-neighbour resize and ImageNet-normalized. `ds` is derived from
`ds_seconds` per video fps (the original `ds=3` frames at ~25-30 fps).
Output is the positive-class softmax of the last (fused) decoder layer.
The checkpoint is loaded strictly: any key mismatch raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .third_party.ddm_net.resnetGEBD import resnetGEBD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FRAMES_PER_SIDE = 5  # the only released configuration
INPUT_SIZE = 224
_CHECKPOINT_DRIVE_ID = "1k66v9VuFgah3Wx6eKmWpj8_b5MXxlwt7"
_DEFAULT_CHECKPOINT_PATH = Path(".cache/ddm_net/checkpoint.pth.tar")
_MAX_BUFFER_BYTES = 8 * 1024**3


def _block_frame_offsets(frames_per_side: int, ds_frames: int) -> np.ndarray:
    shift = np.arange(-frames_per_side, frames_per_side)
    shift[shift >= 0] += 1
    return shift * ds_frames


def ensure_checkpoint(path: str | Path | None = None) -> Path:
    """Download the ~1.75GB Kinetics-GEBD checkpoint from Google Drive if absent."""
    path = Path(path) if path else _DEFAULT_CHECKPOINT_PATH
    if path.exists():
        return path
    import gdown

    path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=_CHECKPOINT_DRIVE_ID, output=str(path), quiet=False)
    if not path.exists():
        raise IOError(f"checkpoint download failed, nothing at {path}")
    return path


@dataclass
class DecodedNative:
    frames: np.ndarray  # (N, 224, 224, 3) uint8 RGB
    fps: float
    start_s: float


def decode_slice_for_gebd(video_path: str, start_s: float, duration_s: float, tail_pad_s: float = 1.0, force: bool = False) -> DecodedNative:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or fps <= 0:
            raise IOError(f"video reports invalid fps ({fps}): {video_path}")
        n_needed = int(round((duration_s + tail_pad_s) * fps))
        est_bytes = n_needed * INPUT_SIZE * INPUT_SIZE * 3
        if est_bytes > _MAX_BUFFER_BYTES and not force:
            raise MemoryError(f"decoding {duration_s:.1f}s would buffer ~{est_bytes / 1024**3:.1f} GB; use a shorter duration or force=True")
        start_frame = min(max(int(round(start_s * fps)), 0), max(n_total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(n_needed):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))
        if not frames:
            raise IOError(f"read 0 frames starting at {start_s:.2f}s from {video_path}")
        return DecodedNative(frames=np.stack(frames), fps=fps, start_s=start_frame / fps)
    finally:
        cap.release()


class DDMNetScorer:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None, dtype: torch.dtype = torch.float32):
        import argparse

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model = resnetGEBD(backbone="resnet50", pretrained=False, num_classes=2, frames_per_side=FRAMES_PER_SIDE)
        ckpt_path = ensure_checkpoint(checkpoint_path)
        torch.serialization.add_safe_globals([argparse.Namespace])  # the checkpoint pickles its training args
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"DDM-Net checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys (e.g. {missing[:3]}, {unexpected[:3]})")
        self.model.to(self.device, dtype=self.dtype).eval()
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.dtype).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.dtype).view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def score_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """`blocks`: (N, 10, 224, 224, 3) uint8 RGB -> (N,) boundary probabilities."""
        x = torch.from_numpy(blocks).to(self.device).permute(0, 1, 4, 2, 3).to(self.dtype) / 255.0
        x = (x - self._mean) / self._std
        results, _, _ = self.model(x)
        return torch.softmax(results[-1], dim=1)[:, 1].float().cpu().numpy()


@dataclass
class GEBDResult:
    times_s: np.ndarray  # slice-relative
    b_t: np.ndarray
    ds_frames: int
    native_fps: float


def compute_gebd_scores(
    video_path: str,
    scorer: DDMNetScorer,
    start_s: float = 0.0,
    duration_s: float = 60.0,
    stride_s: float = 0.125,
    ds_seconds: float = 0.1,
    batch_size: int = 16,
    force: bool = False,
    progress: bool = True,
) -> GEBDResult:
    """`b_t` at every `stride_s` over the slice; candidates whose frame block
    falls outside the decoded range are dropped (the first few and, at the
    video end, the last few)."""
    decoded = decode_slice_for_gebd(video_path, start_s=start_s, duration_s=duration_s, force=force)
    ds_frames = max(1, round(ds_seconds * decoded.fps))
    offsets = _block_frame_offsets(FRAMES_PER_SIDE, ds_frames)
    n_samples = int(np.floor(duration_s / stride_s)) + 1
    current = np.round(np.arange(n_samples) * stride_s * decoded.fps).astype(int)
    last_idx = len(decoded.frames) - 1
    current = current[(current + offsets.min() >= 0) & (current + offsets.max() <= last_idx)]

    iterator = range(0, len(current), batch_size)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=(len(current) + batch_size - 1) // batch_size, desc="DDM-Net b_t")
    scores = [scorer.score_blocks(decoded.frames[np.clip(current[i:i + batch_size, None] + offsets[None, :], 0, last_idx)]) for i in iterator]
    b_t = np.concatenate(scores, axis=0) if scores else np.zeros(0, dtype=np.float32)
    return GEBDResult(times_s=current / decoded.fps, b_t=b_t, ds_frames=ds_frames, native_fps=decoded.fps)
