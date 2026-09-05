from __future__ import annotations

from typing import Protocol

import numpy as np


class FrozenEncoder(Protocol):
    encoder_id: str
    dim: int
    num_frames: int

    def encode_clips(self, frames: np.ndarray) -> np.ndarray:
        """frames: float32 [B, T, C, H, W] -> float32 [B, D]. No grad."""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """texts length M -> float32 [M, D]. No grad."""
