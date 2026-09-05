from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from action_ranker.types import InferenceTiming

T = TypeVar("T")


def elapsed(fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def build_timing(
    *,
    clip_duration_s: float,
    warmup_s: float,
    decode_s: float,
    encode_s: float,
    rank_s: float,
    encoder_id: str,
    frame_count: int,
) -> InferenceTiming:
    return InferenceTiming(
        clip_duration_s=float(clip_duration_s),
        warmup_s=float(warmup_s),
        inference_s=float(decode_s + encode_s + rank_s),
        encoder_id=encoder_id,
        frame_count=int(frame_count),
        decode_s=float(decode_s),
        encode_s=float(encode_s),
        rank_s=float(rank_s),
    )


def format_timing_line(timing: InferenceTiming) -> str:
    return (
        f"clip_duration_s={timing.clip_duration_s:.3f} "
        f"inference_s={timing.inference_s:.3f} "
        f"warmup_s={timing.warmup_s:.3f}"
    )
