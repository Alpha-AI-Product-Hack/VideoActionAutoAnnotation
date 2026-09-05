from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from action_ranker.rank import ranking_from_similarities
from action_ranker.types import ActionRanking

AggregationName = Literal["mean", "max", "top2_mean", "lse"]


@dataclass(frozen=True)
class FaesDebug:
    aggregation: str
    frame_scores_shape: list[int]
    top_frame_index: int
    top_frame_score: float
    mean_top1_margin: float

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_frame_action_scores(
    frame_scores: np.ndarray,
    aggregation: AggregationName = "top2_mean",
) -> np.ndarray:
    """Aggregate FAES scores [T, M] into one action score vector [M]."""
    scores = np.asarray(frame_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("frame_scores must be [T, M]")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("frame_scores must be non-empty")
    if aggregation == "mean":
        return scores.mean(axis=0)
    if aggregation == "max":
        return scores.max(axis=0)
    if aggregation == "top2_mean":
        k = min(2, scores.shape[0])
        top = np.partition(scores, scores.shape[0] - k, axis=0)[-k:]
        return top.mean(axis=0)
    if aggregation == "lse":
        temperature = 0.02
        shifted = scores / temperature
        shifted = shifted - shifted.max(axis=0, keepdims=True)
        return temperature * (np.log(np.exp(shifted).mean(axis=0)) + scores.max(axis=0) / temperature)
    raise ValueError(f"Unsupported FAES aggregation {aggregation!r}")


def rank_faes_scores(
    frame_scores: np.ndarray,
    labels: list[str],
    *,
    clip_id: str = "",
    identity_keys: list | None = None,
    aggregation: AggregationName = "top2_mean",
) -> tuple[ActionRanking, FaesDebug]:
    """Rank actions from frame-action similarities without clip-level pooling."""
    scores = np.asarray(frame_scores, dtype=np.float32)
    action_scores = aggregate_frame_action_scores(scores, aggregation)
    ranking = ranking_from_similarities(
        action_scores,
        labels,
        clip_id=clip_id,
        identity_keys=identity_keys,
    )
    per_frame_best = scores.max(axis=1)
    per_frame_second = np.partition(scores, -2, axis=1)[:, -2] if scores.shape[1] >= 2 else per_frame_best
    top_frame_index = int(np.argmax(per_frame_best))
    debug = FaesDebug(
        aggregation=aggregation,
        frame_scores_shape=[int(item) for item in scores.shape],
        top_frame_index=top_frame_index,
        top_frame_score=float(per_frame_best[top_frame_index]),
        mean_top1_margin=float(np.mean(per_frame_best - per_frame_second)),
    )
    return ranking, debug


def score_faes_clip(
    encoder,
    frames: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list[str],
    *,
    clip_id: str = "",
    identity_keys: list | None = None,
    aggregation: AggregationName = "top2_mean",
) -> tuple[ActionRanking, FaesDebug, np.ndarray]:
    scorer = getattr(encoder, "score_frames_texts", None)
    if not callable(scorer):
        raise TypeError(f"Encoder {encoder.encoder_id!r} does not support FAES frame scoring")
    batch = frames[None, ...] if frames.ndim == 4 else frames
    frame_scores = scorer(batch, text_embeddings)
    if frame_scores.ndim != 3 or frame_scores.shape[0] != 1:
        raise ValueError("FAES scorer must return [1, T, M] for one clip")
    ranking, debug = rank_faes_scores(
        frame_scores[0],
        labels,
        clip_id=clip_id,
        identity_keys=identity_keys,
        aggregation=aggregation,
    )
    return ranking, debug, frame_scores[0]
