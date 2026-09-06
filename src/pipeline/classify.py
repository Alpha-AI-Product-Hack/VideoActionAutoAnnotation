from __future__ import annotations

import csv
import os
from pathlib import Path

from action_ranker.encode_actions import encode_action_batch
from action_ranker.gt_clips import ClipDecodeError, sample_gt_clip, synthetic_clip
from action_ranker.prompts import PROMPT_ID
from action_ranker.rank import ranking_from_similarities, rank_actions
from action_ranker.run import get_encoder
from action_ranker.text_cache import load_or_build_text_cache
from action_ranker.types import DictionaryRow
from pipeline.types import ActionSegment, ClipInterval, Rules


def select_encoder_name(requested: str | None = None) -> str:
    name = (requested or os.environ.get("PIPELINE_ENCODER") or "auto").strip().lower()
    if name in {"", "auto"}:
        return "xclip" if _cuda_available() else "stub"
    return name


def classify_clips(
    source: Path,
    clips: list[ClipInterval],
    rules: Rules,
    work_dir: Path,
    encoder_name: str | None = None,
) -> tuple[list[ActionSegment], str]:
    chosen = select_encoder_name(encoder_name)
    encoder = _load_encoder(chosen)
    labels = list(rules.actions)
    rows = [DictionaryRow(label=label) for label in labels]
    _write_bank_csv(work_dir / "action_bank.csv", rows)
    identity_keys = [row.label for row in rows]
    text = load_or_build_text_cache(
        encoder.encoder_id,
        "custom_rules",
        PROMPT_ID,
        labels,
        lambda labs: encode_action_batch(labs, encoder, PROMPT_ID),
    )

    segments: list[ActionSegment] = []
    for index, clip in enumerate(clips, start=1):
        frames = _load_frames(source, clip, encoder.num_frames)
        clip_id = clip.clip_id
        scorer = getattr(encoder, "score_clip_texts", None)
        if callable(scorer):
            similarities = scorer(frames[None, ...], text)
            ranking = ranking_from_similarities(
                similarities[0], labels, clip_id=clip_id, identity_keys=identity_keys
            )
        else:
            clip_vec = encoder.encode_clips(frames[None, ...])[0]
            ranking = rank_actions(clip_vec, text, labels, clip_id=clip_id, identity_keys=identity_keys)
        confidence = _score_to_confidence(ranking.cosine_similarity[0] if ranking.cosine_similarity else 0.0)
        action = ranking.pred_action
        obj = match_object(action, rules)
        start_ms = int(round(clip.start_sec * 1000))
        end_ms = int(round(clip.end_sec * 1000))
        keyframe_ms = start_ms + max(0, (end_ms - start_ms) // 2)
        segments.append(
            ActionSegment(
                id=str(index),
                start_ms=start_ms,
                end_ms=end_ms,
                action=action,
                object=obj,
                keyframe_ms=keyframe_ms,
                confidence=round(confidence, 4),
                model_version=rules.model_version,
                clip_id=clip.clip_id,
                topk_labels=list(ranking.labels[:5]),
                topk_scores=[float(s) for s in ranking.cosine_similarity[:5]],
            )
        )
    return segments, encoder.encoder_id


def match_object(pred_action: str, rules: Rules) -> str | None:
    pred = pred_action.strip().lower()
    if not rules.objects:
        return _object_from_label(pred_action, rules.actions)
    for obj in rules.objects:
        token = obj.strip().lower()
        if not token:
            continue
        if pred == token or pred.endswith(" " + token) or f" {token} " in f" {pred} ":
            return obj
    return _object_from_label(pred_action, rules.actions)


def _object_from_label(pred_action: str, actions: list[str]) -> str | None:
    pred = pred_action.strip()
    lower = pred.lower()
    for action in sorted(actions, key=len, reverse=True):
        prefix = action.strip().lower()
        if lower == prefix:
            return None
        if lower.startswith(prefix + " "):
            rest = pred[len(action) :].strip(" -_")
            return rest or None
    parts = pred.split()
    if len(parts) >= 2:
        return parts[-1]
    return None


def _load_encoder(name: str):
    try:
        return get_encoder(name)
    except Exception:
        if name != "stub":
            return get_encoder("stub")
        raise


def _load_frames(source: Path, clip: ClipInterval, num_frames: int):
    media = Path(source)
    clip_file = None
    if clip.path:
        candidate = media.parent / clip.path if not Path(clip.path).is_absolute() else Path(clip.path)
        if not candidate.is_file():
            candidate = media.parent / "clips" / f"{clip.clip_id}.mp4"
        if candidate.is_file():
            clip_file = candidate
    try:
        if clip_file is not None:
            frames = sample_gt_clip(clip_file, 0.0, max(0.05, clip.end_sec - clip.start_sec), num_frames=num_frames)
        else:
            frames = sample_gt_clip(media, clip.start_sec, clip.end_sec, num_frames=num_frames)
    except ClipDecodeError:
        frames = None
    if frames is None:
        seed = abs(hash((clip.clip_id, clip.start_sec, clip.end_sec))) % (2**32)
        return synthetic_clip(num_frames=num_frames, seed=seed)
    return frames


def _score_to_confidence(score: float) -> float:
    # Cosine similarity in [-1, 1] mapped to [0, 1].
    return float(min(1.0, max(0.0, (float(score) + 1.0) / 2.0)))


def _write_bank_csv(path: Path, rows: list[DictionaryRow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"label": row.label})
    return path


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
