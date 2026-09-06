from __future__ import annotations

import argparse
import contextlib
import csv
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from action_ranker.encode_actions import encode_action_batch
from action_ranker.encode_clips import encode_clip_batch
from action_ranker.faes import AggregationName, score_faes_clip
from action_ranker.gt_clips import ClipDecodeError, sample_gt_clip, synthetic_clip
from action_ranker.metrics import compute_action_metrics
from action_ranker.prompts import PROMPT_ID
from action_ranker.rank import gold_rank_1based, rank_actions, ranking_from_similarities
from action_ranker.skip_log import write_skip_log
from action_ranker.slice import (
    DataAvailabilityError,
    FrozenSliceError,
    build_slice,
    load_frozen_slice,
    load_intervals,
)
from action_ranker.stub_encoder import StubEncoder
from action_ranker.taxonomies import REPO_ROOT, load_dictionary_rows, taxonomy_path
from action_ranker.text_cache import load_or_build_text_cache
from action_ranker.timing import build_timing, elapsed, format_timing_line
from action_ranker.types import DictionaryRow, GoldInterval, PredictionRecord, SkipEvent

BANK_FOR_DATASET = {
    "assembly101": "assembly101_coarse",
    "epic_kitchens": "epic_kitchens_observed",
}
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"
TOPK = 5


XCLIP_MODEL_IDS = {
    "xclip": "microsoft/xclip-base-patch32",
    "xclip-zs": "microsoft/xclip-base-patch16-zero-shot",
}

INTERNVIDEO2_MODEL_IDS = {
    "internvideo2-1b": "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
    "internvideo2-6b": "OpenGVLab/InternVideo2-Stage2_6B",
}

ENCODER_CHOICES = ["stub", *XCLIP_MODEL_IDS.keys(), *INTERNVIDEO2_MODEL_IDS.keys()]


def get_encoder(
    name: str,
    num_frames: int | None = None,
    device: str | None = None,
    dtype: str = "auto",
    internvideo2_checkpoint_path: str | None = None,
    internvideo2_checkpoint_repo: str | None = None,
):
    if name == "stub":
        return StubEncoder()
    if name in XCLIP_MODEL_IDS:
        from action_ranker.xclip_encoder import XClipEncoder

        kwargs: dict = {"model_id": XCLIP_MODEL_IDS[name]}
        if num_frames is not None:
            kwargs["num_frames"] = num_frames
        if device not in {None, "auto"}:
            kwargs["device"] = device
        return XClipEncoder(**kwargs)
    if name in INTERNVIDEO2_MODEL_IDS:
        from action_ranker.internvideo2_encoder import InternVideo2Encoder

        return InternVideo2Encoder(
            model_id=INTERNVIDEO2_MODEL_IDS[name],
            num_frames=num_frames,
            device=device,
            dtype=dtype,
            checkpoint_path=internvideo2_checkpoint_path,
            checkpoint_repo=internvideo2_checkpoint_repo,
        )
    raise ValueError(f"Unknown encoder {name!r}")


def _rank_frames(encoder, frames, text, labels, clip_id: str, identity_keys):
    batch = frames[None, ...] if frames.ndim == 4 else frames
    scorer = getattr(encoder, "score_clip_texts", None)
    if callable(scorer):
        similarities = scorer(batch, text)
        return ranking_from_similarities(
            similarities[0], labels, clip_id=clip_id, identity_keys=identity_keys
        )
    clip_vec = encode_clip_batch(batch, encoder)[0]
    return rank_actions(clip_vec, text, labels, clip_id=clip_id, identity_keys=identity_keys)


def rank_one_clip(
    encoder,
    dictionary_id: str,
    frames: np.ndarray,
    clip_id: str,
) -> tuple[object, list[str], np.ndarray]:
    rows = load_dictionary_rows(dictionary_id)
    labels = [row.label for row in rows]
    identity_keys = [row.identity_key(dictionary_id) for row in rows]
    text = load_or_build_text_cache(
        encoder.encoder_id,
        dictionary_id,
        PROMPT_ID,
        labels,
        lambda labs: encode_action_batch(labs, encoder, PROMPT_ID),
    )
    ranking = _rank_frames(encoder, frames, text, labels, clip_id, identity_keys)
    return ranking, labels, text


def prediction_from_ranking(
    *,
    dataset: str,
    interval: GoldInterval,
    ranking,
    dictionary_id: str,
    encoder,
    inference_s: float | None = None,
    decode_s: float | None = None,
    encode_s: float | None = None,
    rank_s: float | None = None,
    prompt_id: str = PROMPT_ID,
) -> PredictionRecord:
    k = min(TOPK, len(ranking.labels))
    return PredictionRecord(
        dataset=dataset,
        video_id=interval.video_id,
        start_sec=interval.start_sec,
        end_sec=interval.end_sec,
        gold_action=interval.gold_action,
        pred_action=ranking.pred_action,
        topk_labels=ranking.labels[:k],
        topk_scores=ranking.cosine_similarity[:k],
        dictionary_id=dictionary_id,
        encoder_id=encoder.encoder_id,
        prompt_id=prompt_id,
        gold_rank=gold_rank_1based(
            ranking,
            interval.gold_action,
            gold_verb_id=interval.gold_verb_id,
            gold_noun_id=interval.gold_noun_id,
            dictionary_id=dictionary_id,
        ),
        frame_count=int(encoder.num_frames),
        inference_s=inference_s,
        decode_s=decode_s,
        encode_s=encode_s,
        rank_s=rank_s,
    )


def run_one_clip(args: argparse.Namespace) -> int:
    if args.end <= args.start or not np.isfinite(args.start) or not np.isfinite(args.end):
        raise SystemExit("Invalid interval: start must be < end and finite")
    clip_duration_s = float(args.end) - float(args.start)

    def _warmup():
        encoder = get_encoder(
            args.encoder,
            num_frames=getattr(args, "num_frames", None),
            device=getattr(args, "device", None),
            dtype=getattr(args, "dtype", "auto"),
            internvideo2_checkpoint_path=getattr(args, "internvideo2_checkpoint_path", None),
            internvideo2_checkpoint_repo=getattr(args, "internvideo2_checkpoint_repo", None),
        )
        rows = load_dictionary_rows(args.bank)
        labels = [row.label for row in rows]
        identity_keys = [row.identity_key(args.bank) for row in rows]
        text = load_or_build_text_cache(
            encoder.encoder_id,
            args.bank,
            PROMPT_ID,
            labels,
            lambda labs: encode_action_batch(labs, encoder, PROMPT_ID),
        )
        return encoder, labels, identity_keys, text

    packed, warmup_s = elapsed(_warmup)
    encoder, labels, identity_keys, text = packed

    def _decode():
        if not args.synthetic and not args.video:
            raise SystemExit("Provide --video or --synthetic")
        if args.synthetic:
            frames = synthetic_clip(num_frames=encoder.num_frames, seed=args.seed)
            interval = GoldInterval("synthetic", args.start, args.end, gold_action=args.gold or "")
            clip_id = f"synthetic:{args.start}:{args.end}"
            return frames, interval, clip_id
        try:
            frames = sample_gt_clip(args.video, args.start, args.end, num_frames=encoder.num_frames)
        except ClipDecodeError as exc:
            raise SystemExit(f"Unreadable clip: {exc}") from exc
        if frames is None:
            raise SystemExit("Invalid interval: start must be < end and finite")
        interval = GoldInterval(Path(args.video).stem, args.start, args.end, gold_action=args.gold or "")
        clip_id = f"{interval.video_id}:{args.start}:{args.end}"
        return frames, interval, clip_id

    decoded, decode_s = elapsed(_decode)
    frames, interval, clip_id = decoded
    ranking, encode_s = elapsed(
        lambda: _rank_frames(encoder, frames, text, labels, clip_id, identity_keys)
    )
    rank_s = 0.0
    timing = build_timing(
        clip_duration_s=clip_duration_s,
        warmup_s=warmup_s,
        decode_s=decode_s,
        encode_s=encode_s,
        rank_s=rank_s,
        encoder_id=encoder.encoder_id,
        frame_count=int(encoder.num_frames),
    )
    out_dir = _run_dir(args.run_id)
    out_path = out_dir / "ranking.json"
    payload = ranking.to_dict()
    payload["dictionary_id"] = args.bank
    payload["n_labels"] = len(labels)
    payload["encoder_id"] = encoder.encoder_id
    payload["prompt_id"] = PROMPT_ID
    payload["frame_count"] = encoder.num_frames
    payload.update(timing.to_dict())
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_run_meta(
        out_dir,
        encoder=encoder,
        mode="one-clip",
        dictionaries=[args.bank],
        extra={"clip_id": clip_id, "synthetic": bool(args.synthetic), **timing.to_dict()},
    )
    print(format_timing_line(timing))
    print(out_path)
    return 0


def _warmup_text_banks(encoder, dictionary_ids: list[str]) -> dict[str, tuple[list[str], list, object]]:
    caches: dict[str, tuple[list[str], list, object]] = {}
    for dictionary_id in dictionary_ids:
        rows = load_dictionary_rows(dictionary_id)
        labels = [row.label for row in rows]
        identity_keys = [row.identity_key(dictionary_id) for row in rows]
        text = load_or_build_text_cache(
            encoder.encoder_id,
            dictionary_id,
            PROMPT_ID,
            labels,
            lambda labs, _bank=dictionary_id: encode_action_batch(labs, encoder, PROMPT_ID),
        )
        caches[dictionary_id] = (labels, identity_keys, text)
    return caches


def run_slice(args: argparse.Namespace) -> int:
    encoder, warmup_load_s = elapsed(
        lambda: get_encoder(
            args.encoder,
            num_frames=getattr(args, "num_frames", None),
            device=getattr(args, "device", None),
            dtype=getattr(args, "dtype", "auto"),
            internvideo2_checkpoint_path=getattr(args, "internvideo2_checkpoint_path", None),
            internvideo2_checkpoint_repo=getattr(args, "internvideo2_checkpoint_repo", None),
        )
    )
    out_dir = _run_dir(args.run_id)
    skips: list[SkipEvent] = []
    slice_path = out_dir / "slice.json"
    try:
        if slice_path.is_file() and not args.rebuild_slice:
            frozen, videos = load_frozen_slice(slice_path)
        else:
            frozen, videos, build_skips = build_slice()
            skips.extend(build_skips)
            slice_path.write_text(json.dumps(frozen.to_dict(), indent=2), encoding="utf-8")
    except (DataAvailabilityError, FrozenSliceError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc

    banks = ["assembly101_coarse", "epic_kitchens_observed"]
    caches, warmup_cache_s = elapsed(lambda: _warmup_text_banks(encoder, banks))
    warmup_s = warmup_load_s + warmup_cache_s

    by_bank: dict[str, list[PredictionRecord]] = {
        "assembly101_coarse": [],
        "epic_kitchens_observed": [],
    }
    skipped_by_bank = {"assembly101_coarse": 0, "epic_kitchens_observed": 0}
    pred_path = out_dir / "predictions.jsonl"
    rank_path = out_dir / "rankings.jsonl"
    pred_handle = pred_path.open("w", encoding="utf-8")
    rank_handle = rank_path.open("w", encoding="utf-8")
    try:
        for source in videos:
            dictionary_id = BANK_FOR_DATASET[source.dataset]
            labels, identity_keys, text = caches[dictionary_id]
            intervals, interval_skips = load_intervals(source.labels_path, source.video_id)
            skips.extend(interval_skips)
            skipped_by_bank[dictionary_id] += len(interval_skips)
            for interval in intervals:
                if not _valid_interval(interval):
                    skips.append(
                        SkipEvent(
                            reason="invalid_interval",
                            video_id=interval.video_id,
                            start_sec=interval.start_sec,
                            end_sec=interval.end_sec,
                        )
                    )
                    skipped_by_bank[dictionary_id] += 1
                    continue
                try:
                    frames, decode_s = elapsed(
                        lambda iv=interval, src=source: sample_gt_clip(
                            src.media_path,
                            iv.start_sec,
                            iv.end_sec,
                            num_frames=encoder.num_frames,
                        )
                    )
                except ClipDecodeError as exc:
                    skips.append(
                        SkipEvent(
                            reason="unreadable_clip",
                            video_id=interval.video_id,
                            start_sec=interval.start_sec,
                            end_sec=interval.end_sec,
                            extra={"error": str(exc)},
                        )
                    )
                    skipped_by_bank[dictionary_id] += 1
                    continue
                if frames is None:
                    skips.append(
                        SkipEvent(
                            reason="invalid_interval",
                            video_id=interval.video_id,
                            start_sec=interval.start_sec,
                            end_sec=interval.end_sec,
                        )
                    )
                    skipped_by_bank[dictionary_id] += 1
                    continue
                clip_id = f"{interval.video_id}:{interval.start_sec}:{interval.end_sec}"
                ranking, encode_s = elapsed(
                    lambda fr=frames, cid=clip_id: _rank_frames(
                        encoder, fr, text, labels, cid, identity_keys
                    )
                )
                rank_s = 0.0
                record = prediction_from_ranking(
                    dataset=source.dataset,
                    interval=interval,
                    ranking=ranking,
                    dictionary_id=dictionary_id,
                    encoder=encoder,
                    inference_s=decode_s + encode_s + rank_s,
                    decode_s=decode_s,
                    encode_s=encode_s,
                    rank_s=rank_s,
                )
                by_bank[dictionary_id].append(record)
                pred_handle.write(json.dumps(record.to_dict()) + "\n")
                rank_handle.write(
                    json.dumps(
                        {
                            "clip_id": ranking.clip_id,
                            "dictionary_id": dictionary_id,
                            "pred_action": ranking.pred_action,
                            "labels": ranking.labels,
                            "cosine_distance": ranking.cosine_distance,
                            "cosine_similarity": ranking.cosine_similarity,
                            "identity_keys": ranking.identity_keys,
                            "inference_s": record.inference_s,
                            "decode_s": record.decode_s,
                            "encode_s": record.encode_s,
                            "rank_s": record.rank_s,
                        }
                    )
                    + "\n"
                )
    finally:
        pred_handle.close()
        rank_handle.close()

    write_skip_log(out_dir / "skip_log.jsonl", skips)
    a101 = compute_action_metrics(
        by_bank["assembly101_coarse"],
        "assembly101_coarse",
        n_skipped_intervals=skipped_by_bank["assembly101_coarse"],
        warmup_s=warmup_s,
    )
    epic = compute_action_metrics(
        by_bank["epic_kitchens_observed"],
        "epic_kitchens_observed",
        n_skipped_intervals=skipped_by_bank["epic_kitchens_observed"],
        warmup_s=warmup_s,
    )
    (out_dir / "metrics_assembly101_coarse.json").write_text(
        json.dumps(a101.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "metrics_epic_kitchens_observed.json").write_text(
        json.dumps(epic.to_dict(), indent=2), encoding="utf-8"
    )
    _write_run_meta(
        out_dir,
        encoder=encoder,
        mode="slice",
        dictionaries=["assembly101_coarse", "epic_kitchens_observed"],
        extra={
            "slice": frozen.to_dict(),
            "weight_updates": False,
            "n_skip_events": len(skips),
            "warmup_s": warmup_s,
        },
    )
    print(out_dir)
    return 0


def run_clips(args: argparse.Namespace) -> int:
    dictionary_id = args.bank
    clips_payload = json.loads(Path(args.clips_json).read_text(encoding="utf-8"))
    clip_rows = clips_payload.get("clips")
    if not isinstance(clip_rows, list) or not clip_rows:
        raise SystemExit(f"No clips found in {args.clips_json}")

    encoder, warmup_load_s = elapsed(
        lambda: get_encoder(
            args.encoder,
            num_frames=getattr(args, "num_frames", None),
            device=getattr(args, "device", None),
            dtype=getattr(args, "dtype", "auto"),
            internvideo2_checkpoint_path=getattr(args, "internvideo2_checkpoint_path", None),
            internvideo2_checkpoint_repo=getattr(args, "internvideo2_checkpoint_repo", None),
        )
    )
    rows = _load_bank_rows(dictionary_id, bank_file=getattr(args, "bank_file", None))
    labels = [row.label for row in rows]
    identity_keys = [row.identity_key(dictionary_id) for row in rows]
    text, warmup_cache_s = elapsed(
        lambda: load_or_build_text_cache(
            encoder.encoder_id,
            dictionary_id,
            PROMPT_ID,
            labels,
            lambda labs: encode_action_batch(labs, encoder, PROMPT_ID),
        )
    )
    warmup_s = warmup_load_s + warmup_cache_s

    out_dir = _run_dir(args.run_id)
    records: list[PredictionRecord] = []
    skips: list[SkipEvent] = []
    pred_path = out_dir / "predictions.jsonl"
    rank_path = out_dir / "rankings.jsonl"
    pred_handle = pred_path.open("w", encoding="utf-8")
    rank_handle = rank_path.open("w", encoding="utf-8")
    try:
        for raw in clip_rows:
            dataset = str(raw.get("dataset") or args.dataset)
            interval = GoldInterval(
                video_id=str(raw["video_id"]),
                start_sec=float(raw["start_sec"]),
                end_sec=float(raw["end_sec"]),
                gold_action=str(raw.get("gold_action") or raw.get("prompt_action_label") or ""),
                gold_verb_id=_optional_int(raw.get("gold_verb_id", raw.get("verb_id"))),
                gold_noun_id=_optional_int(raw.get("gold_noun_id", raw.get("noun_id"))),
            )
            if not _valid_interval(interval):
                skips.append(
                    SkipEvent(
                        reason="invalid_interval",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                    )
                )
                continue
            media_path = _find_clip_media(Path(args.media_root), interval.video_id)
            if media_path is None:
                skips.append(
                    SkipEvent(
                        reason="missing_media",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                        extra={"media_root": str(args.media_root)},
                    )
                )
                continue
            try:
                frames, decode_s = elapsed(
                    lambda path=media_path, iv=interval: sample_gt_clip(
                        path,
                        iv.start_sec,
                        iv.end_sec,
                        num_frames=encoder.num_frames,
                    )
                )
            except ClipDecodeError as exc:
                skips.append(
                    SkipEvent(
                        reason="unreadable_clip",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                        extra={"error": str(exc)},
                    )
                )
                continue
            if frames is None:
                skips.append(
                    SkipEvent(
                        reason="invalid_interval",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                    )
                )
                continue
            clip_id = f"{interval.video_id}:{interval.start_sec}:{interval.end_sec}"
            ranking, encode_s = elapsed(
                lambda fr=frames, cid=clip_id: _rank_frames(
                    encoder,
                    fr,
                    text,
                    labels,
                    cid,
                    identity_keys,
                )
            )
            rank_s = 0.0
            record = prediction_from_ranking(
                dataset=dataset,
                interval=interval,
                ranking=ranking,
                dictionary_id=dictionary_id,
                encoder=encoder,
                inference_s=decode_s + encode_s + rank_s,
                decode_s=decode_s,
                encode_s=encode_s,
                rank_s=rank_s,
            )
            records.append(record)
            pred_handle.write(json.dumps(record.to_dict()) + "\n")
            rank_handle.write(
                json.dumps(
                    {
                        "clip_id": ranking.clip_id,
                        "dictionary_id": dictionary_id,
                        "pred_action": ranking.pred_action,
                        "labels": ranking.labels,
                        "cosine_distance": ranking.cosine_distance,
                        "cosine_similarity": ranking.cosine_similarity,
                        "identity_keys": ranking.identity_keys,
                        "inference_s": record.inference_s,
                        "decode_s": record.decode_s,
                        "encode_s": record.encode_s,
                        "rank_s": record.rank_s,
                    }
                )
                + "\n"
            )
    finally:
        pred_handle.close()
        rank_handle.close()

    write_skip_log(out_dir / "skip_log.jsonl", skips)
    report = compute_action_metrics(
        records,
        dictionary_id,
        n_skipped_intervals=len(skips),
        warmup_s=warmup_s,
    )
    (out_dir / f"metrics_{dictionary_id}.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    meta_extra = {
        "clips_json": str(Path(args.clips_json)),
        "media_root": str(Path(args.media_root)),
        "sampling_protocol": "pre-trimmed intervals from clips_json",
        "weight_updates": False,
        "n_skip_events": len(skips),
        "warmup_s": warmup_s,
        **_encoder_runtime_meta(args),
    }
    if getattr(args, "bank_file", None):
        meta_extra["custom_taxonomy_files"] = {dictionary_id: str(Path(args.bank_file))}
    _write_run_meta(
        out_dir,
        encoder=encoder,
        mode="clips",
        dictionaries=[dictionary_id],
        extra=meta_extra,
    )
    print(out_dir)
    return 0


def run_grouped_clips(args: argparse.Namespace) -> int:
    dictionary_id = args.bank
    clips_payload = json.loads(Path(args.clips_json).read_text(encoding="utf-8"))
    clip_rows = clips_payload.get("clips")
    if not isinstance(clip_rows, list) or not clip_rows:
        raise SystemExit(f"No clips found in {args.clips_json}")

    encoder, warmup_load_s = elapsed(
        lambda: get_encoder(
            args.encoder,
            num_frames=getattr(args, "num_frames", None),
            device=getattr(args, "device", None),
            dtype=getattr(args, "dtype", "auto"),
            internvideo2_checkpoint_path=getattr(args, "internvideo2_checkpoint_path", None),
            internvideo2_checkpoint_repo=getattr(args, "internvideo2_checkpoint_repo", None),
        )
    )
    rows = _load_bank_rows(dictionary_id, bank_file=getattr(args, "bank_file", None))
    labels = [row.label for row in rows]
    identity_keys = [row.identity_key(dictionary_id) for row in rows]
    text, warmup_cache_s = elapsed(
        lambda: load_or_build_text_cache(
            encoder.encoder_id,
            dictionary_id,
            PROMPT_ID,
            labels,
            lambda labs: encode_action_batch(labs, encoder, PROMPT_ID),
        )
    )
    warmup_s = warmup_load_s + warmup_cache_s

    out_dir = _run_dir(args.run_id)
    records: list[PredictionRecord] = []
    skips: list[SkipEvent] = []
    grouped: dict[str, list[tuple[str, str, GoldInterval]]] = defaultdict(list)
    for raw in clip_rows:
        dataset = str(raw.get("dataset") or args.dataset)
        interval = GoldInterval(
            video_id=str(raw["video_id"]),
            start_sec=float(raw["start_sec"]),
            end_sec=float(raw["end_sec"]),
            gold_action=str(raw.get("gold_action") or raw.get("prompt_action_label") or ""),
            gold_verb_id=_optional_int(raw.get("gold_verb_id", raw.get("verb_id"))),
            gold_noun_id=_optional_int(raw.get("gold_noun_id", raw.get("noun_id"))),
        )
        if not _valid_interval(interval):
            skips.append(
                SkipEvent(
                    reason="invalid_interval",
                    video_id=interval.video_id,
                    start_sec=interval.start_sec,
                    end_sec=interval.end_sec,
                )
            )
            continue
        media_path = _media_path_for_clip(raw, Path(args.media_root), interval.video_id)
        if media_path is None:
            skips.append(
                SkipEvent(
                    reason="missing_media",
                    video_id=interval.video_id,
                    start_sec=interval.start_sec,
                    end_sec=interval.end_sec,
                    extra={"media_root": str(args.media_root)},
                )
            )
            continue
        clip_id = str(raw.get("clip_id") or f"{interval.video_id}:{interval.start_sec}:{interval.end_sec}")
        grouped[str(media_path)].append((clip_id, dataset, interval))

    pred_path = out_dir / "predictions.jsonl"
    rank_path = out_dir / "rankings.jsonl"
    pred_handle = pred_path.open("w", encoding="utf-8")
    rank_handle = rank_path.open("w", encoding="utf-8")
    try:
        for media_path, items in sorted(grouped.items()):
            try:
                frames_by_clip, decode_s = elapsed(
                    lambda path=Path(media_path), group=items: _sample_video_group_once(
                        path,
                        [(clip_id, interval) for clip_id, _, interval in group],
                        num_frames=encoder.num_frames,
                    )
                )
            except ClipDecodeError as exc:
                for _, _, interval in items:
                    skips.append(
                        SkipEvent(
                            reason="unreadable_media_group",
                            video_id=interval.video_id,
                            start_sec=interval.start_sec,
                            end_sec=interval.end_sec,
                            extra={"media_path": media_path, "error": str(exc)},
                        )
                    )
                continue
            ready = [
                (clip_id, dataset, interval, frames_by_clip.get(clip_id))
                for clip_id, dataset, interval in items
            ]
            ready = [(clip_id, dataset, interval, frames) for clip_id, dataset, interval, frames in ready if frames is not None]
            missing = {clip_id for clip_id, _, _ in items} - {clip_id for clip_id, _, _, _ in ready}
            for clip_id, _, interval in items:
                if clip_id in missing:
                    skips.append(
                        SkipEvent(
                            reason="unreadable_clip",
                            video_id=interval.video_id,
                            start_sec=interval.start_sec,
                            end_sec=interval.end_sec,
                            extra={"media_path": media_path},
                        )
                    )
            decode_per_clip_s = decode_s / max(1, len(ready))
            for offset in range(0, len(ready), args.batch_size):
                batch = ready[offset : offset + args.batch_size]
                frame_batch = np.stack([frames for _, _, _, frames in batch], axis=0)
                started = time.perf_counter()
                rankings = _rank_frame_batch(
                    encoder,
                    frame_batch,
                    text,
                    labels,
                    [clip_id for clip_id, _, _, _ in batch],
                    identity_keys,
                )
                encode_s = time.perf_counter() - started
                encode_per_clip_s = encode_s / max(1, len(batch))
                for (clip_id, dataset, interval, _), ranking in zip(batch, rankings, strict=True):
                    record = prediction_from_ranking(
                        dataset=dataset,
                        interval=interval,
                        ranking=ranking,
                        dictionary_id=dictionary_id,
                        encoder=encoder,
                        inference_s=decode_per_clip_s + encode_per_clip_s,
                        decode_s=decode_per_clip_s,
                        encode_s=encode_per_clip_s,
                        rank_s=0.0,
                    )
                    records.append(record)
                    pred_handle.write(json.dumps(record.to_dict()) + "\n")
                    rank_handle.write(
                        json.dumps(
                            {
                                "clip_id": ranking.clip_id,
                                "dictionary_id": dictionary_id,
                                "pred_action": ranking.pred_action,
                                "labels": ranking.labels,
                                "cosine_distance": ranking.cosine_distance,
                                "cosine_similarity": ranking.cosine_similarity,
                                "identity_keys": ranking.identity_keys,
                                "inference_s": record.inference_s,
                                "decode_s": record.decode_s,
                                "encode_s": record.encode_s,
                                "rank_s": record.rank_s,
                            }
                        )
                        + "\n"
                    )
    finally:
        pred_handle.close()
        rank_handle.close()

    write_skip_log(out_dir / "skip_log.jsonl", skips)
    report = compute_action_metrics(
        records,
        dictionary_id,
        n_skipped_intervals=len(skips),
        warmup_s=warmup_s,
    )
    (out_dir / f"metrics_{dictionary_id}.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    meta_extra = {
        "clips_json": str(Path(args.clips_json)),
        "media_root": str(Path(args.media_root)),
        "sampling_protocol": "grouped media decode; each source media path is decoded once",
        "weight_updates": False,
        "n_input_clips": len(clip_rows),
        "n_scored_clips": len(records),
        "n_media_groups": len(grouped),
        "n_skip_events": len(skips),
        "batch_size": args.batch_size,
        "warmup_s": warmup_s,
        "source_task": clips_payload.get("task"),
        **_encoder_runtime_meta(args),
    }
    if getattr(args, "bank_file", None):
        meta_extra["custom_taxonomy_files"] = {dictionary_id: str(Path(args.bank_file))}
    _write_run_meta(
        out_dir,
        encoder=encoder,
        mode="grouped-clips",
        dictionaries=[dictionary_id],
        extra=meta_extra,
    )
    print(out_dir)
    return 0


def _media_path_for_clip(raw: dict, media_root: Path, video_id: str) -> Path | None:
    raw_media = raw.get("media_path")
    if raw_media:
        path = Path(str(raw_media))
        if path.is_file():
            return path
        rooted = media_root / path
        if rooted.is_file():
            return rooted
    return _find_clip_media(media_root, video_id)


def _rank_frame_batch(encoder, frames: np.ndarray, text, labels: list[str], clip_ids: list[str], identity_keys):
    scorer = getattr(encoder, "score_clip_texts", None)
    if callable(scorer):
        similarities = scorer(frames, text)
        return [
            ranking_from_similarities(row, labels, clip_id=clip_id, identity_keys=identity_keys)
            for row, clip_id in zip(similarities, clip_ids, strict=True)
        ]
    clip_embeddings = encode_clip_batch(frames, encoder)
    return [
        rank_actions(clip_embedding, text, labels, clip_id=clip_id, identity_keys=identity_keys)
        for clip_embedding, clip_id in zip(clip_embeddings, clip_ids, strict=True)
    ]


def _sample_video_group_once(
    path: Path,
    clips: list[tuple[str, GoldInterval]],
    *,
    num_frames: int,
) -> dict[str, np.ndarray]:
    try:
        import av
    except ImportError as exc:
        raise ClipDecodeError("PyAV is required for grouped clip decoding") from exc
    if not path.is_file():
        raise ClipDecodeError(f"Unreadable clip group: missing file {path}")
    targets: list[tuple[float, str, int]] = []
    for clip_id, interval in clips:
        for frame_index, target_sec in enumerate(np.linspace(interval.start_sec, interval.end_sec, num_frames)):
            targets.append((float(target_sec), clip_id, frame_index))
    targets.sort(key=lambda row: row[0])
    if not targets:
        return {}

    output = {clip_id: np.zeros((num_frames, 3, 224, 224), dtype=np.float32) for clip_id, _ in clips}
    filled = {clip_id: np.zeros(num_frames, dtype=bool) for clip_id, _ in clips}
    target_index = 0
    last_chw: np.ndarray | None = None
    container = None
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if target_index >= len(targets):
                break
            ts = _frame_time(frame, stream)
            if ts < targets[target_index][0]:
                continue
            last_chw = _resize_rgb_to_chw(frame.to_ndarray(format="rgb24"))
            while target_index < len(targets) and ts >= targets[target_index][0]:
                _, clip_id, frame_index = targets[target_index]
                output[clip_id][frame_index] = last_chw
                filled[clip_id][frame_index] = True
                target_index += 1
    except Exception as exc:  # noqa: BLE001
        raise ClipDecodeError(str(exc)) from exc
    finally:
        if container is not None:
            with contextlib.suppress(Exception):
                container.close()

    if target_index < len(targets) and last_chw is not None:
        while target_index < len(targets):
            _, clip_id, frame_index = targets[target_index]
            output[clip_id][frame_index] = last_chw
            filled[clip_id][frame_index] = True
            target_index += 1
    return {clip_id: frames for clip_id, frames in output.items() if bool(filled[clip_id].all())}


def _frame_time(frame, stream) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None:
        return 0.0
    return float(frame.pts * stream.time_base)


def _resize_rgb_to_chw(rgb: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    ys = np.linspace(0, src_h - 1, height).astype(np.int32)
    xs = np.linspace(0, src_w - 1, width).astype(np.int32)
    resized = rgb[ys][:, xs]
    if resized.shape[-1] == 3:
        chw = np.transpose(resized, (2, 0, 1))
    else:
        chw = np.repeat(resized[..., None], 3, axis=2).transpose(2, 0, 1)
    return chw.astype(np.float32) / 255.0


class _RandomEncoderMeta:
    def __init__(self, seed: int):
        self.encoder_id = f"random-uniform-seed-{seed}"
        self.num_frames = 0


def run_random_clips(args: argparse.Namespace) -> int:
    dictionary_id = args.bank
    clips_payload = json.loads(Path(args.clips_json).read_text(encoding="utf-8"))
    clip_rows = clips_payload.get("clips")
    if not isinstance(clip_rows, list) or not clip_rows:
        raise SystemExit(f"No clips found in {args.clips_json}")

    rows = _load_bank_rows(dictionary_id, bank_file=getattr(args, "bank_file", None))
    labels = [row.label for row in rows]
    identity_keys = [row.identity_key(dictionary_id) for row in rows]
    rng = np.random.default_rng(args.seed)
    encoder = _RandomEncoderMeta(args.seed)

    out_dir = _run_dir(args.run_id)
    records: list[PredictionRecord] = []
    skips: list[SkipEvent] = []
    pred_path = out_dir / "predictions.jsonl"
    rank_path = out_dir / "rankings.jsonl"
    pred_handle = pred_path.open("w", encoding="utf-8")
    rank_handle = rank_path.open("w", encoding="utf-8")
    try:
        for raw in clip_rows:
            dataset = str(raw.get("dataset") or args.dataset)
            interval = GoldInterval(
                video_id=str(raw["video_id"]),
                start_sec=float(raw["start_sec"]),
                end_sec=float(raw["end_sec"]),
                gold_action=str(raw.get("gold_action") or raw.get("prompt_action_label") or ""),
                gold_verb_id=_optional_int(raw.get("gold_verb_id", raw.get("verb_id"))),
                gold_noun_id=_optional_int(raw.get("gold_noun_id", raw.get("noun_id"))),
            )
            if not _valid_interval(interval):
                skips.append(
                    SkipEvent(
                        reason="invalid_interval",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                    )
                )
                continue
            ranking = ranking_from_similarities(
                rng.random(len(labels), dtype=np.float32),
                labels,
                clip_id=f"{interval.video_id}:{interval.start_sec}:{interval.end_sec}",
                identity_keys=identity_keys,
            )
            record = prediction_from_ranking(
                dataset=dataset,
                interval=interval,
                ranking=ranking,
                dictionary_id=dictionary_id,
                encoder=encoder,
                inference_s=0.0,
                decode_s=0.0,
                encode_s=0.0,
                rank_s=0.0,
                prompt_id="random_uniform",
            )
            records.append(record)
            pred_handle.write(json.dumps(record.to_dict()) + "\n")
            rank_handle.write(
                json.dumps(
                    {
                        "clip_id": ranking.clip_id,
                        "dictionary_id": dictionary_id,
                        "pred_action": ranking.pred_action,
                        "labels": ranking.labels,
                        "cosine_distance": ranking.cosine_distance,
                        "cosine_similarity": ranking.cosine_similarity,
                        "identity_keys": ranking.identity_keys,
                        "inference_s": record.inference_s,
                        "decode_s": record.decode_s,
                        "encode_s": record.encode_s,
                        "rank_s": record.rank_s,
                    }
                )
                + "\n"
            )
    finally:
        pred_handle.close()
        rank_handle.close()

    write_skip_log(out_dir / "skip_log.jsonl", skips)
    report = compute_action_metrics(records, dictionary_id, n_skipped_intervals=len(skips), warmup_s=0.0)
    (out_dir / f"metrics_{dictionary_id}.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    summary = _random_baseline_summary(
        n_labels=len(labels),
        n_clips=len(records),
        n_trials=args.n_trials,
        seed=args.seed,
    )
    (out_dir / f"random_summary_{dictionary_id}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    meta_extra = {
        "clips_json": str(Path(args.clips_json)),
        "dataset": args.dataset,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "n_labels": len(labels),
        "weight_updates": False,
        "n_skip_events": len(skips),
        "random_policy": "iid uniform random score per action label and clip",
    }
    if getattr(args, "bank_file", None):
        meta_extra["custom_taxonomy_files"] = {dictionary_id: str(Path(args.bank_file))}
    _write_run_meta(
        out_dir,
        encoder=encoder,
        mode="random-clips",
        dictionaries=[dictionary_id],
        extra=meta_extra,
    )
    print(out_dir)
    return 0


def _random_baseline_summary(*, n_labels: int, n_clips: int, n_trials: int, seed: int) -> dict:
    if n_labels < 1:
        raise ValueError("n_labels must be positive")
    if n_clips < 1:
        raise ValueError("n_clips must be positive")
    rng = np.random.default_rng(seed + 1)
    ranks = rng.integers(1, n_labels + 1, size=(n_trials, n_clips), dtype=np.int32)
    metrics = {
        "action_top1": (ranks <= 1).mean(axis=1),
        "action_top3": (ranks <= min(3, n_labels)).mean(axis=1),
        "action_top5": (ranks <= min(5, n_labels)).mean(axis=1),
        "action_top20": (ranks <= min(20, n_labels)).mean(axis=1),
        "action_top50": (ranks <= min(50, n_labels)).mean(axis=1),
    }
    return {
        "seed": seed,
        "n_trials": n_trials,
        "n_clips": n_clips,
        "n_labels": n_labels,
        "closed_form_expected_accuracy": {
            "action_top1": 1 / n_labels,
            "action_top3": min(3, n_labels) / n_labels,
            "action_top5": min(5, n_labels) / n_labels,
            "action_top20": min(20, n_labels) / n_labels,
            "action_top50": min(50, n_labels) / n_labels,
        },
        "monte_carlo_accuracy": {
            name: _distribution_summary(values) for name, values in metrics.items()
        },
    }


def _distribution_summary(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _load_bank_rows(dictionary_id: str, *, bank_file: str | None = None) -> list[DictionaryRow]:
    if not bank_file:
        return load_dictionary_rows(dictionary_id)
    path = Path(bank_file)
    if not path.is_file():
        raise FileNotFoundError(f"Custom bank file missing: {path}")
    if path.suffix.lower() == ".txt":
        rows = [DictionaryRow(label=line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty custom bank: {path}")
        return _dedupe_rows(rows)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        label_col = _pick_label_column(fieldnames)
        if label_col is None:
            raise ValueError(f"Custom bank {path} needs label/action_cls/prompt_action_label column")
        rows = [
            DictionaryRow(
                label=(raw.get(label_col) or "").strip(),
                verb_id=_optional_int(raw.get("verb_id")),
                noun_id=_optional_int(raw.get("noun_id")),
            )
            for raw in reader
            if (raw.get(label_col) or "").strip()
        ]
    if not rows:
        raise ValueError(f"Empty custom bank: {path}")
    return _dedupe_rows(rows)


def _pick_label_column(fieldnames: set[str]) -> str | None:
    for name in ("label", "action_cls", "prompt_action_label", "gold_action"):
        if name in fieldnames:
            return name
    return None


def _dedupe_rows(rows: list[DictionaryRow]) -> list[DictionaryRow]:
    out: list[DictionaryRow] = []
    seen: set[str] = set()
    for row in rows:
        if row.label in seen:
            continue
        seen.add(row.label)
        out.append(row)
    return out


def _find_clip_media(media_root: Path, video_id: str) -> Path | None:
    candidates = [
        media_root / video_id,
        media_root / f"{video_id}.mp4",
        media_root / f"{video_id}.MP4",
        media_root / f"{video_id}.webm",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(media_root.rglob(f"{video_id}.*")) if media_root.is_dir() else []
    for path in matches:
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}:
            return path
    return None


def run_faes_clips(args: argparse.Namespace) -> int:
    dictionary_id = args.bank
    clips_payload = json.loads(Path(args.clips_json).read_text(encoding="utf-8"))
    clip_rows = clips_payload.get("clips")
    if not isinstance(clip_rows, list) or not clip_rows:
        raise SystemExit(f"No clips found in {args.clips_json}")

    encoder, warmup_load_s = elapsed(
        lambda: get_encoder(
            args.encoder,
            num_frames=getattr(args, "num_frames", None),
            device=getattr(args, "device", None),
            dtype=getattr(args, "dtype", "auto"),
            internvideo2_checkpoint_path=getattr(args, "internvideo2_checkpoint_path", None),
            internvideo2_checkpoint_repo=getattr(args, "internvideo2_checkpoint_repo", None),
        )
    )
    rows = load_dictionary_rows(dictionary_id)
    labels = [row.label for row in rows]
    identity_keys = [row.identity_key(dictionary_id) for row in rows]
    text, warmup_cache_s = elapsed(
        lambda: load_or_build_text_cache(
            encoder.encoder_id,
            dictionary_id,
            PROMPT_ID,
            labels,
            lambda labs: encode_action_batch(labs, encoder, PROMPT_ID),
        )
    )
    warmup_s = warmup_load_s + warmup_cache_s

    out_dir = _run_dir(args.run_id)
    records: list[PredictionRecord] = []
    skips: list[SkipEvent] = []
    pred_path = out_dir / "predictions.jsonl"
    rank_path = out_dir / "rankings.jsonl"
    debug_path = out_dir / "faes_debug.jsonl"
    pred_handle = pred_path.open("w", encoding="utf-8")
    rank_handle = rank_path.open("w", encoding="utf-8")
    debug_handle = debug_path.open("w", encoding="utf-8")
    try:
        for raw in clip_rows:
            interval = GoldInterval(
                video_id=str(raw["video_id"]),
                start_sec=float(raw["start_sec"]),
                end_sec=float(raw["end_sec"]),
                gold_action=str(raw.get("gold_action") or ""),
                gold_verb_id=raw.get("gold_verb_id"),
                gold_noun_id=raw.get("gold_noun_id"),
            )
            dataset = str(raw.get("dataset") or "assembly101")
            if dataset != "assembly101":
                skips.append(
                    SkipEvent(
                        reason="unsupported_dataset_for_faes_clips",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                        extra={"dataset": dataset},
                    )
                )
                continue
            media_path = Path(args.media_root) / f"{interval.video_id}.mp4"
            if not _valid_interval(interval):
                skips.append(
                    SkipEvent(
                        reason="invalid_interval",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                    )
                )
                continue
            try:
                frames, decode_s = elapsed(
                    lambda path=media_path, iv=interval: sample_gt_clip(
                        path,
                        iv.start_sec,
                        iv.end_sec,
                        num_frames=encoder.num_frames,
                    )
                )
            except ClipDecodeError as exc:
                skips.append(
                    SkipEvent(
                        reason="unreadable_clip",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                        extra={"error": str(exc)},
                    )
                )
                continue
            if frames is None:
                skips.append(
                    SkipEvent(
                        reason="invalid_interval",
                        video_id=interval.video_id,
                        start_sec=interval.start_sec,
                        end_sec=interval.end_sec,
                    )
                )
                continue

            clip_id = f"{interval.video_id}:{interval.start_sec}:{interval.end_sec}"
            packed, encode_s = elapsed(
                lambda fr=frames, cid=clip_id: score_faes_clip(
                    encoder,
                    fr,
                    text,
                    labels,
                    clip_id=cid,
                    identity_keys=identity_keys,
                    aggregation=args.aggregation,
                )
            )
            ranking, debug, frame_scores = packed
            rank_s = 0.0
            record = prediction_from_ranking(
                dataset=dataset,
                interval=interval,
                ranking=ranking,
                dictionary_id=dictionary_id,
                encoder=encoder,
                inference_s=decode_s + encode_s + rank_s,
                decode_s=decode_s,
                encode_s=encode_s,
                rank_s=rank_s,
                prompt_id=f"{PROMPT_ID}+faes_{args.aggregation}",
            )
            records.append(record)
            pred_handle.write(json.dumps(record.to_dict()) + "\n")
            rank_handle.write(
                json.dumps(
                    {
                        "clip_id": ranking.clip_id,
                        "dictionary_id": dictionary_id,
                        "pred_action": ranking.pred_action,
                        "labels": ranking.labels,
                        "cosine_distance": ranking.cosine_distance,
                        "cosine_similarity": ranking.cosine_similarity,
                        "identity_keys": ranking.identity_keys,
                        "inference_s": record.inference_s,
                        "decode_s": record.decode_s,
                        "encode_s": record.encode_s,
                        "rank_s": record.rank_s,
                    }
                )
                + "\n"
            )
            debug_handle.write(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "gold_action": interval.gold_action,
                        "gold_rank": record.gold_rank,
                        "pred_action": record.pred_action,
                        **debug.to_dict(),
                        "frame_top1_labels": _frame_top_labels(frame_scores, labels),
                    }
                )
                + "\n"
            )
    finally:
        pred_handle.close()
        rank_handle.close()
        debug_handle.close()

    write_skip_log(out_dir / "skip_log.jsonl", skips)
    report = compute_action_metrics(
        records,
        dictionary_id,
        n_skipped_intervals=len(skips),
        warmup_s=warmup_s,
    )
    (out_dir / f"metrics_{dictionary_id}.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    _write_run_meta(
        out_dir,
        encoder=encoder,
        mode="faes-clips",
        dictionaries=[dictionary_id],
        extra={
            "clips_json": str(Path(args.clips_json)),
            "media_root": str(Path(args.media_root)),
            "aggregation": args.aggregation,
            "sampling_protocol": "intra-interval uniform frames only",
            "weight_updates": False,
            "n_skip_events": len(skips),
            "warmup_s": warmup_s,
        },
    )
    print(out_dir)
    return 0


def _frame_top_labels(frame_scores: np.ndarray, labels: list[str]) -> list[str]:
    scores = np.asarray(frame_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("frame_scores must be [T, M]")
    top = np.argmax(scores, axis=1)
    return [labels[int(index)] for index in top]


def run_download_slice(args: argparse.Namespace) -> int:
    from action_ranker.download_slice import run_download

    try:
        run_download(dry_run=bool(args.dry_run))
    except DataAvailabilityError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def run_internvideo2_preflight(args: argparse.Namespace) -> int:
    from action_ranker.internvideo2_encoder import internvideo2_preflight

    model_id = INTERNVIDEO2_MODEL_IDS[args.model]
    report = internvideo2_preflight(model_id, dtype=args.dtype, device=args.device)
    if getattr(args, "internvideo2_checkpoint_path", None):
        report["checkpoint_path"] = str(Path(args.internvideo2_checkpoint_path))
        report["checkpoint_path_exists"] = Path(args.internvideo2_checkpoint_path).is_file()
    if getattr(args, "internvideo2_checkpoint_repo", None):
        report["checkpoint_repo"] = args.internvideo2_checkpoint_repo
    print(json.dumps(report, indent=2))
    return 0 if report.get("runnable_on_gpu") or args.device == "cpu" else 1


def _add_encoder_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--internvideo2-checkpoint-path",
        default=None,
        help="Local InternVideo2 1B .pt checkpoint; bypasses gated HF checkpoint download.",
    )
    parser.add_argument(
        "--internvideo2-checkpoint-repo",
        default=None,
        help="HF repo containing InternVideo2-stage2_1b-224p-f4.pt; default is OpenGVLab gated repo.",
    )


def _valid_interval(interval: GoldInterval) -> bool:
    return np.isfinite(interval.start_sec) and np.isfinite(interval.end_sec) and interval.end_sec > interval.start_sec


def _run_dir(run_id: str | None) -> Path:
    name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path = RUNS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_run_meta(out_dir: Path, encoder, mode: str, dictionaries: list[str], extra: dict) -> None:
    extra_payload = dict(extra)
    custom_taxonomy_files = extra_payload.pop("custom_taxonomy_files", {})
    taxonomy_files = {}
    for did in dictionaries:
        try:
            taxonomy_files[did] = str(taxonomy_path(did))
        except KeyError:
            pass
    taxonomy_files.update(custom_taxonomy_files)
    meta = {
        "mode": mode,
        "encoder_id": encoder.encoder_id,
        "prompt_id": PROMPT_ID,
        "frame_count": encoder.num_frames,
        "weight_updates": False,
        "taxonomy_files": taxonomy_files,
        "headline_metrics": ["action_top1", "action_top3", "action_top5", "action_macro_f1"],
        "constitution": "inference-only frozen video-text ranking",
        **extra_payload,
    }
    load_info = getattr(encoder, "load_info", None)
    if load_info is not None:
        meta["encoder_load_info"] = load_info
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _encoder_runtime_meta(args: argparse.Namespace) -> dict:
    keys = (
        "device",
        "dtype",
        "internvideo2_checkpoint_path",
        "internvideo2_checkpoint_repo",
    )
    return {key: value for key in keys if (value := getattr(args, key, None)) is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="action-ranker")
    sub = parser.add_subparsers(dest="mode", required=True)

    one = sub.add_parser("one-clip")
    one.add_argument("--encoder", choices=ENCODER_CHOICES, default="stub")
    one.add_argument("--run-id", default=None)
    one.add_argument("--bank", default="assembly101_coarse")
    one.add_argument("--video", default=None)
    one.add_argument("--start", type=float, default=0.0)
    one.add_argument("--end", type=float, default=1.0)
    one.add_argument("--gold", default="")
    one.add_argument("--synthetic", action="store_true")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--num-frames", type=int, default=None)
    _add_encoder_runtime_args(one)
    one.set_defaults(func=run_one_clip)

    sl = sub.add_parser("slice")
    sl.add_argument("--encoder", choices=ENCODER_CHOICES, default="stub")
    sl.add_argument("--run-id", default=None)
    sl.add_argument("--rebuild-slice", action="store_true")
    sl.add_argument("--num-frames", type=int, default=None)
    _add_encoder_runtime_args(sl)
    sl.set_defaults(func=run_slice)

    clips = sub.add_parser("clips")
    clips.add_argument("--encoder", choices=ENCODER_CHOICES, default="xclip")
    clips.add_argument("--run-id", default=None)
    clips.add_argument("--clips-json", required=True)
    clips.add_argument("--media-root", required=True)
    clips.add_argument("--bank", default="assembly101_coarse")
    clips.add_argument("--bank-file", default=None)
    clips.add_argument("--dataset", default="assembly101")
    clips.add_argument("--num-frames", type=int, default=None)
    _add_encoder_runtime_args(clips)
    clips.set_defaults(func=run_clips)

    grouped = sub.add_parser("grouped-clips")
    grouped.add_argument("--encoder", choices=ENCODER_CHOICES, default="xclip")
    grouped.add_argument("--run-id", default=None)
    grouped.add_argument("--clips-json", required=True)
    grouped.add_argument("--media-root", required=True)
    grouped.add_argument("--bank", default="assembly101_coarse")
    grouped.add_argument("--bank-file", default=None)
    grouped.add_argument("--dataset", default="assembly101")
    grouped.add_argument("--num-frames", type=int, default=None)
    grouped.add_argument("--batch-size", type=int, default=8)
    _add_encoder_runtime_args(grouped)
    grouped.set_defaults(func=run_grouped_clips)

    random = sub.add_parser("random-clips")
    random.add_argument("--run-id", default=None)
    random.add_argument("--clips-json", required=True)
    random.add_argument("--bank", default="assembly101_coarse")
    random.add_argument("--bank-file", default=None)
    random.add_argument("--dataset", default="assembly101")
    random.add_argument("--seed", type=int, default=0)
    random.add_argument("--n-trials", type=int, default=10000)
    random.set_defaults(func=run_random_clips)

    faes = sub.add_parser("faes-clips")
    faes.add_argument("--encoder", choices=["xclip", "xclip-zs"], default="xclip")
    faes.add_argument("--run-id", default=None)
    faes.add_argument("--clips-json", default=str(RUNS_ROOT / "a101-10clips-no-inspect" / "clips.json"))
    faes.add_argument("--media-root", default=str(REPO_ROOT / "data" / "raw" / "assembly101"))
    faes.add_argument("--bank", default="assembly101_coarse")
    faes.add_argument("--num-frames", type=int, default=None)
    faes.add_argument("--aggregation", choices=["mean", "max", "top2_mean", "lse"], default="top2_mean")
    faes.set_defaults(func=run_faes_clips)

    dl = sub.add_parser("download-slice")
    dl.add_argument("--dry-run", action="store_true")
    dl.set_defaults(func=run_download_slice)

    iv2 = sub.add_parser("internvideo2-preflight")
    iv2.add_argument("--model", choices=sorted(INTERNVIDEO2_MODEL_IDS), default="internvideo2-1b")
    iv2.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    iv2.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    iv2.add_argument("--internvideo2-checkpoint-path", default=None)
    iv2.add_argument("--internvideo2-checkpoint-repo", default=None)
    iv2.set_defaults(func=run_internvideo2_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
