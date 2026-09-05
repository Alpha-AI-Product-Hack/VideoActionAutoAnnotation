from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from action_ranker.encode_actions import encode_action_batch
from action_ranker.metrics import compute_action_metrics
from action_ranker.prompts import PROMPT_ID
from action_ranker.rank import gold_rank_1based, l2_normalize, ranking_from_similarities
from action_ranker.slice import load_intervals
from action_ranker.taxonomies import REPO_ROOT
from action_ranker.types import PredictionRecord
from action_ranker.xclip_encoder import XClipEncoder


@dataclass(frozen=True)
class VerbClip:
    clip_id: str
    dataset: str
    video_id: str
    media_path: str
    start_sec: float
    end_sec: float
    gold_action: str
    full_gold_action: str
    source_label_file: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_clips = _load_local_verb_clips()
    counts_all = Counter(clip.gold_action for clip in all_clips)
    candidate_verbs = sorted(
        verb
        for verb, count in counts_all.items()
        if count >= args.min_support and not verb.startswith("attempt to ")
    )
    candidate_clips = [clip for clip in all_clips if clip.gold_action in candidate_verbs]
    counts = Counter(clip.gold_action for clip in candidate_clips)
    by_dataset = Counter(clip.dataset for clip in candidate_clips)

    started = time.perf_counter()
    encoder = XClipEncoder(num_frames=args.num_frames)
    text_embeddings = encode_action_batch(candidate_verbs, encoder, PROMPT_ID)
    text_norm = l2_normalize(text_embeddings)
    distance = 1.0 - (text_norm @ text_norm.T)
    label_to_index = {label: index for index, label in enumerate(candidate_verbs)}

    scored = _score_clips(
        encoder=encoder,
        labels=candidate_verbs,
        text_embeddings=text_embeddings,
        clips=candidate_clips,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
    )
    if not scored:
        raise SystemExit("No clips were scored")

    scored_by_id = {row["clip"].clip_id: row for row in scored}
    available_clips = [row["clip"] for row in scored]
    best, top_candidates = _select_best_subset(
        clips=available_clips,
        scored_by_id=scored_by_id,
        labels=candidate_verbs,
        label_to_index=label_to_index,
        distance=distance,
        target_clips=args.target_clips,
        min_verbs=args.min_verbs,
        min_pair_distance=args.min_pair_distance,
        require_epic=not args.allow_a101_only,
    )
    if best is None:
        raise SystemExit("No subset satisfied the selection constraints")

    dictionary_id = f"combined_verb{len(best['labels'])}_clipdistance_f1max"
    selected_records = _prediction_records_for_subset(
        clips=best["clips"],
        scored_by_id=scored_by_id,
        labels=best["labels"],
        label_to_index=label_to_index,
        dictionary_id=dictionary_id,
        encoder_id=encoder.encoder_id,
        num_frames=args.num_frames,
    )
    report = compute_action_metrics(selected_records, dictionary_id)
    random_records, random_summary = _random_baseline(
        clips=best["clips"],
        labels=best["labels"],
        dictionary_id=dictionary_id,
        seed=args.seed,
        n_trials=args.random_trials,
    )
    random_report = compute_action_metrics(random_records, dictionary_id)

    _write_artifacts(
        run_dir=run_dir,
        args=args,
        all_counts=counts_all,
        candidate_counts=counts,
        candidate_by_dataset=by_dataset,
        candidate_verbs=candidate_verbs,
        distance=distance,
        scored=scored,
        best=best,
        top_candidates=top_candidates,
        dictionary_id=dictionary_id,
        selected_records=selected_records,
        report=report.to_dict(),
        random_records=random_records,
        random_report=random_report.to_dict(),
        random_summary=random_summary,
        elapsed_s=time.perf_counter() - started,
        encoder_id=encoder.encoder_id,
    )
    print(run_dir)
    print(json.dumps(report.to_dict(), indent=2))
    print(json.dumps({"random_seed0": random_report.to_dict()}, indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="combined-200-verbs-clipdistance-f1max")
    parser.add_argument("--target-clips", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-verbs", type=int, default=5)
    parser.add_argument("--min-pair-distance", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--allow-a101-only", action="store_true")
    return parser.parse_args(argv)


def _load_local_verb_clips() -> list[VerbClip]:
    clips: list[VerbClip] = []
    clips.extend(_load_assembly101_clips())
    clips.extend(_load_epic_p0113_clips())
    return clips


def _load_assembly101_clips() -> list[VerbClip]:
    root = REPO_ROOT / "data" / "raw" / "assembly101"
    actions_path = root / "annotations" / "coarse-annotations" / "actions.csv"
    label_dir = root / "annotations" / "coarse-annotations" / "coarse_labels"
    recordings_dir = root / "_hf" / "recordings"
    local_recordings = {path.name for path in recordings_dir.iterdir() if path.is_dir()}
    actions: dict[str, tuple[int, str, str]] = {}
    with actions_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            actions[row["action_cls"]] = (int(row["verb_id"]), row["verb_cls"], row["noun_cls"])

    out: list[VerbClip] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        recording_name = re.sub(r"^(assembly|disassembly)_", "", label_path.stem)
        if recording_name not in local_recordings:
            continue
        match = re.search(r"action_both_([^_]+)_", label_path.name)
        if match is None:
            continue
        video_id = match.group(1)
        media_path = root / f"{video_id}.mp4"
        if not media_path.is_file():
            continue
        for row_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            label = parts[2].strip()
            if label not in actions:
                continue
            try:
                start_sec = int(parts[0]) / 30.0
                end_sec = int(parts[1]) / 30.0
            except ValueError:
                continue
            if end_sec <= start_sec:
                continue
            _, verb, _ = actions[label]
            clip_id = f"assembly101:{label_path.stem}:{row_index}:{start_sec:.3f}:{end_sec:.3f}"
            out.append(
                VerbClip(
                    clip_id=clip_id,
                    dataset="assembly101",
                    video_id=video_id,
                    media_path=str(media_path),
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    gold_action=verb,
                    full_gold_action=label,
                    source_label_file=label_path.name,
                )
            )
    return out


def _load_epic_p0113_clips() -> list[VerbClip]:
    root = REPO_ROOT / "data" / "raw" / "epic_kitchens"
    media_path = root / "P01_13.MP4"
    if not media_path.is_file():
        return []
    intervals, _ = load_intervals(root / "intervals.csv", "P01_13")
    out: list[VerbClip] = []
    for row_index, interval in enumerate(intervals):
        verb = interval.gold_action.split()[0]
        out.append(
            VerbClip(
                clip_id=f"epic_kitchens:P01_13:{row_index}:{interval.start_sec:.3f}:{interval.end_sec:.3f}",
                dataset="epic_kitchens",
                video_id=interval.video_id,
                media_path=str(media_path),
                start_sec=float(interval.start_sec),
                end_sec=float(interval.end_sec),
                gold_action=verb,
                full_gold_action=interval.gold_action,
                source_label_file="intervals.csv",
            )
        )
    return out


def _score_clips(
    *,
    encoder: XClipEncoder,
    labels: list[str],
    text_embeddings: np.ndarray,
    clips: list[VerbClip],
    batch_size: int,
    num_frames: int,
) -> list[dict[str, Any]]:
    by_media: dict[str, list[VerbClip]] = defaultdict(list)
    for clip in clips:
        by_media[clip.media_path].append(clip)
    scored: list[dict[str, Any]] = []
    for media_path, media_clips in sorted(by_media.items()):
        frames_by_clip = _sample_video_once(Path(media_path), media_clips, num_frames=num_frames)
        ready = [(clip, frames_by_clip.get(clip.clip_id)) for clip in media_clips]
        ready = [(clip, frames) for clip, frames in ready if frames is not None]
        for offset in range(0, len(ready), batch_size):
            batch = ready[offset : offset + batch_size]
            frame_batch = np.stack([frames for _, frames in batch], axis=0)
            scores = encoder.score_clip_texts(frame_batch, text_embeddings)
            for (clip, _), row_scores in zip(batch, scores, strict=True):
                scored.append({"clip": clip, "scores": row_scores.astype(np.float32)})
    return scored


def _sample_video_once(path: Path, clips: list[VerbClip], *, num_frames: int) -> dict[str, np.ndarray]:
    import av

    targets: list[tuple[float, str, int]] = []
    for clip in clips:
        for frame_index, target_sec in enumerate(np.linspace(clip.start_sec, clip.end_sec, num_frames)):
            targets.append((float(target_sec), clip.clip_id, frame_index))
    targets.sort(key=lambda row: row[0])
    if not targets:
        return {}

    output = {
        clip.clip_id: np.zeros((num_frames, 3, 224, 224), dtype=np.float32)
        for clip in clips
    }
    filled = {clip.clip_id: np.zeros(num_frames, dtype=bool) for clip in clips}
    target_index = 0
    last_chw: np.ndarray | None = None

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            ts = float(frame.time) if frame.time is not None else float(frame.pts * stream.time_base)
            if target_index >= len(targets):
                break
            if ts < targets[target_index][0]:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            last_chw = _resize_rgb_to_chw(rgb)
            while target_index < len(targets) and ts >= targets[target_index][0]:
                _, clip_id, frame_index = targets[target_index]
                output[clip_id][frame_index] = last_chw
                filled[clip_id][frame_index] = True
                target_index += 1
    finally:
        container.close()

    if target_index < len(targets) and last_chw is not None:
        while target_index < len(targets):
            _, clip_id, frame_index = targets[target_index]
            output[clip_id][frame_index] = last_chw
            filled[clip_id][frame_index] = True
            target_index += 1
    return {clip_id: frames for clip_id, frames in output.items() if bool(filled[clip_id].all())}


def _resize_rgb_to_chw(rgb: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    ys = np.linspace(0, src_h - 1, height).astype(np.int32)
    xs = np.linspace(0, src_w - 1, width).astype(np.int32)
    resized = rgb[ys][:, xs]
    return np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0


def _select_best_subset(
    *,
    clips: list[VerbClip],
    scored_by_id: dict[str, dict[str, Any]],
    labels: list[str],
    label_to_index: dict[str, int],
    distance: np.ndarray,
    target_clips: int,
    min_verbs: int,
    min_pair_distance: float,
    require_epic: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    groups: dict[str, list[VerbClip]] = defaultdict(list)
    for clip in clips:
        groups[clip.gold_action].append(clip)
    for verb in groups:
        groups[verb].sort(key=lambda clip: (clip.dataset, clip.video_id, clip.source_label_file, clip.start_sec))

    candidates: list[dict[str, Any]] = []
    for size in range(max(2, min_verbs), len(labels) + 1):
        for subset in itertools.combinations(labels, size):
            if sum(len(groups[verb]) for verb in subset) < target_clips:
                continue
            if require_epic and not any(
                clip.dataset == "epic_kitchens" for verb in subset for clip in groups[verb]
            ):
                continue
            pairwise = _pairwise_distances(subset, label_to_index, distance)
            if pairwise and min(pairwise) < min_pair_distance:
                continue
            selected = _balanced_take(groups, subset, target_clips)
            if len(selected) != target_clips:
                continue
            metrics = _metrics_for_subset(selected, scored_by_id, subset, label_to_index)
            alloc = Counter(clip.gold_action for clip in selected)
            entropy = _entropy([alloc[verb] for verb in subset]) / math.log(len(subset))
            candidates.append(
                {
                    "labels": list(subset),
                    "clips": selected,
                    "allocation": dict(alloc),
                    "n_verbs": len(subset),
                    "min_pair_distance": float(min(pairwise)) if pairwise else 0.0,
                    "mean_pair_distance": float(np.mean(pairwise)) if pairwise else 0.0,
                    "balance_entropy": float(entropy),
                    **metrics,
                }
            )
    candidates.sort(
        key=lambda row: (
            row["macro_f1"],
            row["mean_pair_distance"],
            row["min_pair_distance"],
            row["balance_entropy"],
        ),
        reverse=True,
    )
    return (candidates[0] if candidates else None), candidates[:30]


def _pairwise_distances(
    subset: tuple[str, ...] | list[str],
    label_to_index: dict[str, int],
    distance: np.ndarray,
) -> list[float]:
    ids = [label_to_index[label] for label in subset]
    return [float(distance[left, right]) for index, left in enumerate(ids) for right in ids[index + 1 :]]


def _balanced_take(
    groups: dict[str, list[VerbClip]], subset: tuple[str, ...], target_clips: int) -> list[VerbClip]:
    caps = {verb: len(groups[verb]) for verb in subset}
    allocation = {verb: 0 for verb in subset}
    remaining = set(subset)
    left = target_clips
    while left > 0 and remaining:
        quota = math.ceil(left / len(remaining))
        changed = False
        for verb in sorted(list(remaining)):
            add = min(quota, caps[verb] - allocation[verb], left)
            if add > 0:
                allocation[verb] += add
                left -= add
                changed = True
            if allocation[verb] >= caps[verb]:
                remaining.remove(verb)
        if not changed:
            break
    if left != 0:
        return []
    selected: list[VerbClip] = []
    for verb in subset:
        selected.extend(_uniform_take(groups[verb], allocation[verb]))
    selected.sort(key=lambda clip: (clip.dataset, clip.video_id, clip.start_sec, clip.gold_action))
    return selected


def _uniform_take(items: list[VerbClip], n: int) -> list[VerbClip]:
    if n >= len(items):
        return list(items)
    indexes = np.linspace(0, len(items) - 1, n, dtype=np.int32)
    return [items[int(index)] for index in indexes]


def _metrics_for_subset(
    clips: list[VerbClip],
    scored_by_id: dict[str, dict[str, Any]],
    labels: tuple[str, ...] | list[str],
    label_to_index: dict[str, int],
) -> dict[str, float]:
    label_indexes = np.array([label_to_index[label] for label in labels], dtype=np.int32)
    golds: list[str] = []
    preds: list[str] = []
    ranks: list[int] = []
    for clip in clips:
        scores = scored_by_id[clip.clip_id]["scores"][label_indexes]
        order = np.argsort(-scores, kind="stable")
        ordered_labels = [labels[int(index)] for index in order]
        golds.append(clip.gold_action)
        preds.append(ordered_labels[0])
        ranks.append(ordered_labels.index(clip.gold_action) + 1)
    return {
        "top1": float(np.mean([rank <= 1 for rank in ranks])),
        "top3": float(np.mean([rank <= min(3, len(labels)) for rank in ranks])),
        "top5": float(np.mean([rank <= min(5, len(labels)) for rank in ranks])),
        "macro_f1": float(f1_score(golds, preds, average="macro", zero_division=0)),
    }


def _prediction_records_for_subset(
    *,
    clips: list[VerbClip],
    scored_by_id: dict[str, dict[str, Any]],
    labels: list[str],
    label_to_index: dict[str, int],
    dictionary_id: str,
    encoder_id: str,
    num_frames: int,
) -> list[PredictionRecord]:
    label_indexes = np.array([label_to_index[label] for label in labels], dtype=np.int32)
    records: list[PredictionRecord] = []
    for clip in clips:
        scores = scored_by_id[clip.clip_id]["scores"][label_indexes]
        ranking = ranking_from_similarities(scores, labels, clip_id=clip.clip_id)
        records.append(
            PredictionRecord(
                dataset=clip.dataset,
                video_id=clip.video_id,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                gold_action=clip.gold_action,
                pred_action=ranking.pred_action,
                topk_labels=ranking.labels[: min(5, len(labels))],
                topk_scores=ranking.cosine_similarity[: min(5, len(labels))],
                dictionary_id=dictionary_id,
                encoder_id=encoder_id,
                prompt_id=PROMPT_ID,
                gold_rank=gold_rank_1based(ranking, clip.gold_action, dictionary_id=dictionary_id),
                frame_count=num_frames,
            )
        )
    return records


def _random_baseline(
    *,
    clips: list[VerbClip],
    labels: list[str],
    dictionary_id: str,
    seed: int,
    n_trials: int,
) -> tuple[list[PredictionRecord], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[PredictionRecord] = []
    for clip in clips:
        scores = rng.random(len(labels), dtype=np.float32)
        ranking = ranking_from_similarities(scores, labels, clip_id=clip.clip_id)
        records.append(
            PredictionRecord(
                dataset=clip.dataset,
                video_id=clip.video_id,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                gold_action=clip.gold_action,
                pred_action=ranking.pred_action,
                topk_labels=ranking.labels[: min(5, len(labels))],
                topk_scores=ranking.cosine_similarity[: min(5, len(labels))],
                dictionary_id=dictionary_id,
                encoder_id=f"random-uniform-seed-{seed}",
                prompt_id="random_uniform",
                gold_rank=gold_rank_1based(ranking, clip.gold_action, dictionary_id=dictionary_id),
                frame_count=0,
            )
        )

    trial_f1 = []
    trial_top1 = []
    golds = [clip.gold_action for clip in clips]
    for _ in range(n_trials):
        pred_ids = rng.integers(0, len(labels), size=len(clips))
        preds = [labels[int(index)] for index in pred_ids]
        trial_f1.append(float(f1_score(golds, preds, average="macro", zero_division=0)))
        trial_top1.append(float(np.mean([gold == pred for gold, pred in zip(golds, preds, strict=True)])))
    return records, {
        "seed": seed,
        "n_trials": n_trials,
        "n_clips": len(clips),
        "n_labels": len(labels),
        "expected_top1": 1.0 / len(labels),
        "expected_top3": min(3, len(labels)) / len(labels),
        "expected_top5": min(5, len(labels)) / len(labels),
        "macro_f1_trials": _summary(np.asarray(trial_f1, dtype=np.float32)),
        "top1_trials": _summary(np.asarray(trial_top1, dtype=np.float32)),
    }


def _entropy(counts: list[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    probs = values / values.sum()
    return -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _write_artifacts(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    all_counts: Counter[str],
    candidate_counts: Counter[str],
    candidate_by_dataset: Counter[str],
    candidate_verbs: list[str],
    distance: np.ndarray,
    scored: list[dict[str, Any]],
    best: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    dictionary_id: str,
    selected_records: list[PredictionRecord],
    report: dict[str, Any],
    random_records: list[PredictionRecord],
    random_report: dict[str, Any],
    random_summary: dict[str, Any],
    elapsed_s: float,
    encoder_id: str,
) -> None:
    selected_clips = best["clips"]
    labels = best["labels"]
    (run_dir / "verb_bank.csv").write_text(
        "label\n" + "\n".join(labels) + "\n",
        encoding="utf-8",
    )
    clips_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_clips": len(selected_clips),
        "labels": labels,
        "selection": _jsonable_candidate(best, include_clips=False),
        "clips": [clip.to_json() for clip in selected_clips],
    }
    (run_dir / "clips_verb.json").write_text(json.dumps(clips_payload, indent=2), encoding="utf-8")
    (run_dir / f"metrics_{dictionary_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / f"metrics_{dictionary_id}_random_seed0.json").write_text(
        json.dumps(random_report, indent=2), encoding="utf-8"
    )
    (run_dir / "random_summary.json").write_text(json.dumps(random_summary, indent=2), encoding="utf-8")
    _write_jsonl(run_dir / "predictions.jsonl", [record.to_dict() for record in selected_records])
    _write_jsonl(run_dir / "predictions_random_seed0.jsonl", [record.to_dict() for record in random_records])
    _write_jsonl(
        run_dir / "all_candidate_scores.jsonl",
        [
            {
                **row["clip"].to_json(),
                "labels": candidate_verbs,
                "scores": [float(value) for value in row["scores"]],
            }
            for row in scored
        ],
    )
    distance_rows = [
        {
            "left": left,
            "right": right,
            "cosine_distance": float(distance[left_i, right_i]),
        }
        for left_i, left in enumerate(candidate_verbs)
        for right_i, right in enumerate(candidate_verbs)
        if left_i < right_i
    ]
    (run_dir / "verb_pairwise_distances.json").write_text(
        json.dumps(distance_rows, indent=2), encoding="utf-8"
    )
    (run_dir / "selection_candidates_top30.json").write_text(
        json.dumps([_jsonable_candidate(row, include_clips=False) for row in top_candidates], indent=2),
        encoding="utf-8",
    )
    meta = {
        "mode": "verb-selection-experiment",
        "encoder_id": encoder_id,
        "prompt_id": PROMPT_ID,
        "frame_count": args.num_frames,
        "weight_updates": False,
        "target_clips": args.target_clips,
        "selection_constraints": {
            "min_support": args.min_support,
            "min_verbs": args.min_verbs,
            "min_pair_distance": args.min_pair_distance,
            "require_epic": not args.allow_a101_only,
            "excluded_attempt_verbs": True,
        },
        "candidate_counts_all_verbs": dict(sorted(all_counts.items())),
        "candidate_counts_after_filter": dict(sorted(candidate_counts.items())),
        "candidate_clips_by_dataset": dict(sorted(candidate_by_dataset.items())),
        "dictionary_id": dictionary_id,
        "elapsed_s": elapsed_s,
        "selection_note": (
            "Subset is chosen by observed X-CLIP macro-F1 after filtering by CLIP text pairwise "
            "distance/support constraints. Treat as selection-tuned, not a held-out estimate."
        ),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(_summary_md(best, report, random_report, random_summary, meta), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _jsonable_candidate(row: dict[str, Any], *, include_clips: bool) -> dict[str, Any]:
    out = {key: value for key, value in row.items() if include_clips or key != "clips"}
    if include_clips and "clips" in out:
        out["clips"] = [clip.to_json() for clip in out["clips"]]
    return out


def _summary_md(best: dict[str, Any], report: dict[str, Any], random_report: dict[str, Any], random_summary: dict[str, Any], meta: dict[str, Any]) -> str:
    labels = ", ".join(best["labels"])
    allocation = ", ".join(f"{key}={value}" for key, value in sorted(best["allocation"].items()))
    datasets = Counter(clip.dataset for clip in best["clips"])
    dataset_line = ", ".join(f"{key}={value}" for key, value in sorted(datasets.items()))
    return f"""# Combined 200 Verb Selection Experiment

Selected verbs: `{labels}`

Dataset mix: {dataset_line}

Allocation: {allocation}

Selection constraints: min support {meta['selection_constraints']['min_support']}, min verbs {meta['selection_constraints']['min_verbs']}, min pairwise CLIP text cosine distance {meta['selection_constraints']['min_pair_distance']}, EPIC required {meta['selection_constraints']['require_epic']}.

This is a selection-tuned diagnostic: the subset is selected by observed X-CLIP macro-F1 after the CLIP text-distance/support filter, not by a held-out validation split.

| run | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| X-CLIP | {report['n_clips']} | {len(best['labels'])} | {report['action_top1']:.4f} | {report['action_top3']:.4f} | {report['action_top5']:.4f} | {report['action_macro_f1']:.4f} |
| random seed 0 | {random_report['n_clips']} | {len(best['labels'])} | {random_report['action_top1']:.4f} | {random_report['action_top3']:.4f} | {random_report['action_top5']:.4f} | {random_report['action_macro_f1']:.4f} |

Expected random Top-1/Top-3/Top-5: {random_summary['expected_top1']:.4f}/{random_summary['expected_top3']:.4f}/{random_summary['expected_top5']:.4f}.
Random macro-F1 Monte Carlo mean: {random_summary['macro_f1_trials']['mean']:.4f}.
"""


if __name__ == "__main__":
    raise SystemExit(main())
