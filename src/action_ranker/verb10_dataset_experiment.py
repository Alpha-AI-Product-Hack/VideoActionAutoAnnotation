from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from action_ranker.encode_actions import encode_action_batch
from action_ranker.metrics import compute_action_metrics
from action_ranker.prompts import PROMPT_ID
from action_ranker.rank import l2_normalize
from action_ranker.slice import load_intervals
from action_ranker.taxonomies import REPO_ROOT
from action_ranker.verb_selection_experiment import (
    VerbClip,
    _load_assembly101_clips,
    _prediction_records_for_subset,
    _random_baseline,
    _score_clips,
)
from action_ranker.xclip_encoder import XClipEncoder


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    clips = _load_dataset_clips(args.dataset)
    counts = Counter(clip.gold_action for clip in clips)
    support = args.min_support if args.allow_imbalanced else args.target_clips // args.n_verbs
    candidate_verbs = sorted(
        verb
        for verb, count in counts.items()
        if count >= support and _allowed_verb(verb, args.dataset)
    )

    availability = {
        "dataset": args.dataset,
        "n_raw_local_clips": len(clips),
        "n_verbs_raw": len(counts),
        "target_clips": args.target_clips,
        "n_verbs_requested": args.n_verbs,
        "clips_per_verb": support,
        "raw_counts": dict(sorted(counts.items())),
        "candidate_verbs": candidate_verbs,
        "candidate_counts": {verb: counts[verb] for verb in candidate_verbs},
    }
    if not args.allow_imbalanced and args.target_clips % args.n_verbs != 0:
        availability["status"] = "unavailable"
        availability["reason"] = "target_clips must divide evenly by n_verbs for balanced sampling"
        _write_unavailable(run_dir, availability)
        return 0
    if len(candidate_verbs) < args.n_verbs:
        availability["status"] = "unavailable"
        availability["reason"] = (
            f"Need {args.n_verbs} verbs with at least {support} local clips each; "
            f"found {len(candidate_verbs)}."
        )
        if args.dataset == "epic_kitchens":
            availability["epic_note"] = (
                "Only local EPIC media are used. In this workspace P01_12.MP4 is incomplete/unreadable "
                "and P01_13.MP4 has 30 decodable intervals, so 200 local EPIC clips are not available."
            )
        _write_unavailable(run_dir, availability)
        return 0

    started = time.perf_counter()
    encoder = XClipEncoder(num_frames=args.num_frames)
    gpu = _gpu_info(encoder)
    text_embeddings = encode_action_batch(candidate_verbs, encoder, PROMPT_ID)
    text_norm = l2_normalize(text_embeddings)
    distance = 1.0 - (text_norm @ text_norm.T)
    selected_verbs, diversity = _select_diverse_verbs(candidate_verbs, distance, args.n_verbs)
    if args.allow_imbalanced:
        selected_clips = _waterfill_sample(clips, selected_verbs, target_clips=args.target_clips)
    else:
        selected_clips = _balanced_sample(clips, selected_verbs, per_verb=support)
    if len(selected_clips) != args.target_clips:
        availability["status"] = "unavailable"
        availability["reason"] = "Balanced sampling unexpectedly produced the wrong number of clips"
        _write_unavailable(run_dir, availability)
        return 0

    selected_text = encode_action_batch(selected_verbs, encoder, PROMPT_ID)
    scored = _score_clips(
        encoder=encoder,
        labels=selected_verbs,
        text_embeddings=selected_text,
        clips=selected_clips,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
    )
    scored_by_id = {row["clip"].clip_id: row for row in scored}
    if len(scored_by_id) != len(selected_clips):
        missing = len(selected_clips) - len(scored_by_id)
        availability["status"] = "unavailable"
        availability["reason"] = f"Only {len(scored_by_id)} of {len(selected_clips)} selected clips decoded/scored; missing {missing}."
        _write_unavailable(run_dir, availability)
        return 0

    dictionary_id = f"{args.dataset}_verb{len(selected_verbs)}_diverse{args.target_clips}"
    label_to_index = {label: index for index, label in enumerate(selected_verbs)}
    records = _prediction_records_for_subset(
        clips=selected_clips,
        scored_by_id=scored_by_id,
        labels=selected_verbs,
        label_to_index=label_to_index,
        dictionary_id=dictionary_id,
        encoder_id=encoder.encoder_id,
        num_frames=args.num_frames,
    )
    report = compute_action_metrics(records, dictionary_id)
    random_records, random_summary = _random_baseline(
        clips=selected_clips,
        labels=selected_verbs,
        dictionary_id=dictionary_id,
        seed=args.seed,
        n_trials=args.random_trials,
    )
    random_report = compute_action_metrics(random_records, dictionary_id)

    if gpu["cuda_available"] and encoder.device.startswith("cuda"):
        encoder.torch.cuda.synchronize()
        gpu["cuda_memory_allocated_after_forward_bytes"] = int(encoder.torch.cuda.memory_allocated())
        gpu["gpu_inference_verified"] = True
    else:
        gpu["gpu_inference_verified"] = False

    _write_success(
        run_dir=run_dir,
        args=args,
        availability=availability,
        selected_verbs=selected_verbs,
        diversity=diversity,
        selected_clips=selected_clips,
        distance=distance,
        candidate_verbs=candidate_verbs,
        records=records,
        report=report.to_dict(),
        random_records=random_records,
        random_report=random_report.to_dict(),
        random_summary=random_summary,
        gpu=gpu,
        elapsed_s=time.perf_counter() - started,
        dictionary_id=dictionary_id,
        encoder_id=encoder.encoder_id,
    )
    print(run_dir)
    print(json.dumps({"gpu": gpu, "metrics": report.to_dict(), "random": random_report.to_dict()}, indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["assembly101_fine", "assembly101_coarse", "epic_kitchens"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-clips", type=int, default=200)
    parser.add_argument("--n-verbs", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--allow-imbalanced", action="store_true")
    return parser.parse_args(argv)


def _load_dataset_clips(dataset: str) -> list[VerbClip]:
    if dataset == "assembly101_fine":
        return _load_assembly101_fine_hmc_clips()
    if dataset == "assembly101_coarse":
        return _load_assembly101_clips()
    if dataset == "epic_kitchens":
        return _load_epic_local_clips()
    raise ValueError(dataset)


def _load_assembly101_fine_hmc_clips() -> list[VerbClip]:
    root = REPO_ROOT / "data" / "raw" / "assembly101"
    validation = root / "annotations" / "fine-grained-annotations" / "validation.csv"
    media_lookup: dict[tuple[str, str], Path] = {}
    for recording_dir in sorted((root / "_hf" / "recordings").glob("*")):
        if not recording_dir.is_dir():
            continue
        for media in sorted(recording_dir.glob("HMC_*.mp4")):
            media_lookup[(recording_dir.name, media.name)] = media
    clips: list[VerbClip] = []
    with validation.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video = (row.get("video") or "").strip()
            if "/" not in video:
                continue
            recording, view_name = video.split("/", 1)
            media_path = media_lookup.get((recording, view_name))
            if media_path is None:
                continue
            try:
                start_sec = int(row["start_frame"]) / 30.0
                end_sec = int(row["end_frame"]) / 30.0
            except (KeyError, ValueError):
                continue
            if end_sec <= start_sec:
                continue
            verb = (row.get("verb_cls") or "").strip()
            action = (row.get("action_cls") or "").strip()
            if not verb:
                continue
            clip_id = f"assembly101_fine:{row.get('id', len(clips))}:{recording}:{view_name}:{start_sec:.3f}:{end_sec:.3f}"
            clips.append(
                VerbClip(
                    clip_id=clip_id,
                    dataset="assembly101",
                    video_id=f"{recording}/{view_name}",
                    media_path=str(media_path),
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    gold_action=verb,
                    full_gold_action=action,
                    source_label_file="fine-grained-annotations/validation.csv",
                )
            )
    return clips


def _load_epic_local_clips() -> list[VerbClip]:
    root = REPO_ROOT / "data" / "raw" / "epic_kitchens"
    clips: list[VerbClip] = []
    media_by_id = {
        path.stem: path
        for path in sorted(root.glob("*.MP4")) + sorted(root.glob("*.mp4"))
        if path.is_file()
    }
    for video_id, media_path in sorted(media_by_id.items()):
        intervals, _ = load_intervals(root / "intervals.csv", video_id)
        for index, interval in enumerate(intervals):
            verb = interval.gold_action.split()[0]
            clips.append(
                VerbClip(
                    clip_id=f"epic_kitchens:{video_id}:{index}:{interval.start_sec:.3f}:{interval.end_sec:.3f}",
                    dataset="epic_kitchens",
                    video_id=video_id,
                    media_path=str(media_path),
                    start_sec=float(interval.start_sec),
                    end_sec=float(interval.end_sec),
                    gold_action=verb,
                    full_gold_action=interval.gold_action,
                    source_label_file="intervals.csv",
                )
            )
    return clips


def _allowed_verb(verb: str, dataset: str) -> bool:
    if dataset.startswith("assembly101") and verb.startswith("attempt to "):
        return False
    return True


def _select_diverse_verbs(labels: list[str], distance: np.ndarray, n_verbs: int) -> tuple[list[str], dict[str, float]]:
    best_labels: tuple[str, ...] | None = None
    best_values: list[float] = []
    best_key: tuple[float, float, float] | None = None
    index = {label: i for i, label in enumerate(labels)}
    for subset in itertools.combinations(labels, n_verbs):
        values = [
            float(distance[index[left], index[right]])
            for offset, left in enumerate(subset)
            for right in subset[offset + 1 :]
        ]
        key = (min(values), float(np.mean(values)), float(np.median(values)))
        if best_key is None or key > best_key:
            best_key = key
            best_labels = subset
            best_values = values
    if best_labels is None:
        raise ValueError("No diverse verb subset found")
    return list(best_labels), {
        "min_pair_distance": float(min(best_values)),
        "mean_pair_distance": float(np.mean(best_values)),
        "median_pair_distance": float(np.median(best_values)),
    }


def _balanced_sample(clips: list[VerbClip], labels: list[str], *, per_verb: int) -> list[VerbClip]:
    groups: dict[str, list[VerbClip]] = defaultdict(list)
    for clip in clips:
        if clip.gold_action in labels:
            groups[clip.gold_action].append(clip)
    selected: list[VerbClip] = []
    for label in labels:
        items = sorted(groups[label], key=lambda clip: (clip.media_path, clip.start_sec, clip.end_sec, clip.clip_id))
        if len(items) < per_verb:
            raise ValueError(f"Need {per_verb} clips for {label}, found {len(items)}")
        indexes = np.linspace(0, len(items) - 1, per_verb, dtype=np.int32)
        selected.extend(items[int(index)] for index in indexes)
    selected.sort(key=lambda clip: (clip.dataset, clip.media_path, clip.start_sec, clip.gold_action))
    return selected


def _waterfill_sample(clips: list[VerbClip], labels: list[str], *, target_clips: int) -> list[VerbClip]:
    groups: dict[str, list[VerbClip]] = defaultdict(list)
    for clip in clips:
        if clip.gold_action in labels:
            groups[clip.gold_action].append(clip)
    for label in labels:
        groups[label].sort(key=lambda clip: (clip.media_path, clip.start_sec, clip.end_sec, clip.clip_id))
    caps = {label: len(groups[label]) for label in labels}
    if sum(caps.values()) < target_clips:
        return []
    allocation = {label: 0 for label in labels}
    remaining = set(labels)
    left = target_clips
    while left > 0 and remaining:
        quota = math.ceil(left / len(remaining))
        changed = False
        for label in sorted(list(remaining)):
            add = min(quota, caps[label] - allocation[label], left)
            if add > 0:
                allocation[label] += add
                left -= add
                changed = True
            if allocation[label] >= caps[label]:
                remaining.remove(label)
        if not changed:
            break
    if left != 0:
        return []
    selected: list[VerbClip] = []
    for label in labels:
        items = groups[label]
        n = allocation[label]
        indexes = np.linspace(0, len(items) - 1, n, dtype=np.int32)
        selected.extend(items[int(index)] for index in indexes)
    selected.sort(key=lambda clip: (clip.dataset, clip.media_path, clip.start_sec, clip.gold_action))
    return selected


def _gpu_info(encoder: XClipEncoder) -> dict[str, Any]:
    torch = encoder.torch
    info: dict[str, Any] = {
        "encoder_device": encoder.device,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
        info["model_parameter_device"] = str(next(encoder.model.parameters()).device)
    return info


def _write_unavailable(run_dir: Path, availability: dict[str, Any]) -> None:
    availability["created_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "availability.json").write_text(json.dumps(availability, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(_unavailable_markdown(availability), encoding="utf-8")
    print(run_dir)
    print(json.dumps(availability, indent=2))


def _write_success(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    availability: dict[str, Any],
    selected_verbs: list[str],
    diversity: dict[str, float],
    selected_clips: list[VerbClip],
    distance: np.ndarray,
    candidate_verbs: list[str],
    records: list[Any],
    report: dict[str, Any],
    random_records: list[Any],
    random_report: dict[str, Any],
    random_summary: dict[str, Any],
    gpu: dict[str, Any],
    elapsed_s: float,
    dictionary_id: str,
    encoder_id: str,
) -> None:
    allocation = Counter(clip.gold_action for clip in selected_clips)
    dataset_counts = Counter(clip.dataset for clip in selected_clips)
    (run_dir / "verb_bank.csv").write_text("label\n" + "\n".join(selected_verbs) + "\n", encoding="utf-8")
    (run_dir / "clips_verb.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset,
                "n_clips": len(selected_clips),
                "verbs": selected_verbs,
                "allocation": dict(sorted(allocation.items())),
                "clips": [clip.to_json() for clip in selected_clips],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / f"metrics_{dictionary_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / f"metrics_{dictionary_id}_random_seed0.json").write_text(json.dumps(random_report, indent=2), encoding="utf-8")
    (run_dir / "random_summary.json").write_text(json.dumps(random_summary, indent=2), encoding="utf-8")
    _write_jsonl(run_dir / "predictions.jsonl", [record.to_dict() for record in records])
    _write_jsonl(run_dir / "predictions_random_seed0.jsonl", [record.to_dict() for record in random_records])
    pairwise = _pairwise_rows(selected_verbs, candidate_verbs, distance)
    (run_dir / "selected_pairwise_distances.json").write_text(json.dumps(pairwise, indent=2), encoding="utf-8")
    meta = {
        "mode": "verb10-dataset-experiment",
        "dataset": args.dataset,
        "dictionary_id": dictionary_id,
        "encoder_id": encoder_id,
        "prompt_id": PROMPT_ID,
        "frame_count": args.num_frames,
        "weight_updates": False,
        "target_clips": args.target_clips,
        "n_verbs": args.n_verbs,
        "clips_per_verb": None if args.allow_imbalanced else args.target_clips // args.n_verbs,
        "allow_imbalanced": bool(args.allow_imbalanced),
        "selection_rule": "maximize minimum pairwise X-CLIP text cosine distance, then mean distance, among verbs with enough local clips",
        "selected_verbs": selected_verbs,
        "allocation": dict(sorted(allocation.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "diversity": diversity,
        "availability": availability,
        "gpu": gpu,
        "elapsed_s": elapsed_s,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(
        _success_markdown(selected_verbs, allocation, dataset_counts, diversity, report, random_report, random_summary, gpu),
        encoding="utf-8",
    )


def _pairwise_rows(labels: list[str], candidate_verbs: list[str], distance: np.ndarray) -> list[dict[str, Any]]:
    index = {label: i for i, label in enumerate(candidate_verbs)}
    return [
        {
            "left": left,
            "right": right,
            "cosine_distance": float(distance[index[left], index[right]]),
        }
        for left_i, left in enumerate(labels)
        for right in labels[left_i + 1 :]
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _success_markdown(
    labels: list[str],
    allocation: Counter[str],
    dataset_counts: Counter[str],
    diversity: dict[str, float],
    report: dict[str, Any],
    random_report: dict[str, Any],
    random_summary: dict[str, Any],
    gpu: dict[str, Any],
) -> str:
    label_text = ", ".join(f"`{label}`" for label in labels)
    allocation_text = ", ".join(f"{key}={value}" for key, value in sorted(allocation.items()))
    dataset_text = ", ".join(f"{key}={value}" for key, value in sorted(dataset_counts.items()))
    return f"""# Verb10 Diverse 200 Experiment

Selected verbs: {label_text}

Dataset mix: {dataset_text}

Allocation: {allocation_text}

Diversity: min pairwise CLIP text cosine distance {diversity['min_pair_distance']:.4f}, mean {diversity['mean_pair_distance']:.4f}, median {diversity['median_pair_distance']:.4f}.

GPU: cuda_available={gpu['cuda_available']}, encoder_device=`{gpu['encoder_device']}`, model_parameter_device=`{gpu.get('model_parameter_device', 'n/a')}`, gpu_inference_verified={gpu['gpu_inference_verified']}.

| run | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| X-CLIP | {report['n_clips']} | {len(labels)} | {report['action_top1']:.4f} | {report['action_top3']:.4f} | {report['action_top5']:.4f} | {report['action_macro_f1']:.4f} |
| random seed 0 | {random_report['n_clips']} | {len(labels)} | {random_report['action_top1']:.4f} | {random_report['action_top3']:.4f} | {random_report['action_top5']:.4f} | {random_report['action_macro_f1']:.4f} |

Expected random Top-1/Top-3/Top-5: {random_summary['expected_top1']:.4f}/{random_summary['expected_top3']:.4f}/{random_summary['expected_top5']:.4f}.
Random macro-F1 Monte Carlo mean: {random_summary['macro_f1_trials']['mean']:.4f}.
"""


def _unavailable_markdown(availability: dict[str, Any]) -> str:
    return f"""# Verb10 Diverse 200 Experiment Unavailable

Dataset: `{availability['dataset']}`

Status: unavailable

Reason: {availability['reason']}

Local clips seen: {availability['n_raw_local_clips']}

Raw verbs seen: {availability['n_verbs_raw']}

Candidate verbs with enough local clips: {len(availability['candidate_verbs'])}

{availability.get('epic_note', '')}
"""


if __name__ == "__main__":
    raise SystemExit(main())
