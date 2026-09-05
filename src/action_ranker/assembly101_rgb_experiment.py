from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from action_ranker.encode_actions import encode_action_batch
from action_ranker.metrics import compute_action_metrics
from action_ranker.prompts import PROMPT_ID
from action_ranker.taxonomies import REPO_ROOT
from action_ranker.verb_selection_experiment import (
    VerbClip,
    _prediction_records_for_subset,
    _random_baseline,
    _score_clips,
)
from action_ranker.verb10_dataset_experiment import _gpu_info
from action_ranker.xclip_encoder import XClipEncoder

A101_REPO = "cvml-nus/assembly101"
RAW_A101 = REPO_ROOT / "data" / "raw" / "assembly101"
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"
DEFAULT_RECORDINGS = [
    "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724",
    "nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253",
]
FINE_SPLITS = ["train.csv", "validation.csv", "test.csv"]
FPS = 30.0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = RUNS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    recordings = args.recording or DEFAULT_RECORDINGS
    views_by_recording = _rgb_views_by_recording(recordings)
    missing_recordings = [rec for rec in recordings if not views_by_recording.get(rec)]
    if missing_recordings:
        raise SystemExit(f"No RGB views found in annotations for: {missing_recordings}")

    download_rows = _ensure_rgb_media(
        recordings=recordings,
        views_by_recording=views_by_recording,
        download=not args.no_download,
    )
    media_by_recording = _media_by_recording(download_rows)
    fine_clips = _load_fine_clips(recordings, media_by_recording)
    coarse_clips = _load_coarse_clips(recordings, media_by_recording)
    if args.max_fine_clips is not None:
        fine_clips = _uniform_subset(fine_clips, args.max_fine_clips)
    if args.max_coarse_clips is not None:
        coarse_clips = _uniform_subset(coarse_clips, args.max_coarse_clips)

    fine_labels = _load_bank_labels(RAW_A101 / "annotations" / "fine-grained-annotations" / "actions.csv")
    coarse_labels = _load_bank_labels(RAW_A101 / "annotations" / "coarse-annotations" / "actions.csv")
    _write_input_artifacts(
        run_dir=run_dir,
        recordings=recordings,
        views_by_recording=views_by_recording,
        download_rows=download_rows,
        fine_clips=fine_clips,
        coarse_clips=coarse_clips,
        fine_labels=fine_labels,
        coarse_labels=coarse_labels,
    )
    if args.download_only:
        print(run_dir)
        print(json.dumps({"downloaded_or_present": len(download_rows), "fine_clips": len(fine_clips), "coarse_clips": len(coarse_clips)}, indent=2))
        return 0

    started = time.perf_counter()
    encoder = XClipEncoder(num_frames=args.num_frames)
    gpu = _gpu_info(encoder)
    reports: dict[str, dict[str, Any]] = {}
    random_reports: dict[str, dict[str, Any]] = {}
    random_summaries: dict[str, dict[str, Any]] = {}
    evaluated: dict[str, int] = {}

    if fine_clips:
        report, random_report, random_summary, n_scored = _evaluate_split(
            run_dir=run_dir,
            split_name="fine",
            dictionary_id="assembly101_fine_rgb_two_recordings",
            clips=fine_clips,
            labels=fine_labels,
            encoder=encoder,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            seed=args.seed,
            random_trials=args.random_trials,
        )
        reports["fine"] = report
        random_reports["fine"] = random_report
        random_summaries["fine"] = random_summary
        evaluated["fine"] = n_scored

    if coarse_clips:
        report, random_report, random_summary, n_scored = _evaluate_split(
            run_dir=run_dir,
            split_name="coarse",
            dictionary_id="assembly101_coarse_rgb_two_recordings",
            clips=coarse_clips,
            labels=coarse_labels,
            encoder=encoder,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            seed=args.seed,
            random_trials=args.random_trials,
        )
        reports["coarse"] = report
        random_reports["coarse"] = random_report
        random_summaries["coarse"] = random_summary
        evaluated["coarse"] = n_scored

    if gpu["cuda_available"] and encoder.device.startswith("cuda"):
        encoder.torch.cuda.synchronize()
        gpu["cuda_memory_allocated_after_forward_bytes"] = int(encoder.torch.cuda.memory_allocated())
        gpu["gpu_inference_verified"] = True
    else:
        gpu["gpu_inference_verified"] = False
    elapsed_s = time.perf_counter() - started
    _write_run_meta(
        run_dir=run_dir,
        args=args,
        recordings=recordings,
        views_by_recording=views_by_recording,
        download_rows=download_rows,
        reports=reports,
        random_reports=random_reports,
        random_summaries=random_summaries,
        gpu=gpu,
        elapsed_s=elapsed_s,
        evaluated=evaluated,
    )
    (run_dir / "summary.md").write_text(
        _summary_markdown(
            recordings=recordings,
            views_by_recording=views_by_recording,
            fine_clips=fine_clips,
            coarse_clips=coarse_clips,
            fine_labels=fine_labels,
            coarse_labels=coarse_labels,
            reports=reports,
            random_reports=random_reports,
            random_summaries=random_summaries,
            gpu=gpu,
            elapsed_s=elapsed_s,
            evaluated=evaluated,
        ),
        encoding="utf-8",
    )
    print(run_dir)
    print(json.dumps({"metrics": reports, "random": random_reports, "gpu": gpu}, indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/evaluate Assembly101 RGB views for two recordings")
    parser.add_argument("--run-id", default="assembly101-rgb-two-recordings-xclip")
    parser.add_argument("--recording", action="append", help="Recording id; repeat to override defaults")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--max-fine-clips", type=int)
    parser.add_argument("--max-coarse-clips", type=int)
    return parser.parse_args(argv)


def _rgb_views_by_recording(recordings: list[str]) -> dict[str, list[str]]:
    wanted = set(recordings)
    out: dict[str, set[str]] = {rec: set() for rec in recordings}
    fine_dir = RAW_A101 / "annotations" / "fine-grained-annotations"
    for split in FINE_SPLITS:
        path = fine_dir / split
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                video = (row.get("video") or "").strip()
                if "/" not in video:
                    continue
                recording, view = video.split("/", 1)
                if recording in wanted and view.endswith("_rgb.mp4"):
                    out[recording].add(view)

    seq_views = RAW_A101 / "annotations" / "coarse-annotations" / "coarse_seq_views.txt"
    if seq_views.is_file():
        for line in seq_views.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "/" not in line or not line.endswith("_rgb.mp4"):
                continue
            prefixed_recording, view = line.split("/", 1)
            for phase in ("assembly_", "disassembly_"):
                if prefixed_recording.startswith(phase):
                    recording = prefixed_recording[len(phase) :]
                    break
            else:
                recording = prefixed_recording
            if recording in wanted:
                out[recording].add(view)
    return {recording: sorted(views) for recording, views in out.items()}


def _ensure_rgb_media(
    *,
    recordings: list[str],
    views_by_recording: dict[str, list[str]],
    download: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    if download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub to download Assembly101 RGB media") from exc
    else:
        hf_hub_download = None

    for recording in recordings:
        for view in views_by_recording[recording]:
            rel = f"recordings/{recording}/{view}"
            local = RAW_A101 / "_hf" / rel
            status = "present" if local.is_file() else "missing"
            if status == "missing" and download:
                downloaded = hf_hub_download(
                    A101_REPO,
                    rel,
                    repo_type="dataset",
                    local_dir=str(RAW_A101 / "_hf"),
                )
                local = Path(downloaded)
                status = "downloaded"
            if not local.is_file():
                missing.append(rel)
                continue
            rows.append(
                {
                    "recording": recording,
                    "view": view,
                    "repo_path": rel,
                    "media_path": str(local),
                    "size_bytes": int(local.stat().st_size),
                    "status": status,
                }
            )
    if missing:
        raise SystemExit("Missing RGB media: " + ", ".join(missing[:20]))
    return rows


def _media_by_recording(download_rows: list[dict[str, Any]]) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for row in download_rows:
        out.setdefault(str(row["recording"]), {})[str(row["view"])] = Path(str(row["media_path"]))
    return out


def _load_fine_clips(recordings: list[str], media_by_recording: dict[str, dict[str, Path]]) -> list[VerbClip]:
    wanted = set(recordings)
    clips: list[VerbClip] = []
    fine_dir = RAW_A101 / "annotations" / "fine-grained-annotations"
    for split in FINE_SPLITS:
        path = fine_dir / split
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                video = (row.get("video") or "").strip()
                if "/" not in video:
                    continue
                recording, view = video.split("/", 1)
                if recording not in wanted or view not in media_by_recording.get(recording, {}):
                    continue
                try:
                    start_sec = int(row["start_frame"]) / FPS
                    end_sec = int(row["end_frame"]) / FPS
                except (KeyError, ValueError):
                    continue
                if end_sec <= start_sec:
                    continue
                action = (row.get("action_cls") or "").strip()
                if not action:
                    continue
                clips.append(
                    VerbClip(
                        clip_id=(
                            f"assembly101_fine_rgb:{split}:{row.get('id', len(clips))}:"
                            f"{recording}:{view}:{start_sec:.3f}:{end_sec:.3f}"
                        ),
                        dataset="assembly101",
                        video_id=f"{recording}/{view}",
                        media_path=str(media_by_recording[recording][view]),
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                        gold_action=action,
                        full_gold_action=action,
                        source_label_file=f"fine-grained-annotations/{split}",
                    )
                )
    clips.sort(key=lambda clip: (clip.video_id, clip.start_sec, clip.end_sec, clip.gold_action))
    return clips


def _load_coarse_clips(recordings: list[str], media_by_recording: dict[str, dict[str, Path]]) -> list[VerbClip]:
    clips: list[VerbClip] = []
    label_dir = RAW_A101 / "annotations" / "coarse-annotations" / "coarse_labels"
    for recording in recordings:
        views = media_by_recording.get(recording, {})
        for phase in ("disassembly", "assembly"):
            label_path = label_dir / f"{phase}_{recording}.txt"
            if not label_path.is_file():
                continue
            for row_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.split("\t")
                if len(parts) < 3:
                    parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                try:
                    start_sec = int(parts[0]) / FPS
                    end_sec = int(parts[1]) / FPS
                except ValueError:
                    continue
                action = parts[2].strip()
                if end_sec <= start_sec or not action:
                    continue
                for view, media_path in sorted(views.items()):
                    clips.append(
                        VerbClip(
                            clip_id=(
                                f"assembly101_coarse_rgb:{phase}:{recording}:{view}:"
                                f"{row_index}:{start_sec:.3f}:{end_sec:.3f}"
                            ),
                            dataset="assembly101",
                            video_id=f"{recording}/{view}",
                            media_path=str(media_path),
                            start_sec=float(start_sec),
                            end_sec=float(end_sec),
                            gold_action=action,
                            full_gold_action=action,
                            source_label_file=f"coarse-annotations/coarse_labels/{phase}_{recording}.txt",
                        )
                    )
    clips.sort(key=lambda clip: (clip.video_id, clip.start_sec, clip.end_sec, clip.gold_action))
    return clips


def _load_bank_labels(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        labels = [(row.get("action_cls") or "").strip() for row in reader]
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    if not out:
        raise ValueError(f"No action_cls labels in {path}")
    return out


def _uniform_subset(clips: list[VerbClip], n: int) -> list[VerbClip]:
    if n < 1:
        return []
    if n >= len(clips):
        return list(clips)
    indexes = np.linspace(0, len(clips) - 1, n, dtype=np.int32)
    return [clips[int(index)] for index in indexes]


def _evaluate_split(
    *,
    run_dir: Path,
    split_name: str,
    dictionary_id: str,
    clips: list[VerbClip],
    labels: list[str],
    encoder: XClipEncoder,
    batch_size: int,
    num_frames: int,
    seed: int,
    random_trials: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    text = encode_action_batch(labels, encoder, PROMPT_ID)
    scored = _score_clips(
        encoder=encoder,
        labels=labels,
        text_embeddings=text,
        clips=clips,
        batch_size=batch_size,
        num_frames=num_frames,
    )
    scored_by_id = {row["clip"].clip_id: row for row in scored}
    scored_clips = [clip for clip in clips if clip.clip_id in scored_by_id]
    label_to_index = {label: index for index, label in enumerate(labels)}
    records = _prediction_records_for_subset(
        clips=scored_clips,
        scored_by_id=scored_by_id,
        labels=labels,
        label_to_index=label_to_index,
        dictionary_id=dictionary_id,
        encoder_id=encoder.encoder_id,
        num_frames=num_frames,
    )
    report = compute_action_metrics(
        records,
        dictionary_id,
        n_skipped_intervals=len(clips) - len(records),
    )
    random_records, random_summary = _random_baseline(
        clips=scored_clips,
        labels=labels,
        dictionary_id=dictionary_id,
        seed=seed,
        n_trials=random_trials,
    )
    random_report = compute_action_metrics(random_records, dictionary_id)

    _write_json(run_dir / f"metrics_{split_name}_{dictionary_id}.json", report.to_dict())
    _write_json(run_dir / f"metrics_{split_name}_{dictionary_id}_random_seed{seed}.json", random_report.to_dict())
    _write_json(run_dir / f"random_summary_{split_name}.json", random_summary)
    _write_jsonl(run_dir / f"predictions_{split_name}.jsonl", [record.to_dict() for record in records])
    _write_jsonl(run_dir / f"predictions_{split_name}_random_seed{seed}.jsonl", [record.to_dict() for record in random_records])
    return report.to_dict(), random_report.to_dict(), random_summary, len(records)


def _write_input_artifacts(
    *,
    run_dir: Path,
    recordings: list[str],
    views_by_recording: dict[str, list[str]],
    download_rows: list[dict[str, Any]],
    fine_clips: list[VerbClip],
    coarse_clips: list[VerbClip],
    fine_labels: list[str],
    coarse_labels: list[str],
) -> None:
    _write_json(run_dir / "downloaded_rgb_files.json", {"files": download_rows})
    _write_json(run_dir / "clips_fine.json", _clips_payload("assembly101_fine_rgb", recordings, views_by_recording, fine_clips))
    _write_json(run_dir / "clips_coarse.json", _clips_payload("assembly101_coarse_rgb", recordings, views_by_recording, coarse_clips))
    (run_dir / "fine_bank.csv").write_text("label\n" + "\n".join(fine_labels) + "\n", encoding="utf-8")
    (run_dir / "coarse_bank.csv").write_text("label\n" + "\n".join(coarse_labels) + "\n", encoding="utf-8")


def _clips_payload(
    dataset: str,
    recordings: list[str],
    views_by_recording: dict[str, list[str]],
    clips: list[VerbClip],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "recordings": recordings,
        "views_by_recording": views_by_recording,
        "n_clips": len(clips),
        "gold_counts": dict(sorted(Counter(clip.gold_action for clip in clips).items())),
        "clips": [clip.to_json() for clip in clips],
    }


def _write_run_meta(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    recordings: list[str],
    views_by_recording: dict[str, list[str]],
    download_rows: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    random_reports: dict[str, dict[str, Any]],
    random_summaries: dict[str, dict[str, Any]],
    gpu: dict[str, Any],
    elapsed_s: float,
    evaluated: dict[str, int],
) -> None:
    _write_json(
        run_dir / "run_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "assembly101-rgb-two-recordings-xclip",
            "args": vars(args),
            "recordings": recordings,
            "views_by_recording": views_by_recording,
            "n_rgb_files": len(download_rows),
            "downloaded_or_present_bytes": int(sum(row["size_bytes"] for row in download_rows)),
            "encoder_id": "microsoft/xclip-base-patch32",
            "prompt_id": PROMPT_ID,
            "frame_count": args.num_frames,
            "batch_size": args.batch_size,
            "headline_metrics": ["action_top1", "action_top3", "action_top5", "action_macro_f1"],
            "reports": reports,
            "random_reports": random_reports,
            "random_summaries": random_summaries,
            "evaluated": evaluated,
            "gpu": gpu,
            "elapsed_s": elapsed_s,
        },
    )


def _summary_markdown(
    *,
    recordings: list[str],
    views_by_recording: dict[str, list[str]],
    fine_clips: list[VerbClip],
    coarse_clips: list[VerbClip],
    fine_labels: list[str],
    coarse_labels: list[str],
    reports: dict[str, dict[str, Any]],
    random_reports: dict[str, dict[str, Any]],
    random_summaries: dict[str, dict[str, Any]],
    gpu: dict[str, Any],
    elapsed_s: float,
    evaluated: dict[str, int],
) -> str:
    rec_text = "\n".join(f"- `{recording}`: {len(views_by_recording[recording])} RGB views" for recording in recordings)
    rows = []
    for split_name, labels in (("fine", fine_labels), ("coarse", coarse_labels)):
        if split_name not in reports:
            continue
        report = reports[split_name]
        random_report = random_reports[split_name]
        random_summary = random_summaries[split_name]
        rows.append(
            f"| {split_name} X-CLIP | {report['n_clips']} | {len(labels)} | "
            f"{report['action_top1']:.4f} | {report['action_top3']:.4f} | "
            f"{report['action_top5']:.4f} | {report['action_macro_f1']:.4f} |"
        )
        rows.append(
            f"| {split_name} random seed {random_summary['seed']} | {random_report['n_clips']} | {len(labels)} | "
            f"{random_report['action_top1']:.4f} | {random_report['action_top3']:.4f} | "
            f"{random_report['action_top5']:.4f} | {random_report['action_macro_f1']:.4f} |"
        )
    metric_table = "\n".join(rows)
    expected = []
    for split_name in ("fine", "coarse"):
        if split_name not in random_summaries:
            continue
        summary = random_summaries[split_name]
        expected.append(
            f"- {split_name}: random expected Top-1/Top-3/Top-5 "
            f"{summary['expected_top1']:.4f}/{summary['expected_top3']:.4f}/{summary['expected_top5']:.4f}; "
            f"macro-F1 MC mean {summary['macro_f1_trials']['mean']:.4f}"
        )
    return f"""# Assembly101 RGB Two Recordings

Recordings:
{rec_text}

Fine clips: {len(fine_clips)} requested, {evaluated.get('fine', 0)} evaluated, {len(fine_labels)} action labels.

Coarse clips: {len(coarse_clips)} requested, {evaluated.get('coarse', 0)} evaluated, {len(coarse_labels)} action labels.

GPU: cuda_available={gpu['cuda_available']}, encoder_device=`{gpu['encoder_device']}`, model_parameter_device=`{gpu.get('model_parameter_device', 'n/a')}`, gpu_inference_verified={gpu['gpu_inference_verified']}.

| run | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
{metric_table}

Expected random:
{chr(10).join(expected)}

Elapsed wall time: {elapsed_s:.1f}s.
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
