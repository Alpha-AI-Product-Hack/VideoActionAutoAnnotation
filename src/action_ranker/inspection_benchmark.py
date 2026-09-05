from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score

from action_ranker.encode_actions import encode_action_batch
from action_ranker.metrics import compute_action_metrics
from action_ranker.prompts import PROMPT_ID
from action_ranker.rank import gold_rank_1based, ranking_from_similarities
from action_ranker.taxonomies import REPO_ROOT
from action_ranker.types import PredictionRecord
from action_ranker.xclip_encoder import XClipEncoder

RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"


@dataclass(frozen=True)
class InspectionClip:
    clip_id: str
    dataset: str
    video_id: str
    media_path: str
    start_sec: float
    end_sec: float
    gold_action: str
    full_gold_action: str | None = None
    source_label_file: str | None = None
    phase: str | None = None
    recording: str | None = None
    view: str | None = None
    segment_index: int | None = None
    cut_path: str | None = None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "InspectionClip":
        return cls(
            clip_id=str(row["clip_id"]),
            dataset=str(row.get("dataset") or "unknown"),
            video_id=str(row.get("video_id") or Path(str(row["media_path"])).stem),
            media_path=str(row["media_path"]),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            gold_action=str(row["gold_action"]),
            full_gold_action=row.get("full_gold_action"),
            source_label_file=row.get("source_label_file"),
            phase=row.get("phase"),
            recording=row.get("recording"),
            view=row.get("view"),
            segment_index=row.get("segment_index"),
            cut_path=row.get("cut_path"),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "xclip":
        return _run_xclip(args)
    if args.command == "merge":
        return _run_merge(args)
    raise SystemExit(f"Unknown command: {args.command}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspection benchmark helpers for VLM vs X-CLIP")
    sub = parser.add_subparsers(dest="command", required=True)

    xclip = sub.add_parser("xclip", help="Evaluate X-CLIP on an inspection clips file")
    xclip.add_argument("--clips-file", required=True)
    xclip.add_argument("--bank-file", required=True)
    xclip.add_argument("--run-id", required=True)
    xclip.add_argument("--dictionary-id", default="assembly101_coarse_verb10_inspection")
    xclip.add_argument("--model-id", default="microsoft/xclip-base-patch32")
    xclip.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    xclip.add_argument("--num-frames", type=int, default=8)
    xclip.add_argument("--batch-size", type=int, default=4)
    xclip.add_argument("--seed", type=int, default=0)
    xclip.add_argument("--random-trials", type=int, default=10000)

    merge = sub.add_parser("merge", help="Merge X-CLIP and VLM predictions into inspection artifacts")
    merge.add_argument("--clips-file", required=True)
    merge.add_argument("--bank-file", required=True)
    merge.add_argument("--xclip-predictions", required=True)
    merge.add_argument("--vlm-predictions", required=True)
    merge.add_argument("--xclip-metrics", required=True)
    merge.add_argument("--vlm-metrics", required=True)
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--write-review-clips", action="store_true")
    merge.add_argument("--review-fps", type=float, default=12.0)
    merge.add_argument("--review-max-width", type=int, default=720)
    merge.add_argument("--overwrite-review-clips", action="store_true")
    return parser.parse_args(argv)


def _run_xclip(args: argparse.Namespace) -> int:
    clips_payload = _read_json(Path(args.clips_file))
    clips = [InspectionClip.from_json(row) for row in clips_payload.get("clips", [])]
    labels = _load_labels(Path(args.bank_file))
    if not clips:
        raise SystemExit("No clips to evaluate")
    if not labels:
        raise SystemExit("No labels in bank")

    run_dir = RUNS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoder = XClipEncoder(model_id=args.model_id, device=args.device, num_frames=args.num_frames)
    gpu = _gpu_info(encoder)
    text_embeddings = encode_action_batch(labels, encoder, PROMPT_ID)

    scored = _score_clips_sparse(
        encoder=encoder,
        clips=clips,
        labels=labels,
        text_embeddings=text_embeddings,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
    )
    records = _prediction_records(
        scored=scored,
        labels=labels,
        dictionary_id=args.dictionary_id,
        encoder_id=encoder.encoder_id,
        frame_count=args.num_frames,
    )
    report = compute_action_metrics(
        records,
        args.dictionary_id,
        n_skipped_intervals=len(clips) - len(records),
    )
    random_records, random_report, random_summary = _random_baseline(
        clips=[row["clip"] for row in scored],
        labels=labels,
        dictionary_id=args.dictionary_id,
        seed=args.seed,
        n_trials=args.random_trials,
    )
    elapsed_s = time.perf_counter() - started

    _write_json(run_dir / f"metrics_{args.dictionary_id}.json", report.to_dict())
    _write_json(run_dir / f"metrics_{args.dictionary_id}_random_seed{args.seed}.json", random_report.to_dict())
    _write_json(run_dir / "random_summary.json", random_summary)
    _write_jsonl(run_dir / "predictions.jsonl", [record.to_dict() for record in records])
    _write_jsonl(run_dir / f"predictions_random_seed{args.seed}.jsonl", [record.to_dict() for record in random_records])
    _write_jsonl(run_dir / "rankings.jsonl", [_ranking_row(row, labels) for row in scored])
    _write_json(
        run_dir / "run_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "inspection-xclip-sparse",
            "args": vars(args),
            "dictionary_id": args.dictionary_id,
            "encoder_id": encoder.encoder_id,
            "prompt_id": PROMPT_ID,
            "frame_count": args.num_frames,
            "n_labels": len(labels),
            "n_input_clips": len(clips),
            "n_scored_clips": len(records),
            "gpu": gpu,
            "elapsed_s": elapsed_s,
        },
    )
    (run_dir / "summary.md").write_text(
        _xclip_summary(report.to_dict(), random_report.to_dict(), random_summary, labels, gpu, elapsed_s),
        encoding="utf-8",
    )
    print(run_dir)
    print(json.dumps({"metrics": report.to_dict(), "random": random_report.to_dict(), "gpu": gpu}, indent=2))
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [InspectionClip.from_json(row) for row in _read_json(Path(args.clips_file)).get("clips", [])]
    labels = _load_labels(Path(args.bank_file))
    xclip = _read_predictions(Path(args.xclip_predictions))
    vlm = _read_predictions(Path(args.vlm_predictions))
    xclip_metrics = _read_json(Path(args.xclip_metrics))
    vlm_metrics = _read_json(Path(args.vlm_metrics))

    rows: list[dict[str, Any]] = []
    review_dir = out_dir / "review_clips"
    if args.write_review_clips:
        review_dir.mkdir(parents=True, exist_ok=True)

    for index, clip in enumerate(clips, start=1):
        xrow = xclip.get(_prediction_key(clip))
        vrow = vlm.get(_prediction_key(clip))
        review_path = ""
        if args.write_review_clips and xrow and vrow:
            review_path = str(
                _write_review_clip(
                    clip=clip,
                    index=index,
                    xclip_pred=str(xrow["pred_action"]),
                    vlm_pred=str(vrow["pred_action"]),
                    out_dir=review_dir,
                    fps=args.review_fps,
                    max_width=args.review_max_width,
                    overwrite=args.overwrite_review_clips,
                )
            )
        rows.append(
            {
                "index": index,
                "review_clip": review_path,
                "cut_path": clip.cut_path or "",
                "gold_verb": clip.gold_action,
                "coarse_action": clip.full_gold_action or "",
                "xclip_pred": _value(xrow, "pred_action"),
                "xclip_correct": _correct(xrow, clip.gold_action),
                "xclip_gold_rank": _value(xrow, "gold_rank"),
                "xclip_top5": _topk(xrow),
                "vlm_pred": _value(vrow, "pred_action"),
                "vlm_correct": _correct(vrow, clip.gold_action),
                "vlm_gold_rank": _value(vrow, "gold_rank"),
                "vlm_top5": _topk(vrow),
                "recording": clip.recording or "",
                "view": clip.view or "",
                "phase": clip.phase or "",
                "start_sec": f"{clip.start_sec:.3f}",
                "end_sec": f"{clip.end_sec:.3f}",
                "source_media": clip.media_path,
            }
        )

    _write_csv(out_dir / "inspection_table_predictions.csv", rows)
    (out_dir / "summary_predictions.md").write_text(
        _merged_summary(rows, labels, xclip_metrics, vlm_metrics),
        encoding="utf-8",
    )
    print(out_dir)
    print(json.dumps({"rows": len(rows), "review_clips": sum(bool(row["review_clip"]) for row in rows)}, indent=2))
    return 0


def _score_clips_sparse(
    *,
    encoder: XClipEncoder,
    clips: list[InspectionClip],
    labels: list[str],
    text_embeddings: np.ndarray,
    batch_size: int,
    num_frames: int,
) -> list[dict[str, Any]]:
    by_media: dict[str, list[InspectionClip]] = defaultdict(list)
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
            started = time.perf_counter()
            scores = encoder.score_clip_texts(frame_batch, text_embeddings)
            inference_s = time.perf_counter() - started
            per_clip_s = inference_s / max(1, len(batch))
            for (clip, _), row_scores in zip(batch, scores, strict=True):
                scored.append(
                    {
                        "clip": clip,
                        "scores": row_scores.astype(np.float32),
                        "inference_s": float(per_clip_s),
                    }
                )
    return scored


def _sample_video_once(path: Path, clips: list[InspectionClip], *, num_frames: int) -> dict[str, np.ndarray]:
    import av

    targets: list[tuple[float, str, int]] = []
    for clip in clips:
        for frame_index, target_sec in enumerate(np.linspace(clip.start_sec, clip.end_sec, num_frames)):
            targets.append((float(target_sec), clip.clip_id, frame_index))
    targets.sort(key=lambda row: row[0])
    if not targets:
        return {}

    output = {clip.clip_id: np.zeros((num_frames, 3, 224, 224), dtype=np.float32) for clip in clips}
    filled = {clip.clip_id: np.zeros(num_frames, dtype=bool) for clip in clips}
    target_index = 0
    last_chw: np.ndarray | None = None

    container = av.open(str(path))
    try:
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


def _prediction_records(
    *,
    scored: list[dict[str, Any]],
    labels: list[str],
    dictionary_id: str,
    encoder_id: str,
    frame_count: int,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for row in scored:
        clip: InspectionClip = row["clip"]
        ranking = ranking_from_similarities(row["scores"], labels, clip_id=clip.clip_id)
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
                frame_count=frame_count,
                inference_s=row.get("inference_s"),
                decode_s=None,
                encode_s=None,
                rank_s=0.0,
            )
        )
    return records


def _random_baseline(
    *,
    clips: list[InspectionClip],
    labels: list[str],
    dictionary_id: str,
    seed: int,
    n_trials: int,
) -> tuple[list[PredictionRecord], Any, dict[str, Any]]:
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
    report = compute_action_metrics(records, dictionary_id)

    golds = [clip.gold_action for clip in clips]
    trial_f1 = []
    trial_top1 = []
    for _ in range(n_trials):
        pred_ids = rng.integers(0, len(labels), size=len(clips))
        preds = [labels[int(index)] for index in pred_ids]
        trial_f1.append(float(f1_score(golds, preds, average="macro", zero_division=0)))
        trial_top1.append(float(np.mean([gold == pred for gold, pred in zip(golds, preds, strict=True)])))
    summary = {
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
    return records, report, summary


def _ranking_row(row: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    clip: InspectionClip = row["clip"]
    ranking = ranking_from_similarities(row["scores"], labels, clip_id=clip.clip_id)
    return {
        **clip.to_json(),
        "labels": ranking.labels,
        "cosine_similarity": ranking.cosine_similarity,
        "cosine_distance": ranking.cosine_distance,
        "pred_action": ranking.pred_action,
    }


def _write_review_clip(
    *,
    clip: InspectionClip,
    index: int,
    xclip_pred: str,
    vlm_pred: str,
    out_dir: Path,
    fps: float,
    max_width: int,
    overwrite: bool,
) -> Path:
    filename = (
        f"{index:02d}_gold-{_slug(clip.gold_action)}__xclip-{_slug(xclip_pred)}__"
        f"vlm-{_slug(vlm_pred)}__{_slug(clip.full_gold_action or clip.gold_action)}.mp4"
    )
    out_path = out_dir / filename
    if out_path.is_file() and not overwrite:
        return out_path
    _cut_review_clip(clip, out_path=out_path, xclip_pred=xclip_pred, vlm_pred=vlm_pred, fps=fps, max_width=max_width)
    return out_path


def _cut_review_clip(
    clip: InspectionClip,
    *,
    out_path: Path,
    xclip_pred: str,
    vlm_pred: str,
    fps: float,
    max_width: int,
) -> None:
    import av

    container = av.open(clip.media_path)
    output = av.open(str(out_path), mode="w")
    try:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "AUTO"
        width, height = _preview_size(in_stream.width, in_stream.height, max_width=max_width)
        out_stream = output.add_stream(_pick_encoder(av), rate=int(round(fps)))
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"

        next_emit = clip.start_sec
        emitted = 0
        for frame in container.decode(in_stream):
            ts = _frame_time(frame, in_stream)
            if ts < clip.start_sec:
                continue
            if ts > clip.end_sec:
                break
            if ts + 1e-6 < next_emit:
                continue
            image = frame.to_image().convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            _draw_review_overlay(image, clip=clip, xclip_pred=xclip_pred, vlm_pred=vlm_pred)
            out_frame = av.VideoFrame.from_image(image)
            for packet in out_stream.encode(out_frame):
                output.mux(packet)
            emitted += 1
            next_emit += 1.0 / fps
        if emitted == 0:
            raise RuntimeError(f"No frames emitted for {clip.clip_id}")
        for packet in out_stream.encode():
            output.mux(packet)
    finally:
        output.close()
        container.close()


def _draw_review_overlay(image: Image.Image, *, clip: InspectionClip, xclip_pred: str, vlm_pred: str) -> None:
    draw = ImageDraw.Draw(image)
    lines = [
        f"gold: {clip.gold_action} | action: {clip.full_gold_action or clip.gold_action}",
        f"xclip: {xclip_pred} | vlm: {vlm_pred}",
    ]
    box_h = 44
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0))
    draw.text((8, 6), lines[0][:130], fill=(255, 255, 255))
    draw.text((8, 24), lines[1][:130], fill=(255, 255, 255))


def _load_labels(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "label" in reader.fieldnames:
            return [row["label"].strip() for row in reader if row.get("label", "").strip()]
    rows = path.read_text(encoding="utf-8").splitlines()
    return [row.strip() for row in rows if row.strip() and row.strip().lower() != "label"]


def _read_predictions(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[_prediction_key(row)] = row
    return rows


def _prediction_key(row: InspectionClip | dict[str, Any]) -> tuple[str, str, str, str]:
    if isinstance(row, InspectionClip):
        return (row.dataset, row.video_id, f"{row.start_sec:.3f}", f"{row.end_sec:.3f}")
    return (
        str(row.get("dataset") or ""),
        str(row.get("video_id") or ""),
        f"{float(row.get('start_sec')):.3f}",
        f"{float(row.get('end_sec')):.3f}",
    )


def _value(row: dict[str, Any] | None, key: str) -> str:
    if not row:
        return ""
    value = row.get(key)
    return "" if value is None else str(value)


def _correct(row: dict[str, Any] | None, gold: str) -> str:
    if not row:
        return ""
    return "1" if row.get("pred_action") == gold else "0"


def _topk(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return " | ".join(str(item) for item in row.get("topk_labels", []))


def _frame_time(frame: Any, stream: Any) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None:
        return 0.0
    return float(frame.pts * stream.time_base)


def _pick_encoder(av: Any) -> str:
    for name in ("libx264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:  # noqa: BLE001
            continue
    return "mpeg4"


def _preview_size(width: int, height: int, *, max_width: int) -> tuple[int, int]:
    if width <= max_width:
        out_w, out_h = width, height
    else:
        scale = max_width / float(width)
        out_w = max_width
        out_h = int(round(height * scale))
    out_w = max(2, out_w - out_w % 2)
    out_h = max(2, out_h - out_h % 2)
    return out_w, out_h


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text[:80] or "item"


def _gpu_info(encoder: XClipEncoder) -> dict[str, Any]:
    torch = encoder.torch
    info: dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "encoder_device": str(encoder.device),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
    try:
        info["model_parameter_device"] = str(next(encoder.model.parameters()).device)
    except StopIteration:
        info["model_parameter_device"] = "unknown"
    info["gpu_inference_verified"] = bool(torch.cuda.is_available() and "cuda" in info["model_parameter_device"])
    return info


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "p05": float(np.quantile(values, 0.05)) if values.size else 0.0,
        "p50": float(np.quantile(values, 0.50)) if values.size else 0.0,
        "p95": float(np.quantile(values, 0.95)) if values.size else 0.0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _xclip_summary(
    report: dict[str, Any],
    random_report: dict[str, Any],
    random_summary: dict[str, Any],
    labels: list[str],
    gpu: dict[str, Any],
    elapsed_s: float,
) -> str:
    return f"""# X-CLIP Inspection Benchmark

Labels: {', '.join(f'`{label}`' for label in labels)}

GPU inference verified: `{gpu.get('gpu_inference_verified')}` ({gpu.get('cuda_device_name', 'no cuda')})

Elapsed: {elapsed_s:.2f}s

| run | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| X-CLIP | {report['n_clips']} | {len(labels)} | {report['action_top1']:.4f} | {report['action_top3']:.4f} | {report['action_top5']:.4f} | {report['action_macro_f1']:.4f} |
| random seed 0 | {random_report['n_clips']} | {len(labels)} | {random_report['action_top1']:.4f} | {random_report['action_top3']:.4f} | {random_report['action_top5']:.4f} | {random_report['action_macro_f1']:.4f} |

Expected random Top-1/Top-3/Top-5: {random_summary['expected_top1']:.4f}/{random_summary['expected_top3']:.4f}/{random_summary['expected_top5']:.4f}.
Random macro-F1 Monte Carlo mean: {random_summary['macro_f1_trials']['mean']:.4f}.
"""


def _merged_summary(
    rows: list[dict[str, Any]],
    labels: list[str],
    xclip_metrics: dict[str, Any],
    vlm_metrics: dict[str, Any],
) -> str:
    body = [
        "# VLM vs X-CLIP Inspection",
        "",
        f"Labels: {', '.join(f'`{label}`' for label in labels)}",
        "",
        "| model | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| X-CLIP | {xclip_metrics['n_clips']} | {len(labels)} | "
            f"{xclip_metrics['action_top1']:.4f} | {xclip_metrics['action_top3']:.4f} | "
            f"{xclip_metrics['action_top5']:.4f} | {xclip_metrics['action_macro_f1']:.4f} |"
        ),
        (
            f"| VLM | {vlm_metrics['n_clips']} | {len(labels)} | "
            f"{vlm_metrics['action_top1']:.4f} | {vlm_metrics['action_top3']:.4f} | "
            f"{vlm_metrics['action_top5']:.4f} | {vlm_metrics['action_macro_f1']:.4f} |"
        ),
        "",
        "| # | gold | action | X-CLIP | VLM | review clip |",
        "|---:|---|---|---|---|---|",
    ]
    for row in rows:
        review = Path(row["review_clip"]).name if row.get("review_clip") else Path(row.get("cut_path", "")).name
        body.append(
            f"| {row['index']} | `{row['gold_verb']}` | `{row['coarse_action']}` | "
            f"`{row['xclip_pred']}` | `{row['vlm_pred']}` | {review} |"
        )
    body.append("")
    return "\n".join(body)


if __name__ == "__main__":
    raise SystemExit(main())
