from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score

from action_ranker.gt_clips import ClipDecodeError, sample_gt_clip
from action_ranker.metrics import compute_action_metrics
from action_ranker.rank import gold_rank_1based
from action_ranker.taxonomies import REPO_ROOT
from action_ranker.types import PredictionRecord

RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"
PROMPT_ID = "gemini_constrained_contact_sheet_v1"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(frozen=True)
class ClipItem:
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
    cut_path: str | None = None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "ClipItem":
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
            cut_path=row.get("cut_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeminiPrediction:
    clip: ClipItem
    model_id: str
    raw_text: str
    parsed_topk: list[str]
    topk_labels: list[str]
    invalid_labels: list[str]
    parse_error: str | None
    inference_s: float
    decode_s: float
    prompt_labels: list[str]
    contact_sheet_path: str
    finish_reason: str | None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    api_key = _read_api_key(args)
    out_dir = RUNS_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(Path(args.bank_file))
    clips_payload = json.loads(Path(args.clips_file).read_text(encoding="utf-8"))
    clips = [ClipItem.from_json(row) for row in clips_payload.get("clips", [])]
    clips = _select_clips(clips, limit=args.limit, distinct_gold=args.distinct_gold)
    if not labels:
        raise SystemExit("Empty label bank")
    if not clips:
        raise SystemExit("No clips selected")

    model_candidates, model_list_meta = _resolve_model_candidates(api_key, args)
    if not model_candidates:
        raise SystemExit("No Gemini generateContent models found")

    rows: list[GeminiPrediction] = []
    records: list[PredictionRecord] = []
    failures: list[dict[str, Any]] = []
    selected_model: str | None = None
    started = time.perf_counter()

    pred_path = out_dir / "predictions.jsonl"
    raw_path = out_dir / "raw_responses.jsonl"
    with pred_path.open("w", encoding="utf-8") as pred_handle, raw_path.open("w", encoding="utf-8") as raw_handle:
        for index, clip in enumerate(clips, start=1):
            candidate_order = [selected_model] if selected_model else list(model_candidates)
            if selected_model is None:
                candidate_order = list(model_candidates)
            prediction = None
            last_error: dict[str, Any] | None = None
            for model_id in candidate_order:
                if model_id is None:
                    continue
                try:
                    prediction = _classify_clip(
                        clip=clip,
                        labels=labels,
                        model_id=model_id,
                        api_key=api_key,
                        out_dir=out_dir,
                        num_frames=args.num_frames,
                        tile_size=args.tile_size,
                        cols=args.sheet_cols,
                        top_k=min(args.top_k, len(labels)),
                        max_output_tokens=args.max_output_tokens,
                        overwrite_sheet=args.overwrite_sheets,
                    )
                    selected_model = model_id
                    break
                except ClipDecodeError as exc:
                    last_error = {"clip": clip.to_dict(), "model_id": model_id, "reason": "decode_error", "error": str(exc)}
                    break
                except GeminiApiError as exc:
                    last_error = {"clip": clip.to_dict(), "model_id": model_id, "reason": "api_error", **exc.to_dict()}
                    if selected_model is not None or not exc.is_fallback_candidate:
                        break
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_error = {"clip": clip.to_dict(), "model_id": model_id, "reason": "unexpected_error", "error": repr(exc)}
                    if args.stop_on_error:
                        raise
                    continue
            if prediction is None:
                if last_error is not None:
                    failures.append(last_error)
                continue

            rows.append(prediction)
            record = _prediction_record(prediction, dictionary_id=args.dictionary_id, frame_count=args.num_frames)
            records.append(record)
            pred_handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            raw_handle.write(json.dumps(_raw_row(prediction), ensure_ascii=False) + "\n")
            pred_handle.flush()
            raw_handle.flush()
            print(
                json.dumps(
                    {
                        "clip": index,
                        "n_clips": len(clips),
                        "model": selected_model,
                        "gold": clip.gold_action,
                        "pred": record.pred_action,
                        "topk": record.topk_labels,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report = compute_action_metrics(records, args.dictionary_id, n_skipped_intervals=len(clips) - len(records))
    random_records, random_report, random_summary = _random_baseline(
        clips=[row.clip for row in rows],
        labels=labels,
        dictionary_id=args.dictionary_id,
        seed=args.seed,
        n_trials=args.random_trials,
    )
    elapsed_s = time.perf_counter() - started

    _write_json(out_dir / f"metrics_{args.dictionary_id}.json", report.to_dict())
    _write_json(out_dir / f"metrics_{args.dictionary_id}_random_seed{args.seed}.json", random_report.to_dict())
    _write_json(out_dir / "random_summary.json", random_summary)
    _write_jsonl(out_dir / f"predictions_random_seed{args.seed}.jsonl", [row.to_dict() for row in random_records])
    _write_jsonl(out_dir / "failure_log.jsonl", failures)
    if args.write_review_clips:
        _write_review_outputs(out_dir=out_dir, rows=rows, overwrite=args.overwrite_review_clips)
    _write_run_meta(
        out_dir=out_dir,
        args=args,
        labels=labels,
        clips=clips,
        model_candidates=model_candidates,
        selected_model=selected_model,
        model_list_meta=model_list_meta,
        records=records,
        failures=failures,
        elapsed_s=elapsed_s,
    )
    (out_dir / "summary.md").write_text(
        _summary_markdown(labels, rows, report.to_dict(), random_report.to_dict(), selected_model, failures),
        encoding="utf-8",
    )
    print(out_dir)
    print(json.dumps({"model": selected_model, "metrics": report.to_dict(), "failures": failures}, indent=2, ensure_ascii=False))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained Gemini VLM classification over GT-cut clips")
    parser.add_argument("--clips-file", required=True)
    parser.add_argument("--bank-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dictionary-id", required=True)
    parser.add_argument("--model-id", default="auto", help="Gemini model id, or auto")
    parser.add_argument("--candidate-model", action="append", help="Candidate model id; repeat to override auto order")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--distinct-gold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--overwrite-sheets", action="store_true")
    parser.add_argument("--write-review-clips", action="store_true")
    parser.add_argument("--overwrite-review-clips", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args(argv)


def _read_api_key(args: argparse.Namespace) -> str:
    if args.api_key_stdin:
        key = sys.stdin.readline().strip()
    else:
        key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise SystemExit(f"Gemini API key is empty; set {args.api_key_env} or use --api-key-stdin")
    return key


def _load_labels(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "label" in reader.fieldnames:
            return [row["label"].strip() for row in reader if row.get("label", "").strip()]
    return [row.strip() for row in path.read_text(encoding="utf-8").splitlines() if row.strip() and row.strip() != "label"]


def _select_clips(clips: list[ClipItem], *, limit: int | None, distinct_gold: bool) -> list[ClipItem]:
    if limit is None:
        return clips
    if not distinct_gold:
        return clips[:limit]
    out: list[ClipItem] = []
    seen: set[str] = set()
    for clip in clips:
        if clip.gold_action in seen:
            continue
        seen.add(clip.gold_action)
        out.append(clip)
        if len(out) >= limit:
            return out
    for clip in clips:
        if clip in out:
            continue
        out.append(clip)
        if len(out) >= limit:
            break
    return out


def _resolve_model_candidates(api_key: str, args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if args.candidate_model:
        return [_normalize_model_id(model) for model in args.candidate_model], {"source": "args"}
    if args.model_id != "auto":
        return [_normalize_model_id(args.model_id)], {"source": "args"}
    try:
        payload = _api_get(f"{API_ROOT}/models?key={api_key}")
    except GeminiApiError as exc:
        fallback = ["models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.0-flash"]
        return fallback, {"source": "fallback_after_list_error", "list_error": exc.to_dict()}
    models = payload.get("models", [])
    generate_models = [model for model in models if "generateContent" in model.get("supportedGenerationMethods", [])]
    candidates = [_normalize_model_id(model["name"]) for model in generate_models if _looks_like_text_vlm(model.get("name", ""))]
    candidates = sorted(set(candidates), key=_model_priority, reverse=True)
    return candidates, {
        "source": "models_list",
        "n_models": len(models),
        "n_generate_content_models": len(generate_models),
        "top_candidates": candidates[:10],
    }


def _looks_like_text_vlm(model_name: str) -> bool:
    name = model_name.lower()
    if "gemini" not in name:
        return False
    banned = ["embedding", "imagen", "image-generation", "tts", "aqa"]
    return not any(item in name for item in banned)


def _model_priority(model_name: str) -> tuple[Any, ...]:
    name = model_name.lower().removeprefix("models/")
    numbers = tuple(int(part) for part in re.findall(r"\d+", name)[:3])
    numbers = numbers + (0,) * (3 - len(numbers))
    tier = 0
    if "pro" in name:
        tier = 100
    elif "flash" in name:
        tier = 60
    elif "lite" in name:
        tier = 30
    stable = 0 if "preview" in name or "experimental" in name or "exp" in name else 10
    return (*numbers, tier, stable, name)


def _normalize_model_id(model_id: str) -> str:
    return model_id if model_id.startswith("models/") else f"models/{model_id}"


def _classify_clip(
    *,
    clip: ClipItem,
    labels: list[str],
    model_id: str,
    api_key: str,
    out_dir: Path,
    num_frames: int,
    tile_size: int,
    cols: int,
    top_k: int,
    max_output_tokens: int,
    overwrite_sheet: bool,
) -> GeminiPrediction:
    sheet_path = _contact_sheet_path(out_dir, clip)
    decode_started = time.perf_counter()
    if overwrite_sheet or not sheet_path.is_file():
        frames = sample_gt_clip(clip.media_path, clip.start_sec, clip.end_sec, num_frames=num_frames, height=tile_size, width=tile_size)
        if frames is None:
            raise ClipDecodeError("invalid interval")
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        image = _contact_sheet(frames, cols=cols)
        image.save(sheet_path, quality=92)
    else:
        image = Image.open(sheet_path).convert("RGB")
    decode_s = time.perf_counter() - decode_started

    prompt = _build_prompt(labels, top_k=top_k)
    raw_text, finish_reason, inference_s = _generate(
        model_id=model_id,
        api_key=api_key,
        image=image,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
    )
    parsed = _parse_topk(raw_text, labels=labels, top_k=top_k)
    return GeminiPrediction(
        clip=clip,
        model_id=model_id,
        raw_text=raw_text,
        parsed_topk=parsed["parsed_topk"],
        topk_labels=parsed["topk_labels"],
        invalid_labels=parsed["invalid_labels"],
        parse_error=parsed["parse_error"],
        inference_s=float(inference_s),
        decode_s=float(decode_s),
        prompt_labels=labels,
        contact_sheet_path=str(sheet_path),
        finish_reason=finish_reason,
    )


def _generate(*, model_id: str, api_key: str, image: Image.Image, prompt: str, max_output_tokens: int) -> tuple[str, str | None, float]:
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG", quality=92)
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(image_bytes.getvalue()).decode("ascii")}},
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "candidateCount": 1,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    started = time.perf_counter()
    payload = _api_post(f"{API_ROOT}/{model_id}:generateContent?key={api_key}", data)
    inference_s = time.perf_counter() - started
    candidates = payload.get("candidates") or []
    if not candidates:
        raise GeminiApiError(200, "empty_candidates", json.dumps(payload)[:500])
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
    return text.strip(), candidate.get("finishReason"), float(inference_s)


def _api_get(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    return _api_request(request)


def _api_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    return _api_request(request)


def _api_request(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = exc.code
        message = body[:2000]
        try:
            parsed = json.loads(body)
            message = parsed.get("error", {}).get("message", message)
        except json.JSONDecodeError:
            pass
        raise GeminiApiError(status, message, body[:2000]) from exc
    except urllib.error.URLError as exc:
        raise GeminiApiError(0, str(exc.reason), "") from exc


class GeminiApiError(Exception):
    def __init__(self, status: int, message: str, body: str):
        super().__init__(f"Gemini API error {status}: {message}")
        self.status = int(status)
        self.message = message
        self.body = body

    @property
    def is_fallback_candidate(self) -> bool:
        return self.status in {400, 403, 404, 429, 500, 503}

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "message": self.message, "body_excerpt": self.body[:500]}


def _build_prompt(labels: list[str], *, top_k: int) -> str:
    label_lines = "\n".join(f"- {label}" for label in labels)
    return f"""You are classifying the action shown in an ordered contact sheet of video frames.
Frames are ordered left to right, top to bottom. Use temporal order, hand motion, tool use, and object state changes.
The allowed labels are official Assembly101 verb classes. Ignore object identity unless it helps distinguish the motion.

Choose exactly {top_k} labels from the allowed labels. Use only exact strings from the list.
Return JSON only, no prose, in this format:
{{"top5": ["label1", "label2", "label3", "label4", "label5"]}}

Allowed labels:
{label_lines}
"""


def _parse_topk(raw_text: str, *, labels: list[str], top_k: int) -> dict[str, Any]:
    allowed = set(labels)
    parsed: list[str] = []
    parse_error = None
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                data = None
                parse_error = str(exc)
        else:
            data = None
            parse_error = "no_json_object"
    if isinstance(data, dict):
        value = data.get("top5") or data.get("top_k") or data.get("labels") or data.get("predictions")
        if isinstance(value, list):
            parsed = [str(item).strip() for item in value]
        elif isinstance(value, str):
            parsed = [value.strip()]
    elif isinstance(data, list):
        parsed = [str(item).strip() for item in data]
    if not parsed and parse_error is None:
        parse_error = "json_without_topk"

    out: list[str] = []
    invalid: list[str] = []
    label_lookup = {label.lower(): label for label in labels}
    for item in parsed:
        label = item if item in allowed else label_lookup.get(item.lower())
        if label and label not in out:
            out.append(label)
        elif item not in allowed:
            invalid.append(item)
    for label in labels:
        if len(out) >= top_k:
            break
        if label not in out:
            out.append(label)
    return {
        "parsed_topk": parsed,
        "topk_labels": out[:top_k],
        "invalid_labels": invalid,
        "parse_error": parse_error,
    }


def _prediction_record(row: GeminiPrediction, *, dictionary_id: str, frame_count: int) -> PredictionRecord:
    topk = row.topk_labels
    gold_rank = topk.index(row.clip.gold_action) + 1 if row.clip.gold_action in topk else None
    if gold_rank is None and row.clip.gold_action in row.prompt_labels:
        # Gemini returns only top-k, so a non-top-k official label is present but rank is unknown.
        gold_rank = gold_rank_1based_labels(row.prompt_labels, row.clip.gold_action, floor=len(topk) + 1)
    return PredictionRecord(
        dataset=row.clip.dataset,
        video_id=row.clip.video_id,
        start_sec=row.clip.start_sec,
        end_sec=row.clip.end_sec,
        gold_action=row.clip.gold_action,
        pred_action=topk[0] if topk else "",
        topk_labels=topk,
        topk_scores=[float(1.0 - index / max(1, len(topk))) for index in range(len(topk))],
        dictionary_id=dictionary_id,
        encoder_id=row.model_id,
        prompt_id=PROMPT_ID,
        gold_rank=gold_rank,
        frame_count=frame_count,
        inference_s=row.inference_s,
        decode_s=row.decode_s,
        encode_s=0.0,
        rank_s=0.0,
    )


def gold_rank_1based_labels(labels: list[str], gold: str, *, floor: int) -> int | None:
    if gold not in labels:
        return None
    return max(floor, labels.index(gold) + 1)


def _random_baseline(
    *,
    clips: list[ClipItem],
    labels: list[str],
    dictionary_id: str,
    seed: int,
    n_trials: int,
) -> tuple[list[PredictionRecord], Any, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[PredictionRecord] = []
    for clip in clips:
        order = list(labels)
        rng.shuffle(order)
        topk = order[: min(5, len(order))]
        records.append(
            PredictionRecord(
                dataset=clip.dataset,
                video_id=clip.video_id,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                gold_action=clip.gold_action,
                pred_action=topk[0],
                topk_labels=topk,
                topk_scores=[float(1.0 - index / max(1, len(topk))) for index in range(len(topk))],
                dictionary_id=dictionary_id,
                encoder_id=f"random-uniform-seed-{seed}",
                prompt_id="random_uniform",
                gold_rank=order.index(clip.gold_action) + 1 if clip.gold_action in order else None,
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
    return records, report, {
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


def _contact_sheet_path(out_dir: Path, clip: ClipItem) -> Path:
    digest = re.sub(r"[^a-zA-Z0-9]+", "-", clip.clip_id.lower()).strip("-")[:120]
    return out_dir / "contact_sheets" / f"{digest}.jpg"


def _contact_sheet(frames: np.ndarray, *, cols: int) -> Image.Image:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must be [T, C, H, W]")
    t, _, h, w = frames.shape
    cols = max(1, min(cols, t))
    rows = int(np.ceil(t / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index in range(t):
        hwc = np.transpose(np.clip(frames[index] * 255.0, 0, 255).astype(np.uint8), (1, 2, 0))
        tile = Image.fromarray(hwc, mode="RGB")
        x = (index % cols) * w
        y = (index // cols) * h
        sheet.paste(tile, (x, y))
        draw.rectangle((x + 4, y + 4, x + 34, y + 28), fill=(0, 0, 0))
        draw.text((x + 12, y + 8), str(index + 1), fill=(255, 255, 255))
    return sheet


def _write_review_outputs(*, out_dir: Path, rows: list[GeminiPrediction], overwrite: bool) -> None:
    review_dir = out_dir / "review_clips"
    review_dir.mkdir(parents=True, exist_ok=True)
    table: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        review_path = _write_review_clip(row, index=index, out_dir=review_dir, overwrite=overwrite)
        table.append(
            {
                "index": index,
                "review_clip": str(review_path),
                "cut_path": row.clip.cut_path or "",
                "gold_verb": row.clip.gold_action,
                "full_action": row.clip.full_gold_action or "",
                "gemini_pred": row.topk_labels[0] if row.topk_labels else "",
                "gemini_correct": "1" if row.topk_labels and row.topk_labels[0] == row.clip.gold_action else "0",
                "gemini_top5": " | ".join(row.topk_labels),
                "model_id": row.model_id,
                "contact_sheet": row.contact_sheet_path,
                "recording": row.clip.recording or "",
                "view": row.clip.view or "",
                "phase": row.clip.phase or "",
                "start_sec": f"{row.clip.start_sec:.3f}",
                "end_sec": f"{row.clip.end_sec:.3f}",
                "source_media": row.clip.media_path,
            }
        )
    _write_csv(out_dir / "inspection_table_gemini.csv", table)


def _write_review_clip(row: GeminiPrediction, *, index: int, out_dir: Path, overwrite: bool) -> Path:
    clip = row.clip
    pred = row.topk_labels[0] if row.topk_labels else "unknown"
    filename = f"{index:02d}_gold-{_slug(clip.gold_action)}__gemini-{_slug(pred)}__{_slug(clip.full_gold_action or clip.gold_action)}.mp4"
    out_path = out_dir / filename
    if out_path.is_file() and not overwrite:
        return out_path
    _cut_review_clip(clip, out_path=out_path, pred=pred, model_id=row.model_id)
    return out_path


def _cut_review_clip(clip: ClipItem, *, out_path: Path, pred: str, model_id: str) -> None:
    import av

    container = av.open(clip.media_path)
    output = av.open(str(out_path), mode="w")
    try:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "AUTO"
        width, height = _preview_size(in_stream.width, in_stream.height, max_width=720)
        out_stream = output.add_stream(_pick_encoder(av), rate=12)
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
            _draw_overlay(image, clip=clip, pred=pred, model_id=model_id)
            out_frame = av.VideoFrame.from_image(image)
            for packet in out_stream.encode(out_frame):
                output.mux(packet)
            emitted += 1
            next_emit += 1.0 / 12.0
        if emitted == 0:
            raise RuntimeError(f"No frames emitted for {clip.clip_id}")
        for packet in out_stream.encode():
            output.mux(packet)
    finally:
        output.close()
        container.close()


def _draw_overlay(image: Image.Image, *, clip: ClipItem, pred: str, model_id: str) -> None:
    draw = ImageDraw.Draw(image)
    lines = [
        f"gold: {clip.gold_action} | action: {clip.full_gold_action or clip.gold_action}",
        f"gemini: {pred} | model: {model_id.removeprefix('models/')}",
    ]
    draw.rectangle((0, 0, image.width, 44), fill=(0, 0, 0))
    draw.text((8, 6), lines[0][:130], fill=(255, 255, 255))
    draw.text((8, 24), lines[1][:130], fill=(255, 255, 255))


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
    return max(2, out_w - out_w % 2), max(2, out_h - out_h % 2)


def _raw_row(row: GeminiPrediction) -> dict[str, Any]:
    return {
        "clip": row.clip.to_dict(),
        "model_id": row.model_id,
        "raw_text": row.raw_text,
        "parsed_topk": row.parsed_topk,
        "topk_labels": row.topk_labels,
        "invalid_labels": row.invalid_labels,
        "parse_error": row.parse_error,
        "finish_reason": row.finish_reason,
        "inference_s": row.inference_s,
        "decode_s": row.decode_s,
        "contact_sheet_path": row.contact_sheet_path,
    }


def _write_run_meta(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    labels: list[str],
    clips: list[ClipItem],
    model_candidates: list[str],
    selected_model: str | None,
    model_list_meta: dict[str, Any],
    records: list[PredictionRecord],
    failures: list[dict[str, Any]],
    elapsed_s: float,
) -> None:
    _write_json(
        out_dir / "run_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "gemini-vlm-constrained-classification",
            "args": {key: value for key, value in vars(args).items() if key != "api_key_stdin"},
            "api_key_source": "stdin" if args.api_key_stdin else args.api_key_env,
            "dictionary_id": args.dictionary_id,
            "prompt_id": PROMPT_ID,
            "prompt_order_instruction": "Frames are ordered left to right, top to bottom.",
            "n_labels": len(labels),
            "labels": labels,
            "n_requested_clips": len(clips),
            "n_records": len(records),
            "n_failures": len(failures),
            "model_candidates": model_candidates,
            "selected_model": selected_model,
            "model_list_meta": model_list_meta,
            "elapsed_s": elapsed_s,
        },
    )


def _summary_markdown(
    labels: list[str],
    rows: list[GeminiPrediction],
    report: dict[str, Any],
    random_report: dict[str, Any],
    selected_model: str | None,
    failures: list[dict[str, Any]],
) -> str:
    body = [
        "# Gemini VLM Constrained Classification",
        "",
        f"Selected model: `{selected_model or 'none'}`",
        "",
        "Prompt states: `Frames are ordered left to right, top to bottom.`",
        "",
        f"Labels: {', '.join(f'`{label}`' for label in labels)}",
        "",
        "| run | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Gemini | {report['n_clips']} | {len(labels)} | {report['action_top1']:.4f} | "
            f"{report['action_top3']:.4f} | {report['action_top5']:.4f} | {report['action_macro_f1']:.4f} |"
        ),
        (
            f"| random seed 0 | {random_report['n_clips']} | {len(labels)} | {random_report['action_top1']:.4f} | "
            f"{random_report['action_top3']:.4f} | {random_report['action_top5']:.4f} | {random_report['action_macro_f1']:.4f} |"
        ),
        "",
        "| # | gold | action | Gemini | review clip |",
        "|---:|---|---|---|---|",
    ]
    for index, row in enumerate(rows, start=1):
        pred = row.topk_labels[0] if row.topk_labels else ""
        body.append(
            f"| {index} | `{row.clip.gold_action}` | `{row.clip.full_gold_action or ''}` | "
            f"`{pred}` | {Path(row.clip.cut_path or '').name} |"
        )
    if failures:
        body.extend(["", f"Failures: {len(failures)}. See `failure_log.jsonl`."])
    body.append("")
    return "\n".join(body)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "p05": float(np.quantile(values, 0.05)) if values.size else 0.0,
        "p50": float(np.quantile(values, 0.50)) if values.size else 0.0,
        "p95": float(np.quantile(values, 0.95)) if values.size else 0.0,
    }


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text[:80] or "item"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
