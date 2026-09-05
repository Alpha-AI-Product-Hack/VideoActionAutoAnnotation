from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score

from action_ranker.gt_clips import ClipDecodeError, sample_gt_clip
from action_ranker.metrics import compute_action_metrics
from action_ranker.taxonomies import REPO_ROOT
from action_ranker.types import PredictionRecord

PROMPT_ID = "qwen3_vl_constrained_contact_sheet_v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"


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

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "ClipItem":
        return cls(
            clip_id=str(row.get("clip_id") or _clip_id_from_row(row)),
            dataset=str(row.get("dataset") or "unknown"),
            video_id=str(row.get("video_id") or Path(str(row.get("media_path", ""))).stem),
            media_path=str(row["media_path"]),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            gold_action=str(row["gold_action"]),
            full_gold_action=row.get("full_gold_action"),
            source_label_file=row.get("source_label_file"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QwenPrediction:
    clip: ClipItem
    raw_text: str
    parsed_topk: list[str]
    topk_labels: list[str]
    invalid_labels: list[str]
    parse_error: str | None
    inference_s: float
    decode_s: float
    prompt_labels: list[str]
    contact_sheet_path: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = RUNS_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(Path(args.bank_file))
    if not labels:
        raise SystemExit("Empty label bank")
    if len(set(labels)) != len(labels):
        duplicates = [label for label, count in Counter(labels).items() if count > 1]
        raise SystemExit(f"Label bank contains duplicates: {duplicates}")

    clip_payload = json.loads(Path(args.clips_file).read_text(encoding="utf-8"))
    clips = [ClipItem.from_json(row) for row in clip_payload.get("clips", [])]
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit("No clips to classify")

    dictionary_id = args.dictionary_id or _default_dictionary_id(clip_payload, labels, len(clips))
    prompt_labels = _prompt_label_order(labels, seed=args.label_shuffle_seed)

    if args.prepare_only:
        prepared = _prepare_contact_sheets(
            clips=clips,
            out_dir=out_dir,
            num_frames=args.num_frames,
            tile_size=args.tile_size,
            cols=args.sheet_cols,
            context_sec=args.context_sec,
            overwrite=args.overwrite_sheets,
        )
        _write_prepare_summary(out_dir, args, labels, clips, prepared)
        print(out_dir)
        print(json.dumps({"prepared_contact_sheets": len(prepared), "labels": labels}, indent=2))
        return 0

    model_pack = _load_qwen_model(args)
    qwen_rows: list[QwenPrediction] = []
    records: list[PredictionRecord] = []
    skipped: list[dict[str, Any]] = []
    started = time.perf_counter()

    pred_path = out_dir / "predictions.jsonl"
    raw_path = out_dir / "raw_responses.jsonl"
    pred_handle = pred_path.open("w", encoding="utf-8")
    raw_handle = raw_path.open("w", encoding="utf-8")
    try:
        for index, clip in enumerate(clips, start=1):
            try:
                qwen_pred = _classify_clip(
                    clip=clip,
                    labels=labels,
                    prompt_labels=prompt_labels,
                    model_pack=model_pack,
                    out_dir=out_dir,
                    num_frames=args.num_frames,
                    tile_size=args.tile_size,
                    cols=args.sheet_cols,
                    context_sec=args.context_sec,
                    top_k=min(args.top_k, len(labels)),
                    max_new_tokens=args.max_new_tokens,
                    overwrite_sheet=args.overwrite_sheets,
                )
            except ClipDecodeError as exc:
                skipped.append({"clip": clip.to_dict(), "reason": "decode_error", "error": str(exc)})
                continue
            except Exception as exc:  # noqa: BLE001
                skipped.append({"clip": clip.to_dict(), "reason": "inference_error", "error": repr(exc)})
                if args.stop_on_error:
                    raise
                continue
            qwen_rows.append(qwen_pred)
            record = _prediction_record(
                qwen_pred=qwen_pred,
                labels=labels,
                dictionary_id=dictionary_id,
                encoder_id=args.model_id,
                frame_count=args.num_frames,
            )
            records.append(record)
            pred_handle.write(json.dumps(record.to_dict()) + "\n")
            raw_handle.write(json.dumps(_raw_response_row(qwen_pred)) + "\n")
            pred_handle.flush()
            raw_handle.flush()
            print(
                json.dumps(
                    {
                        "clip": index,
                        "n_clips": len(clips),
                        "gold": clip.gold_action,
                        "pred": record.pred_action,
                        "topk": record.topk_labels,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        pred_handle.close()
        raw_handle.close()

    report = compute_action_metrics(records, dictionary_id, n_skipped_intervals=len(skipped))
    random_records, random_report, random_summary = _random_baseline(
        clips=[row.clip for row in qwen_rows],
        labels=labels,
        dictionary_id=dictionary_id,
        seed=args.seed,
        n_trials=args.random_trials,
    )
    elapsed_s = time.perf_counter() - started

    _write_json(out_dir / f"metrics_{dictionary_id}.json", report.to_dict())
    _write_json(out_dir / f"metrics_{dictionary_id}_random_seed{args.seed}.json", random_report.to_dict())
    _write_json(out_dir / "random_summary.json", random_summary)
    _write_jsonl(out_dir / f"predictions_random_seed{args.seed}.jsonl", [row.to_dict() for row in random_records])
    _write_jsonl(out_dir / "skip_log.jsonl", skipped)
    _write_run_meta(
        out_dir=out_dir,
        args=args,
        dictionary_id=dictionary_id,
        labels=labels,
        prompt_labels=prompt_labels,
        clips=clips,
        records=records,
        skipped=skipped,
        model_pack=model_pack,
        elapsed_s=elapsed_s,
    )
    (out_dir / "summary.md").write_text(
        _summary_markdown(
            model_id=args.model_id,
            dictionary_id=dictionary_id,
            labels=labels,
            records=records,
            skipped=skipped,
            report=report.to_dict(),
            random_report=random_report.to_dict(),
            random_summary=random_summary,
            qwen_rows=qwen_rows,
            elapsed_s=elapsed_s,
            gpu=model_pack["gpu"],
        ),
        encoding="utf-8",
    )
    print(out_dir)
    print(json.dumps({"metrics": report.to_dict(), "random": random_report.to_dict()}, indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained Qwen-VL classification over GT-cut clips")
    parser.add_argument("--clips-file", required=True, help="JSON with a top-level clips array")
    parser.add_argument("--bank-file", required=True, help="CSV with a label column")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dictionary-id")
    parser.add_argument("--load-backend", default="hf", choices=["hf", "unsloth"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--device-map", default="none", help="Use none to avoid accelerate; or pass auto/cuda/etc.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["auto", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--unsloth-max-seq-length", type=int, default=512)
    parser.add_argument("--unsloth-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unsloth-fast-inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=336)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--context-sec", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--label-shuffle-seed", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-sheets", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args(argv)


def _load_labels(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "label" in reader.fieldnames:
            return [row["label"].strip() for row in reader if row.get("label", "").strip()]
    rows = path.read_text(encoding="utf-8").splitlines()
    return [row.strip() for row in rows if row.strip() and row.strip().lower() != "label"]


def _default_dictionary_id(payload: dict[str, Any], labels: list[str], n_clips: int) -> str:
    dataset = str(payload.get("dataset") or "custom").replace(" ", "_")
    target = int(payload.get("n_clips") or n_clips)
    return f"{dataset}_verb{len(labels)}_diverse{target}"


def _prompt_label_order(labels: list[str], seed: int | None) -> list[str]:
    ordered = list(labels)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(ordered)
    return ordered


def _load_qwen_model(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Qwen3-VL requires torch, transformers with Qwen3-VL support, and Pillow. "
            "Install/update the project torch extras first."
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false in this process")

    if args.load_backend == "unsloth":
        return _load_qwen_model_unsloth(args, torch=torch, device=device)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_kwargs: dict[str, Any] = {}
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    dtype = _torch_dtype(torch, args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["dtype"] = "auto"

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)

    if args.device_map == "none":
        model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    gpu = _gpu_info(torch, model, device=device, device_map=args.device_map)
    return {"torch": torch, "model": model, "processor": processor, "device": device, "gpu": gpu, "load_backend": "hf"}


def _load_qwen_model_unsloth(args: argparse.Namespace, *, torch: Any, device: str) -> dict[str, Any]:
    if device != "cuda":
        raise SystemExit("Unsloth VLM backend is intended for CUDA in this benchmark")
    # Import Unsloth before Transformers model classes so its patches are active.
    try:
        from unsloth import FastVisionModel
    except ImportError as exc:
        raise SystemExit("Install unsloth to use --load-backend unsloth") from exc

    dtype = _torch_dtype(torch, args.dtype)
    if args.device_map == "none":
        device_map: Any = {"": 0}
    else:
        device_map = args.device_map
    try:
        model, processor = FastVisionModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.unsloth_max_seq_length,
            dtype=dtype,
            load_in_4bit=bool(args.unsloth_load_in_4bit),
            device_map=device_map,
            trust_remote_code=False,
            offload_embedding=False,
        )
    except TypeError:
        model, processor = FastVisionModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.unsloth_max_seq_length,
            dtype=dtype,
            load_in_4bit=bool(args.unsloth_load_in_4bit),
            device_map=device_map,
            trust_remote_code=False,
        )
    if args.unsloth_fast_inference:
        FastVisionModel.for_inference(model)
    model.eval()
    gpu = _gpu_info(torch, model, device=device, device_map=str(device_map))
    gpu["unsloth_max_seq_length"] = int(args.unsloth_max_seq_length)
    gpu["unsloth_load_in_4bit"] = bool(args.unsloth_load_in_4bit)
    gpu["unsloth_fast_inference"] = bool(args.unsloth_fast_inference)
    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "device": device,
        "gpu": gpu,
        "load_backend": "unsloth",
    }


def _torch_dtype(torch: Any, name: str) -> Any | None:
    if name == "auto":
        return None
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def _gpu_info(torch: Any, model: Any, *, device: str, device_map: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "requested_device": device,
        "device_map": device_map,
    }
    if torch.cuda.is_available():
        info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated_before_forward_bytes"] = int(torch.cuda.memory_allocated())
    try:
        info["model_parameter_device"] = str(next(model.parameters()).device)
    except StopIteration:
        info["model_parameter_device"] = "unknown"
    info["gpu_inference_verified"] = bool(torch.cuda.is_available() and "cuda" in info["model_parameter_device"])
    return info


def _classify_clip(
    *,
    clip: ClipItem,
    labels: list[str],
    prompt_labels: list[str],
    model_pack: dict[str, Any],
    out_dir: Path,
    num_frames: int,
    tile_size: int,
    cols: int,
    context_sec: float,
    top_k: int,
    max_new_tokens: int,
    overwrite_sheet: bool,
) -> QwenPrediction:
    sheet_path = _contact_sheet_path(out_dir, clip)
    start = max(0.0, clip.start_sec - context_sec)
    end = clip.end_sec + context_sec
    decode_started = time.perf_counter()
    if overwrite_sheet or not sheet_path.is_file():
        frames = sample_gt_clip(
            clip.media_path,
            start,
            end,
            num_frames=num_frames,
            height=tile_size,
            width=tile_size,
        )
        if frames is None:
            raise ClipDecodeError("invalid interval")
        image = _contact_sheet(frames, cols=cols)
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(sheet_path, quality=92)
    else:
        image = Image.open(sheet_path).convert("RGB")
    decode_s = time.perf_counter() - decode_started

    prompt = _build_prompt(prompt_labels, top_k=top_k)
    raw_text, inference_s = _generate(model_pack, image, prompt, max_new_tokens=max_new_tokens)
    parsed = _parse_topk(raw_text, labels=labels, top_k=top_k)
    return QwenPrediction(
        clip=clip,
        raw_text=raw_text,
        parsed_topk=parsed["parsed_topk"],
        topk_labels=parsed["topk_labels"],
        invalid_labels=parsed["invalid_labels"],
        parse_error=parsed["parse_error"],
        inference_s=float(inference_s),
        decode_s=float(decode_s),
        prompt_labels=prompt_labels,
        contact_sheet_path=str(sheet_path),
    )


def _generate(model_pack: dict[str, Any], image: Image.Image, prompt: str, *, max_new_tokens: int) -> tuple[str, float]:
    torch = model_pack["torch"]
    model = model_pack["model"]
    processor = model_pack["processor"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(_input_device(model_pack))
    else:
        inputs = {key: value.to(_input_device(model_pack)) for key, value in inputs.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        model_pack["gpu"]["cuda_memory_allocated_after_forward_bytes"] = int(torch.cuda.memory_allocated())
        model_pack["gpu"]["gpu_inference_verified"] = True
    inference_s = time.perf_counter() - started
    input_len = int(inputs["input_ids"].shape[1])
    generated_trimmed = generated[:, input_len:]
    text_out = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text_out.strip(), float(inference_s)


def _input_device(model_pack: dict[str, Any]) -> str:
    model = model_pack["model"]
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return model_pack["device"]


def _build_prompt(labels: list[str], *, top_k: int) -> str:
    label_lines = "\n".join(f"- {label}" for label in labels)
    return f"""You are classifying the action shown in an ordered contact sheet of video frames.
Frames are ordered left to right, top to bottom. Use temporal order, hand motion, tool use, and object state changes.
The allowed labels are verbs or short verb phrases. Ignore object identity unless it helps distinguish the motion.

Choose exactly {top_k} labels from the allowed labels. Use only exact strings from the list.
Return JSON only, no prose, in this format:
{{"top5": ["label1", "label2", "label3", "label4", "label5"]}}

Allowed labels:
{label_lines}
"""


def _parse_topk(raw_text: str, *, labels: list[str], top_k: int) -> dict[str, Any]:
    valid = set(labels)
    normalized = {_normalize_label(label): label for label in labels}
    parse_error: str | None = None
    parsed_items: list[Any] = []
    try:
        payload = json.loads(_extract_json(raw_text))
        if isinstance(payload, dict):
            parsed_items = payload.get("top5") or payload.get("top_k") or payload.get("labels") or []
        elif isinstance(payload, list):
            parsed_items = payload
        else:
            parse_error = f"Unsupported JSON type: {type(payload).__name__}"
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
        parsed_items = _fallback_parse_list(raw_text)

    parsed_topk: list[str] = []
    invalid_labels: list[str] = []
    for item in parsed_items:
        label = str(item).strip()
        if label in valid:
            coerced = label
        else:
            coerced = normalized.get(_normalize_label(label))
        if coerced is None:
            invalid_labels.append(label)
            continue
        if coerced not in parsed_topk:
            parsed_topk.append(coerced)
        if len(parsed_topk) >= top_k:
            break

    topk_labels = list(parsed_topk)
    for label in labels:
        if len(topk_labels) >= top_k:
            break
        if label not in topk_labels:
            topk_labels.append(label)
    return {
        "parsed_topk": parsed_topk,
        "topk_labels": topk_labels,
        "invalid_labels": invalid_labels,
        "parse_error": parse_error,
    }


def _extract_json(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") or text.startswith("["):
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("No JSON object or array found")
    return match.group(1)


def _fallback_parse_list(raw_text: str) -> list[str]:
    lines = [line.strip(" -0123456789.\t\r") for line in raw_text.splitlines()]
    return [line.strip().strip('"\'') for line in lines if line.strip()]


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.casefold())


def _prediction_record(
    *,
    qwen_pred: QwenPrediction,
    labels: list[str],
    dictionary_id: str,
    encoder_id: str,
    frame_count: int,
) -> PredictionRecord:
    full_order = list(qwen_pred.topk_labels)
    full_order.extend(label for label in labels if label not in full_order)
    gold_rank = full_order.index(qwen_pred.clip.gold_action) + 1 if qwen_pred.clip.gold_action in full_order else None
    topk_scores = [float(1.0 - index / max(1, len(qwen_pred.topk_labels))) for index in range(len(qwen_pred.topk_labels))]
    return PredictionRecord(
        dataset=qwen_pred.clip.dataset,
        video_id=qwen_pred.clip.video_id,
        start_sec=qwen_pred.clip.start_sec,
        end_sec=qwen_pred.clip.end_sec,
        gold_action=qwen_pred.clip.gold_action,
        pred_action=qwen_pred.topk_labels[0] if qwen_pred.topk_labels else "",
        topk_labels=qwen_pred.topk_labels,
        topk_scores=topk_scores,
        dictionary_id=dictionary_id,
        encoder_id=encoder_id,
        prompt_id=PROMPT_ID,
        gold_rank=gold_rank,
        frame_count=frame_count,
        inference_s=qwen_pred.inference_s,
        decode_s=qwen_pred.decode_s,
        encode_s=0.0,
        rank_s=0.0,
    )


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


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "p05": float(np.quantile(values, 0.05)) if values.size else 0.0,
        "p50": float(np.quantile(values, 0.50)) if values.size else 0.0,
        "p95": float(np.quantile(values, 0.95)) if values.size else 0.0,
    }


def _prepare_contact_sheets(
    *,
    clips: list[ClipItem],
    out_dir: Path,
    num_frames: int,
    tile_size: int,
    cols: int,
    context_sec: float,
    overwrite: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip in clips:
        path = _contact_sheet_path(out_dir, clip)
        if overwrite or not path.is_file():
            frames = sample_gt_clip(
                clip.media_path,
                max(0.0, clip.start_sec - context_sec),
                clip.end_sec + context_sec,
                num_frames=num_frames,
                height=tile_size,
                width=tile_size,
            )
            if frames is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            _contact_sheet(frames, cols=cols).save(path, quality=92)
        rows.append({"clip_id": clip.clip_id, "path": str(path)})
    return rows


def _contact_sheet_path(out_dir: Path, clip: ClipItem) -> Path:
    digest = hashlib.sha1(clip.clip_id.encode("utf-8")).hexdigest()[:16]
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
        chw = frames[index]
        hwc = np.transpose(np.clip(chw * 255.0, 0, 255).astype(np.uint8), (1, 2, 0))
        tile = Image.fromarray(hwc, mode="RGB")
        x = (index % cols) * w
        y = (index // cols) * h
        sheet.paste(tile, (x, y))
        draw.rectangle((x + 4, y + 4, x + 34, y + 28), fill=(0, 0, 0))
        draw.text((x + 12, y + 8), str(index + 1), fill=(255, 255, 255))
    return sheet


def _raw_response_row(qwen_pred: QwenPrediction) -> dict[str, Any]:
    return {
        "clip": qwen_pred.clip.to_dict(),
        "raw_text": qwen_pred.raw_text,
        "parsed_topk": qwen_pred.parsed_topk,
        "topk_labels": qwen_pred.topk_labels,
        "invalid_labels": qwen_pred.invalid_labels,
        "parse_error": qwen_pred.parse_error,
        "inference_s": qwen_pred.inference_s,
        "decode_s": qwen_pred.decode_s,
        "prompt_labels": qwen_pred.prompt_labels,
        "contact_sheet_path": qwen_pred.contact_sheet_path,
    }


def _write_prepare_summary(
    out_dir: Path,
    args: argparse.Namespace,
    labels: list[str],
    clips: list[ClipItem],
    prepared: list[dict[str, Any]],
) -> None:
    _write_json(
        out_dir / "prepare_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "qwen-vl-prepare-only",
            "args": vars(args),
            "n_labels": len(labels),
            "labels": labels,
            "n_clips": len(clips),
            "prepared_contact_sheets": prepared,
        },
    )
    (out_dir / "summary.md").write_text(
        f"# Qwen-VL Prepare Only\n\nPrepared {len(prepared)} contact sheets for {len(clips)} clips.\n",
        encoding="utf-8",
    )


def _write_run_meta(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    dictionary_id: str,
    labels: list[str],
    prompt_labels: list[str],
    clips: list[ClipItem],
    records: list[PredictionRecord],
    skipped: list[dict[str, Any]],
    model_pack: dict[str, Any],
    elapsed_s: float,
) -> None:
    invalid_count = sum(1 for row in records if row.topk_labels and row.topk_labels[0] not in labels)
    _write_json(
        out_dir / "run_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "qwen-vl-constrained-classification",
            "model_id": args.model_id,
            "dictionary_id": dictionary_id,
            "prompt_id": PROMPT_ID,
            "args": vars(args),
            "n_labels": len(labels),
            "labels": labels,
            "prompt_labels": prompt_labels,
            "n_requested_clips": len(clips),
            "n_records": len(records),
            "n_skipped": len(skipped),
            "invalid_top1_count": invalid_count,
            "gpu": model_pack["gpu"],
            "elapsed_s": elapsed_s,
        },
    )


def _summary_markdown(
    *,
    model_id: str,
    dictionary_id: str,
    labels: list[str],
    records: list[PredictionRecord],
    skipped: list[dict[str, Any]],
    report: dict[str, Any],
    random_report: dict[str, Any],
    random_summary: dict[str, Any],
    qwen_rows: list[QwenPrediction],
    elapsed_s: float,
    gpu: dict[str, Any],
) -> str:
    label_text = ", ".join(f"`{label}`" for label in labels)
    invalid_json = sum(1 for row in qwen_rows if row.parse_error)
    invalid_labels = sum(1 for row in qwen_rows if row.invalid_labels)
    return f"""# Qwen3-VL Constrained Classification

Model: `{model_id}`

Dictionary: `{dictionary_id}`

Labels: {label_text}

GPU: cuda_available={gpu['cuda_available']}, model_parameter_device=`{gpu.get('model_parameter_device', 'n/a')}`, gpu_inference_verified={gpu.get('gpu_inference_verified')}.

JSON parse errors: {invalid_json}; responses with invalid labels: {invalid_labels}; skipped clips: {len(skipped)}.

| run | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL constrained | {report['n_clips']} | {len(labels)} | {report['action_top1']:.4f} | {report['action_top3']:.4f} | {report['action_top5']:.4f} | {report['action_macro_f1']:.4f} |
| random seed {random_summary['seed']} | {random_report['n_clips']} | {len(labels)} | {random_report['action_top1']:.4f} | {random_report['action_top3']:.4f} | {random_report['action_top5']:.4f} | {random_report['action_macro_f1']:.4f} |

Expected random Top-1/Top-3/Top-5: {random_summary['expected_top1']:.4f}/{random_summary['expected_top3']:.4f}/{random_summary['expected_top5']:.4f}.
Random macro-F1 Monte Carlo mean: {random_summary['macro_f1_trials']['mean']:.4f}.

Elapsed wall time: {elapsed_s:.1f}s.
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _clip_id_from_row(row: dict[str, Any]) -> str:
    return f"{row.get('video_id', '')}:{row.get('start_sec', '')}:{row.get('end_sec', '')}:{row.get('gold_action', '')}"


if __name__ == "__main__":
    raise SystemExit(main())
