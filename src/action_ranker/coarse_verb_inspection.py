from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from action_ranker.taxonomies import REPO_ROOT

A101_REPO = "cvml-nus/assembly101"
RAW_A101 = REPO_ROOT / "data" / "raw" / "assembly101"
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"
FPS = 30.0

DEFAULT_RECORDINGS = [
    "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724",
    "nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253",
]
DEFAULT_EXTRA_RECORDING = "nusar-2021_action_both_9061-a01_9061_user_id_2021-02-09_134524"
DEFAULT_EXTRA_VIEW = "C10095_rgb.mp4"
DEFAULT_VERBS = [
    "attach",
    "detach",
    "inspect",
    "unscrew",
    "screw",
    "attempt to attach",
    "demonstrate",
    "attempt to detach",
    "position",
    "remove",
]


@dataclass(frozen=True)
class CoarseVerbClip:
    clip_id: str
    dataset: str
    video_id: str
    media_path: str
    start_sec: float
    end_sec: float
    gold_action: str
    full_gold_action: str
    source_label_file: str
    phase: str
    recording: str
    view: str
    segment_index: int
    cut_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = RUNS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cut_dir = run_dir / "cut_clips"
    cut_dir.mkdir(parents=True, exist_ok=True)

    verbs = args.verb or DEFAULT_VERBS
    if len(verbs) != 10:
        raise SystemExit(f"Expected exactly 10 verbs, got {len(verbs)}")
    recordings = args.recording or DEFAULT_RECORDINGS
    extra_recordings = [] if args.no_extra_recording else [args.extra_recording]

    media = _ensure_media(
        recordings=recordings,
        extra_recordings=extra_recordings,
        extra_view=args.extra_view,
        download=not args.no_download,
    )
    action_to_verb = _load_action_to_verb()
    candidates = _load_candidates(action_to_verb=action_to_verb, media=media, verbs=verbs)
    selected = _select_clips(candidates, verbs=verbs, per_verb=args.clips_per_verb)
    if len(selected) != args.clips_per_verb * len(verbs):
        raise SystemExit(f"Expected {args.clips_per_verb * len(verbs)} clips, got {len(selected)}")

    selected_with_cuts = _write_cut_clips(
        selected,
        cut_dir=cut_dir,
        fps=args.preview_fps,
        max_width=args.preview_max_width,
        overwrite=args.overwrite_cuts,
    )
    _write_artifacts(
        run_dir=run_dir,
        verbs=verbs,
        media=media,
        candidates=candidates,
        selected=selected_with_cuts,
        args=args,
    )
    print(run_dir)
    print(
        json.dumps(
            {
                "verbs": verbs,
                "clips": len(selected_with_cuts),
                "cuts_dir": str(cut_dir),
                "candidate_counts": dict(sorted(Counter(c.gold_action for c in candidates).items())),
                "selected_counts": dict(sorted(Counter(c.gold_action for c in selected_with_cuts).items())),
            },
            indent=2,
        )
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 20 RGB Assembly101 coarse verb clips for inspection")
    parser.add_argument("--run-id", default="assembly101-coarse-verb10-vlm-xclip-inspection")
    parser.add_argument("--recording", action="append", help="Main recording id; repeat to override defaults")
    parser.add_argument("--extra-recording", default=DEFAULT_EXTRA_RECORDING)
    parser.add_argument("--extra-view", default=DEFAULT_EXTRA_VIEW)
    parser.add_argument("--no-extra-recording", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--verb", action="append", help="Allowed coarse verb; repeat exactly 10 times")
    parser.add_argument("--clips-per-verb", type=int, default=2)
    parser.add_argument("--preview-fps", type=float, default=12.0)
    parser.add_argument("--preview-max-width", type=int, default=720)
    parser.add_argument("--overwrite-cuts", action="store_true")
    return parser.parse_args(argv)


def _ensure_media(
    *,
    recordings: list[str],
    extra_recordings: list[str],
    extra_view: str,
    download: bool,
) -> dict[str, dict[str, Path]]:
    media: dict[str, dict[str, Path]] = {}
    for recording in recordings:
        local_views = sorted((RAW_A101 / "_hf" / "recordings" / recording).glob("*_rgb.mp4"))
        media[recording] = {path.name: path for path in local_views}
    for recording in extra_recordings:
        path = RAW_A101 / "_hf" / "recordings" / recording / extra_view
        if not path.is_file() and download:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise SystemExit("Install huggingface_hub to download Assembly101 RGB media") from exc
            downloaded = hf_hub_download(
                A101_REPO,
                f"recordings/{recording}/{extra_view}",
                repo_type="dataset",
                local_dir=str(RAW_A101 / "_hf"),
            )
            path = Path(downloaded)
        media.setdefault(recording, {})[path.name] = path
    missing = [f"{recording}/*.rgb.mp4" for recording in recordings if not media.get(recording)]
    missing.extend(
        f"{recording}/{view}"
        for recording, views in media.items()
        for view, path in views.items()
        if not path.is_file()
    )
    if missing:
        raise SystemExit("Missing RGB media: " + ", ".join(missing))
    return media


def _load_action_to_verb() -> dict[str, str]:
    path = RAW_A101 / "annotations" / "coarse-annotations" / "actions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["action_cls"].strip(): row["verb_cls"].strip() for row in csv.DictReader(handle)}


def _load_candidates(
    *,
    action_to_verb: dict[str, str],
    media: dict[str, dict[str, Path]],
    verbs: list[str],
) -> list[CoarseVerbClip]:
    wanted = set(verbs)
    label_dir = RAW_A101 / "annotations" / "coarse-annotations" / "coarse_labels"
    clips: list[CoarseVerbClip] = []
    for recording, views in sorted(media.items()):
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
                action = parts[2].strip()
                verb = action_to_verb.get(action)
                if verb not in wanted:
                    continue
                try:
                    start_sec = int(parts[0]) / FPS
                    end_sec = int(parts[1]) / FPS
                except ValueError:
                    continue
                if end_sec <= start_sec:
                    continue
                for view, media_path in sorted(views.items()):
                    clips.append(
                        CoarseVerbClip(
                            clip_id=f"coarse_verb_rgb:{phase}:{recording}:{view}:{row_index}:{start_sec:.3f}:{end_sec:.3f}",
                            dataset="assembly101",
                            video_id=f"{recording}/{view}",
                            media_path=str(media_path),
                            start_sec=float(start_sec),
                            end_sec=float(end_sec),
                            gold_action=verb,
                            full_gold_action=action,
                            source_label_file=f"coarse-annotations/coarse_labels/{phase}_{recording}.txt",
                            phase=phase,
                            recording=recording,
                            view=view,
                            segment_index=row_index,
                        )
                    )
    return clips


def _select_clips(candidates: list[CoarseVerbClip], *, verbs: list[str], per_verb: int) -> list[CoarseVerbClip]:
    selected: list[CoarseVerbClip] = []
    by_verb: dict[str, list[CoarseVerbClip]] = defaultdict(list)
    for clip in candidates:
        by_verb[clip.gold_action].append(clip)

    for verb in verbs:
        rows = sorted(
            by_verb[verb],
            key=lambda c: (
                _main_recording_rank(c.recording),
                c.recording,
                c.phase,
                c.segment_index,
                c.view,
            ),
        )
        picked: list[CoarseVerbClip] = []
        seen_segments: set[tuple[str, str, int, str]] = set()
        for clip in rows:
            key = (clip.recording, clip.phase, clip.segment_index, clip.full_gold_action)
            if key in seen_segments:
                continue
            seen_segments.add(key)
            picked.append(clip)
            if len(picked) >= per_verb:
                break
        if len(picked) < per_verb:
            for clip in rows:
                if clip in picked:
                    continue
                picked.append(clip)
                if len(picked) >= per_verb:
                    break
        if len(picked) < per_verb:
            raise SystemExit(f"Need {per_verb} clips for verb {verb!r}, found {len(rows)} candidates")
        selected.extend(picked)
    return selected


def _main_recording_rank(recording: str) -> int:
    if recording in DEFAULT_RECORDINGS:
        return 0
    return 1


def _write_cut_clips(
    clips: list[CoarseVerbClip],
    *,
    cut_dir: Path,
    fps: float,
    max_width: int,
    overwrite: bool,
) -> list[CoarseVerbClip]:
    out: list[CoarseVerbClip] = []
    for index, clip in enumerate(clips, start=1):
        filename = f"{index:02d}_{_slug(clip.gold_action)}__{_slug(clip.full_gold_action)}__{_slug(clip.view)}.mp4"
        out_path = cut_dir / filename
        if overwrite or not out_path.is_file():
            _cut_preview_clip(clip, out_path=out_path, fps=fps, max_width=max_width)
        out.append(_replace_cut_path(clip, str(out_path)))
    return out


def _replace_cut_path(clip: CoarseVerbClip, cut_path: str) -> CoarseVerbClip:
    data = clip.to_json()
    data["cut_path"] = cut_path
    return CoarseVerbClip(**data)


def _cut_preview_clip(clip: CoarseVerbClip, *, out_path: Path, fps: float, max_width: int) -> None:
    try:
        import av
    except ImportError as exc:
        raise SystemExit("PyAV is required to cut preview clips") from exc

    container = av.open(clip.media_path)
    output = av.open(str(out_path), mode="w")
    try:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "AUTO"
        width, height = _preview_size(in_stream.width, in_stream.height, max_width=max_width)
        codec = _pick_encoder(av)
        out_stream = output.add_stream(codec, rate=int(round(fps)))
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"

        next_emit = clip.start_sec
        emitted = 0
        for frame in container.decode(in_stream):
            ts = float(frame.time) if frame.time is not None else float(frame.pts * in_stream.time_base)
            if ts < clip.start_sec:
                continue
            if ts > clip.end_sec:
                break
            if ts + 1e-6 < next_emit:
                continue
            image = frame.to_image().convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            _draw_overlay(image, clip)
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


def _draw_overlay(image: Image.Image, clip: CoarseVerbClip) -> None:
    draw = ImageDraw.Draw(image)
    text = f"gold verb: {clip.gold_action} | action: {clip.full_gold_action}"
    box_h = 28
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0))
    draw.text((8, 7), text[:120], fill=(255, 255, 255))


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text[:80] or "item"


def _write_artifacts(
    *,
    run_dir: Path,
    verbs: list[str],
    media: dict[str, dict[str, Path]],
    candidates: list[CoarseVerbClip],
    selected: list[CoarseVerbClip],
    args: argparse.Namespace,
) -> None:
    (run_dir / "verb_bank.csv").write_text("label\n" + "\n".join(verbs) + "\n", encoding="utf-8")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "assembly101_coarse_rgb_verb10_inspection",
        "n_clips": len(selected),
        "verbs": verbs,
        "allocation": dict(sorted(Counter(clip.gold_action for clip in selected).items())),
        "clips": [clip.to_json() for clip in selected],
    }
    (run_dir / "clips.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "coarse-verb10-inspection",
        "args": vars(args),
        "verbs": verbs,
        "n_clips": len(selected),
        "media": {
            recording: {
                view: {"path": str(path), "size_bytes": int(path.stat().st_size)}
                for view, path in sorted(views.items())
            }
            for recording, views in sorted(media.items())
        },
        "candidate_counts": dict(sorted(Counter(clip.gold_action for clip in candidates).items())),
        "selected_counts": dict(sorted(Counter(clip.gold_action for clip in selected).items())),
    }
    (run_dir / "selection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows = [
        {
            "index": index,
            "cut_path": clip.cut_path,
            "gold_verb": clip.gold_action,
            "coarse_action": clip.full_gold_action,
            "recording": clip.recording,
            "view": clip.view,
            "phase": clip.phase,
            "start_sec": f"{clip.start_sec:.3f}",
            "end_sec": f"{clip.end_sec:.3f}",
            "source_media": clip.media_path,
        }
        for index, clip in enumerate(selected, start=1)
    ]
    _write_csv(run_dir / "inspection_table_base.csv", rows)
    (run_dir / "summary.md").write_text(_summary_markdown(selected, verbs), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(clips: list[CoarseVerbClip], verbs: list[str]) -> str:
    rows = [
        f"| {index} | `{clip.gold_action}` | `{clip.full_gold_action}` | `{clip.view}` | [{Path(clip.cut_path or '').name}]({clip.cut_path}) |"
        for index, clip in enumerate(clips, start=1)
    ]
    return f"""# Coarse Verb10 Inspection Clips

Selected verbs: {', '.join(f'`{verb}`' for verb in verbs)}

Clips: {len(clips)} total, 2 per verb.

| # | gold verb | coarse action | view | clip |
|---:|---|---|---|---|
{chr(10).join(rows)}
"""


if __name__ == "__main__":
    raise SystemExit(main())
