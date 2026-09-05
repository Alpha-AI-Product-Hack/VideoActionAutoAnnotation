from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from action_ranker.taxonomies import REPO_ROOT

RAW_A101 = REPO_ROOT / "data" / "raw" / "assembly101"
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"
FPS = 30.0

DEFAULT_FINE_VERBS = [
    "pick up",
    "put down",
    "inspect",
    "unscrew",
    "rotate",
    "position",
    "screw",
    "remove",
    "push",
    "pull",
]
DEFAULT_COARSE_VERBS = [
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
FINE_SPLITS = ["train.csv", "validation.csv", "test.csv"]


@dataclass(frozen=True)
class VerbInspectionClip:
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
    media = _local_rgb_media()
    if not media:
        raise SystemExit("No local Assembly101 RGB media under data/raw/assembly101/_hf/recordings")

    built: dict[str, dict[str, Any]] = {}
    if args.kind in {"fine", "both"}:
        built["fine"] = _build_kind(
            kind="fine",
            run_id=args.fine_run_id,
            verbs=args.fine_verb or DEFAULT_FINE_VERBS,
            candidates=_load_fine_candidates(media, args.fine_verb or DEFAULT_FINE_VERBS),
            clips_per_verb=args.clips_per_verb,
            preview_fps=args.preview_fps,
            preview_max_width=args.preview_max_width,
            overwrite_cuts=args.overwrite_cuts,
        )
    if args.kind in {"coarse", "both"}:
        built["coarse"] = _build_kind(
            kind="coarse",
            run_id=args.coarse_run_id,
            verbs=args.coarse_verb or DEFAULT_COARSE_VERBS,
            candidates=_load_coarse_candidates(media, args.coarse_verb or DEFAULT_COARSE_VERBS),
            clips_per_verb=args.clips_per_verb,
            preview_fps=args.preview_fps,
            preview_max_width=args.preview_max_width,
            overwrite_cuts=args.overwrite_cuts,
        )
    print(json.dumps(built, indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fine/coarse Assembly101 official verb_cls inspection clips")
    parser.add_argument("--kind", choices=["fine", "coarse", "both"], default="both")
    parser.add_argument("--fine-run-id", default="assembly101-fine-official-verb20-inspection")
    parser.add_argument("--coarse-run-id", default="assembly101-coarse-official-verb20-inspection")
    parser.add_argument("--fine-verb", action="append", help="Official fine verb_cls; repeat to override defaults")
    parser.add_argument("--coarse-verb", action="append", help="Official coarse verb_cls; repeat to override defaults")
    parser.add_argument("--clips-per-verb", type=int, default=2)
    parser.add_argument("--preview-fps", type=float, default=12.0)
    parser.add_argument("--preview-max-width", type=int, default=720)
    parser.add_argument("--overwrite-cuts", action="store_true")
    return parser.parse_args(argv)


def _build_kind(
    *,
    kind: str,
    run_id: str,
    verbs: list[str],
    candidates: list[VerbInspectionClip],
    clips_per_verb: int,
    preview_fps: float,
    preview_max_width: int,
    overwrite_cuts: bool,
) -> dict[str, Any]:
    if len(set(verbs)) != len(verbs):
        raise SystemExit(f"Duplicate {kind} verbs: {[v for v, c in Counter(verbs).items() if c > 1]}")
    selected = _select_balanced(candidates, verbs=verbs, per_verb=clips_per_verb)
    run_dir = RUNS_ROOT / run_id
    cut_dir = run_dir / "cut_clips"
    cut_dir.mkdir(parents=True, exist_ok=True)
    selected = _write_cut_clips(
        selected,
        kind=kind,
        cut_dir=cut_dir,
        fps=preview_fps,
        max_width=preview_max_width,
        overwrite=overwrite_cuts,
    )
    _write_artifacts(run_dir=run_dir, kind=kind, verbs=verbs, candidates=candidates, selected=selected)
    return {
        "run_dir": str(run_dir),
        "kind": kind,
        "verbs": verbs,
        "clips": len(selected),
        "candidate_counts": dict(sorted(Counter(c.gold_action for c in candidates).items())),
        "selected_counts": dict(sorted(Counter(c.gold_action for c in selected).items())),
    }


def _local_rgb_media() -> dict[str, Path]:
    root = RAW_A101 / "_hf" / "recordings"
    media: dict[str, Path] = {}
    for path in sorted(root.glob("*/*_rgb.mp4")):
        media[f"{path.parent.name}/{path.name}"] = path
    return media


def _load_fine_candidates(media: dict[str, Path], verbs: list[str]) -> list[VerbInspectionClip]:
    wanted = set(verbs)
    clips: list[VerbInspectionClip] = []
    fine_dir = RAW_A101 / "annotations" / "fine-grained-annotations"
    for split in FINE_SPLITS:
        path = fine_dir / split
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                video_id = (row.get("video") or "").strip()
                verb = (row.get("verb_cls") or "").strip()
                if video_id not in media or verb not in wanted:
                    continue
                if "/" not in video_id:
                    continue
                try:
                    start_sec = int(row["start_frame"]) / FPS
                    end_sec = int(row["end_frame"]) / FPS
                except (KeyError, ValueError):
                    continue
                if end_sec <= start_sec:
                    continue
                recording, view = video_id.split("/", 1)
                action = (row.get("action_cls") or verb).strip()
                clips.append(
                    VerbInspectionClip(
                        clip_id=(
                            f"assembly101_fine_official_verb_rgb:{split}:{row.get('id', len(clips))}:"
                            f"{recording}:{view}:{start_sec:.3f}:{end_sec:.3f}"
                        ),
                        dataset="assembly101",
                        video_id=video_id,
                        media_path=str(media[video_id]),
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                        gold_action=verb,
                        full_gold_action=action,
                        source_label_file=f"fine-grained-annotations/{split}",
                        phase=split.removesuffix(".csv"),
                        recording=recording,
                        view=view,
                        segment_index=int(row.get("id") or len(clips)),
                    )
                )
    return clips


def _load_coarse_candidates(media: dict[str, Path], verbs: list[str]) -> list[VerbInspectionClip]:
    wanted = set(verbs)
    action_to_verb = _load_coarse_action_to_verb()
    views_by_recording: dict[str, dict[str, Path]] = defaultdict(dict)
    for video_id, path in media.items():
        recording, view = video_id.split("/", 1)
        views_by_recording[recording][view] = path

    clips: list[VerbInspectionClip] = []
    label_dir = RAW_A101 / "annotations" / "coarse-annotations" / "coarse_labels"
    for recording, views in sorted(views_by_recording.items()):
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
                for view, path in sorted(views.items()):
                    clips.append(
                        VerbInspectionClip(
                            clip_id=(
                                f"assembly101_coarse_official_verb_rgb:{phase}:{recording}:"
                                f"{view}:{row_index}:{start_sec:.3f}:{end_sec:.3f}"
                            ),
                            dataset="assembly101",
                            video_id=f"{recording}/{view}",
                            media_path=str(path),
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


def _load_coarse_action_to_verb() -> dict[str, str]:
    path = RAW_A101 / "annotations" / "coarse-annotations" / "actions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["action_cls"].strip(): row["verb_cls"].strip() for row in csv.DictReader(handle)}


def _select_balanced(candidates: list[VerbInspectionClip], *, verbs: list[str], per_verb: int) -> list[VerbInspectionClip]:
    by_verb: dict[str, list[VerbInspectionClip]] = defaultdict(list)
    for clip in candidates:
        by_verb[clip.gold_action].append(clip)
    selected: list[VerbInspectionClip] = []
    for verb in verbs:
        unique = _dedupe_segments(by_verb.get(verb, []))
        if len(unique) < per_verb:
            raise SystemExit(f"Need {per_verb} local RGB clips for official verb_cls {verb!r}, found {len(unique)}")
        selected.extend(_uniform_take(unique, per_verb))
    return selected


def _dedupe_segments(clips: list[VerbInspectionClip]) -> list[VerbInspectionClip]:
    best_by_segment: dict[tuple[Any, ...], VerbInspectionClip] = {}
    for clip in sorted(clips, key=lambda item: (_view_rank(item.view), item.view)):
        key = (
            clip.source_label_file,
            clip.recording,
            clip.phase,
            f"{clip.start_sec:.3f}",
            f"{clip.end_sec:.3f}",
            clip.full_gold_action,
        )
        best_by_segment.setdefault(key, clip)
    return sorted(
        best_by_segment.values(),
        key=lambda item: (item.source_label_file, item.recording, item.phase, item.start_sec, item.end_sec, item.view),
    )


def _view_rank(view: str) -> int:
    if view == "C10095_rgb.mp4":
        return 0
    return 1


def _uniform_take(items: list[VerbInspectionClip], n: int) -> list[VerbInspectionClip]:
    if n >= len(items):
        return list(items)
    indexes = np.linspace(0, len(items) - 1, n, dtype=np.int32)
    return [items[int(index)] for index in indexes]


def _write_cut_clips(
    clips: list[VerbInspectionClip],
    *,
    kind: str,
    cut_dir: Path,
    fps: float,
    max_width: int,
    overwrite: bool,
) -> list[VerbInspectionClip]:
    out: list[VerbInspectionClip] = []
    for index, clip in enumerate(clips, start=1):
        filename = f"{index:02d}_{kind}_{_slug(clip.gold_action)}__{_slug(clip.full_gold_action)}__{_slug(clip.view)}.mp4"
        out_path = cut_dir / filename
        if overwrite or not out_path.is_file():
            _cut_preview_clip(clip, out_path=out_path, fps=fps, max_width=max_width)
        data = clip.to_json()
        data["cut_path"] = str(out_path)
        out.append(VerbInspectionClip(**data))
    return out


def _cut_preview_clip(clip: VerbInspectionClip, *, out_path: Path, fps: float, max_width: int) -> None:
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


def _draw_overlay(image: Image.Image, clip: VerbInspectionClip) -> None:
    draw = ImageDraw.Draw(image)
    lines = [
        f"official verb_cls: {clip.gold_action} | action: {clip.full_gold_action}",
        f"source: {clip.phase} | {clip.view} | {clip.start_sec:.2f}-{clip.end_sec:.2f}s",
    ]
    draw.rectangle((0, 0, image.width, 44), fill=(0, 0, 0))
    draw.text((8, 6), lines[0][:130], fill=(255, 255, 255))
    draw.text((8, 24), lines[1][:130], fill=(255, 255, 255))


def _write_artifacts(
    *,
    run_dir: Path,
    kind: str,
    verbs: list[str],
    candidates: list[VerbInspectionClip],
    selected: list[VerbInspectionClip],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "verb_bank.csv").write_text("label\n" + "\n".join(verbs) + "\n", encoding="utf-8")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": f"assembly101_{kind}_official_verb_rgb_inspection",
        "annotation_kind": kind,
        "label_field": "verb_cls",
        "n_clips": len(selected),
        "verbs": verbs,
        "allocation": dict(sorted(Counter(clip.gold_action for clip in selected).items())),
        "clips": [clip.to_json() for clip in selected],
    }
    (run_dir / "clips.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_json(
        run_dir / "selection_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "assembly101-official-verb-inspection",
            "annotation_kind": kind,
            "label_field": "verb_cls",
            "verbs": verbs,
            "n_clips": len(selected),
            "candidate_counts": dict(sorted(Counter(clip.gold_action for clip in candidates).items())),
            "selected_counts": dict(sorted(Counter(clip.gold_action for clip in selected).items())),
        },
    )
    rows = [
        {
            "index": index,
            "cut_path": clip.cut_path,
            "gold_verb": clip.gold_action,
            "full_action": clip.full_gold_action,
            "annotation_kind": kind,
            "source_label_file": clip.source_label_file,
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
    (run_dir / "summary.md").write_text(_summary_markdown(kind, selected, verbs), encoding="utf-8")


def _summary_markdown(kind: str, clips: list[VerbInspectionClip], verbs: list[str]) -> str:
    rows = [
        f"| {index} | `{clip.gold_action}` | `{clip.full_gold_action}` | `{clip.phase}` | [{Path(clip.cut_path or '').name}]({clip.cut_path}) |"
        for index, clip in enumerate(clips, start=1)
    ]
    return f"""# Assembly101 {kind.title()} Official Verb Inspection Clips

Gold labels use official `{kind}` `verb_cls`.

Selected verbs: {', '.join(f'`{verb}`' for verb in verbs)}

Clips: {len(clips)} total, {len(clips) // max(1, len(verbs))} per verb.

| # | gold verb_cls | full action | source | clip |
|---:|---|---|---|---|
{chr(10).join(rows)}
"""


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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
