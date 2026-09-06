from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from pipeline.classify import classify_clips, select_encoder_name
from pipeline.clips import build_clip_intervals, materialize_clips
from pipeline.export import write_actions_csv, write_actions_json
from pipeline.media import probe_video
from pipeline.segment import boundaries_to_intervals, propose_boundaries
from pipeline.types import PipelineResult, Rules

ProgressFn = Callable[[int, str], None]


def process_video(
    video_path: Path,
    work_dir: Path,
    rules: Rules | dict | None = None,
    *,
    video_id: str | None = None,
    encoder_name: str | None = None,
    sources: list[str] | None = None,
    config_path: Path | None = None,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    parsed = rules if isinstance(rules, Rules) else Rules.from_dict(rules)
    vid = video_id or video_path.stem
    progress and progress(10, "probe video")
    fps, duration_s = probe_video(video_path)
    duration_ms = int(round(duration_s * 1000))

    source_list = sources or _sources_from_env()
    progress and progress(25, "segment video")
    times, segmenter, warnings = propose_boundaries(
        video_path,
        work_dir,
        sources=source_list,
        config_path=config_path or _config_from_env(),
        progress=progress,
    )
    min_duration_s = parsed.min_duration_ms / 1000.0
    ranges = boundaries_to_intervals(times, duration_s, min_duration_s=min_duration_s)
    clips = build_clip_intervals(ranges)

    progress and progress(55, "cut clips")
    clips = materialize_clips(video_path, work_dir, clips)

    progress and progress(75, "classify clips")
    segments, encoder_id = classify_clips(
        video_path,
        clips,
        parsed,
        work_dir,
        encoder_name=encoder_name or select_encoder_name(),
    )
    write_actions_json(work_dir / "actions.json", segments)
    write_actions_csv(work_dir / "actions.csv", segments)
    result = PipelineResult(
        video_id=vid,
        duration_ms=duration_ms,
        fps=fps,
        segments=segments,
        encoder_id=encoder_id,
        segmenter=segmenter,
        clips=clips,
        warnings=warnings,
    )
    (work_dir / "run_meta.json").write_text(
        json.dumps({**result.to_meta_dict(), "rules": parsed.to_dict()}, indent=2),
        encoding="utf-8",
    )
    progress and progress(100, "done")
    return result


def _sources_from_env() -> list[str]:
    raw = os.environ.get("PIPELINE_SOURCES", "motion")
    return [part.strip() for part in raw.split(",") if part.strip()] or ["motion"]


def _config_from_env() -> Path | None:
    raw = os.environ.get("PIPELINE_CONFIG")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None
