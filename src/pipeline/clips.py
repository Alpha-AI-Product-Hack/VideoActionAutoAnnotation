from __future__ import annotations

import json
from pathlib import Path

from pipeline.media import extract_clip_mp4
from pipeline.types import ClipInterval


def build_clip_intervals(ranges: list[tuple[float, float]]) -> list[ClipInterval]:
    clips: list[ClipInterval] = []
    for index, (start, end) in enumerate(ranges):
        clips.append(
            ClipInterval(
                clip_id=f"clip_{index:03d}",
                start_sec=float(start),
                end_sec=float(end),
            )
        )
    return clips


def materialize_clips(
    source: Path,
    work_dir: Path,
    clips: list[ClipInterval],
) -> list[ClipInterval]:
    clip_dir = work_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[ClipInterval] = []
    for clip in clips:
        dest = clip_dir / f"{clip.clip_id}.mp4"
        path = extract_clip_mp4(source, dest, clip.start_sec, clip.end_sec)
        materialized.append(
            ClipInterval(
                clip_id=clip.clip_id,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                path=str(path.relative_to(work_dir)) if path is not None else None,
            )
        )
    write_clips_json(work_dir, source, materialized)
    return materialized


def write_clips_json(work_dir: Path, source: Path, clips: list[ClipInterval]) -> Path:
    payload = {
        "source": str(source),
        "clips": [
            {
                "clip_id": clip.clip_id,
                "dataset": "custom",
                "video_id": source.name,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "clip_path": clip.path,
            }
            for clip in clips
        ],
    }
    out = work_dir / "clips.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
