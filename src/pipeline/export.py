from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pipeline.types import ActionSegment

EXPORT_FIELDS = [
    "id",
    "start_ms",
    "end_ms",
    "action",
    "object",
    "keyframe_ms",
    "confidence",
    "model_version",
]


def segments_to_json(segments: Iterable[ActionSegment | dict[str, Any]], *, indent: int = 2) -> str:
    rows = [_as_export_dict(item) for item in segments]
    return json.dumps(rows, indent=indent, ensure_ascii=False)


def segments_to_csv(segments: Iterable[ActionSegment | dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for item in segments:
        row = _as_export_dict(item)
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in EXPORT_FIELDS})
    return buffer.getvalue()


def write_actions_json(path: Path, segments: Iterable[ActionSegment | dict[str, Any]]) -> Path:
    path = Path(path)
    path.write_text(segments_to_json(segments), encoding="utf-8")
    return path


def write_actions_csv(path: Path, segments: Iterable[ActionSegment | dict[str, Any]]) -> Path:
    path = Path(path)
    path.write_text(segments_to_csv(segments), encoding="utf-8")
    return path


def load_actions_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("segments") or payload.get("actions") or []
    if not isinstance(payload, list):
        raise ValueError(f"actions.json must be a list, got {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def _as_export_dict(item: ActionSegment | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, ActionSegment):
        return item.to_export_dict()
    return {key: item.get(key) for key in EXPORT_FIELDS}
