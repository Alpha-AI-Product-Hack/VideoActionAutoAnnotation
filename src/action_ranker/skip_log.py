from __future__ import annotations

import json
from pathlib import Path

from action_ranker.types import SkipEvent


def write_skip_log(path: Path, events: list[SkipEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for event in events:
        payload = {
            "reason": event.reason,
            "video_id": event.video_id,
            "start_sec": event.start_sec,
            "end_sec": event.end_sec,
            **event.extra,
        }
        lines.append(json.dumps(payload, ensure_ascii=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
