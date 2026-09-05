from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from action_ranker.gt_clips import ClipDecodeError, sample_gt_clip
from action_ranker.slice import load_intervals
from action_ranker.taxonomies import load_dictionary_rows


def build_local_epic_clips(
    *,
    labels_csv: str | Path,
    media_root: str | Path,
    out_path: str | Path,
    n_clips: int = 10,
) -> dict:
    root = Path(media_root)
    labels = Path(labels_csv)
    bank_rows = load_dictionary_rows("epic_kitchens_observed")
    bank_ids = {(row.verb_id, row.noun_id) for row in bank_rows if row.verb_id is not None and row.noun_id is not None}
    bank_labels = {row.label for row in bank_rows}
    video_ids = _available_epic_video_ids(root)
    clips: list[dict] = []
    for video_id in video_ids:
        media_path = _find_media(root, video_id)
        if media_path is None:
            continue
        intervals, _ = load_intervals(labels, video_id)
        for interval in intervals:
            in_bank = (interval.gold_verb_id, interval.gold_noun_id) in bank_ids or interval.gold_action in bank_labels
            if not in_bank:
                continue
            if not _can_decode_interval(media_path, interval.start_sec, interval.end_sec):
                continue
            clips.append(
                {
                    "dataset": "epic_kitchens",
                    "video_id": interval.video_id,
                    "start_sec": interval.start_sec,
                    "end_sec": interval.end_sec,
                    "gold_action": interval.gold_action,
                    "gold_verb_id": interval.gold_verb_id,
                    "gold_noun_id": interval.gold_noun_id,
                }
            )
            if len(clips) >= n_clips:
                payload = _payload(clips, labels, root)
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                return payload
    raise ValueError(f"Need {n_clips} EPIC clips with local media; found {len(clips)}")


def _available_epic_video_ids(root: Path) -> list[str]:
    ids: set[str] = set()
    if not root.is_dir():
        return []
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}:
            ids.add(path.stem)
    return sorted(ids)


def _find_media(root: Path, video_id: str) -> Path | None:
    for suffix in (".MP4", ".mp4", ".webm", ".mkv", ".avi"):
        path = root / f"{video_id}{suffix}"
        if path.is_file():
            return path
    return None


def _can_decode_interval(media_path: Path, start_sec: float, end_sec: float) -> bool:
    try:
        return sample_gt_clip(media_path, start_sec, end_sec, num_frames=1) is not None
    except ClipDecodeError:
        return False


def _payload(clips: list[dict], labels_csv: Path, media_root: Path) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "10 EPIC-KITCHENS validation clips from locally available media, gold from EPIC_100_validation intervals.",
        "labels_csv": str(labels_csv),
        "media_root": str(media_root),
        "videos": sorted({clip["video_id"] for clip in clips}),
        "n_clips": len(clips),
        "clips": clips,
    }
