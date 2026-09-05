from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from action_ranker.data_layout import DEFAULT_PATHS
from action_ranker.types import FrozenSliceList, GoldInterval, SkipEvent, SourceVideo

N_PER_DATASET = 5

ENV = {
    "epic_kitchens": {
        "video_dir": "ACTION_RANKER_EPIC_VIDEO_DIR",
        "val_list": "ACTION_RANKER_EPIC_VAL_LIST",
        "labels": "ACTION_RANKER_EPIC_LABELS",
    },
    "assembly101": {
        "video_dir": "ACTION_RANKER_A101_VIDEO_DIR",
        "val_list": "ACTION_RANKER_A101_VAL_LIST",
        "labels": "ACTION_RANKER_A101_LABELS",
    },
}


class DataAvailabilityError(Exception):
    pass


class FrozenSliceError(Exception):
    pass


def build_slice(n_per_dataset: int = N_PER_DATASET) -> tuple[FrozenSliceList, list[SourceVideo], list[SkipEvent]]:
    skips: list[SkipEvent] = []
    videos: list[SourceVideo] = []
    epic_ids = _select_dataset("epic_kitchens", n_per_dataset, skips, videos)
    a101_ids = _select_dataset("assembly101", n_per_dataset, skips, videos)
    frozen = FrozenSliceList(
        epic_kitchens_ids=epic_ids,
        assembly101_ids=a101_ids,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return frozen, videos, skips


def load_frozen_slice(path: Path, allow_rebuild: bool = False) -> tuple[FrozenSliceList, list[SourceVideo]]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = FrozenSliceList(
        epic_kitchens_ids=list(payload["epic_kitchens_ids"]),
        assembly101_ids=list(payload["assembly101_ids"]),
        created_at=str(payload["created_at"]),
    )
    videos: list[SourceVideo] = []
    for dataset, ids in (
        ("epic_kitchens", frozen.epic_kitchens_ids),
        ("assembly101", frozen.assembly101_ids),
    ):
        for video_id in ids:
            source = _source_for_id(dataset, video_id)
            if source is None:
                if allow_rebuild:
                    raise FrozenSliceError("rebuild requested but selection would change")
                raise DataAvailabilityError(
                    f"Frozen video {dataset}:{video_id} is missing media or labels. "
                    "Do not silently substitute another ID."
                )
            videos.append(source)
    return frozen, videos


def load_intervals(labels_path: str | Path, video_id: str) -> tuple[list[GoldInterval], list[SkipEvent]]:
    path = Path(labels_path)
    rows: list[GoldInterval] = []
    skips: list[SkipEvent] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = {name.lower(): name for name in (reader.fieldnames or [])}
        vid_col = _pick(fields, "video_id", "video", "id")
        start_col = _pick(fields, "start_sec", "start_timestamp", "start")
        end_col = _pick(fields, "end_sec", "stop_timestamp", "end_timestamp", "end")
        action_col = _pick(fields, "gold_action", "prompt_action_label", "action_cls", "narration")
        verb_col = fields.get("gold_verb_id") or fields.get("verb_id") or fields.get("verb_class")
        noun_col = fields.get("gold_noun_id") or fields.get("noun_id") or fields.get("noun_class")
        split_col = fields.get("split")
        if not all([vid_col, start_col, end_col, action_col]):
            raise ValueError(f"Labels CSV {path} needs video, start, end, action columns")
        for row in reader:
            if (row.get(vid_col) or "").strip() != video_id:
                continue
            if split_col:
                split = (row.get(split_col) or "").strip().lower()
                if split and split not in {"val", "validation", "valid"}:
                    continue
            start = _parse_time(row.get(start_col))
            end = _parse_time(row.get(end_col))
            action = (row.get(action_col) or "").strip()
            if start is None or end is None:
                skips.append(
                    SkipEvent(
                        reason="unparseable_interval",
                        video_id=video_id,
                        extra={"start_raw": row.get(start_col), "end_raw": row.get(end_col)},
                    )
                )
                continue
            if start != start or end != end or end <= start:
                skips.append(
                    SkipEvent(
                        reason="invalid_interval",
                        video_id=video_id,
                        start_sec=start,
                        end_sec=end,
                    )
                )
                continue
            if not action:
                skips.append(
                    SkipEvent(
                        reason="empty_action",
                        video_id=video_id,
                        start_sec=start,
                        end_sec=end,
                    )
                )
                continue
            verb = _opt_int(row.get(verb_col) if verb_col else None)
            noun = _opt_int(row.get(noun_col) if noun_col else None)
            rows.append(
                GoldInterval(
                    video_id=video_id,
                    start_sec=start,
                    end_sec=end,
                    gold_action=action,
                    gold_verb_id=verb,
                    gold_noun_id=noun,
                )
            )
    return rows, skips


def _parse_time(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return None



def resolve_path(dataset: str, key: str) -> Path:
    env_name = ENV[dataset][key]
    raw = os.environ.get(env_name, "").strip()
    if raw:
        return Path(raw)
    return Path(DEFAULT_PATHS[dataset][key])


def _select_dataset(
    dataset: str,
    n: int,
    skips: list[SkipEvent],
    videos: list[SourceVideo],
) -> list[str]:
    keys = ENV[dataset]
    val_list = resolve_path(dataset, "val_list")
    if not val_list.is_file():
        raise DataAvailabilityError(
            f"Need {n} complete {dataset} validation videos; missing val list {val_list}. "
            f"Run `python -m action_ranker download-slice` or set {keys['val_list']}."
        )
    ids = sorted({line.strip() for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()})
    chosen: list[str] = []
    for video_id in ids:
        source = _source_for_id(dataset, video_id)
        if source is None:
            skips.append(SkipEvent(reason="incomplete_val_id", video_id=video_id, extra={"dataset": dataset}))
            continue
        chosen.append(video_id)
        videos.append(source)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise DataAvailabilityError(
            f"Need {n} complete {dataset} validation videos; found {len(chosen)}"
        )
    return chosen


def _source_for_id(dataset: str, video_id: str) -> SourceVideo | None:
    video_dir = resolve_path(dataset, "video_dir")
    labels_path = resolve_path(dataset, "labels")
    if not video_dir.is_dir() or not labels_path.is_file():
        return None
    media = _find_media(video_dir, video_id)
    if not media:
        return None
    return SourceVideo(
        dataset=dataset,  # type: ignore[arg-type]
        video_id=video_id,
        split="validation",
        media_path=str(media),
        labels_path=str(labels_path),
    )


def _find_media(video_dir: Path, video_id: str) -> Path | None:
    candidates = [
        video_dir / video_id,
        video_dir / f"{video_id}.mp4",
        video_dir / f"{video_id}.MP4",
        video_dir / f"{video_id}.webm",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = list(video_dir.rglob(f"{video_id}.*"))
    for path in matches:
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}:
            return path
    return None


def _pick(fields: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in fields:
            return fields[name]
    return None


def _opt_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
