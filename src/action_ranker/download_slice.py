from __future__ import annotations

import csv
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from action_ranker.data_layout import (
    DEFAULT_PATHS,
    ENV_EXPORT,
    RAW_A101,
    RAW_EPIC,
    SPLITS_DIR,
    rel_to_repo,
)
from action_ranker.slice import DataAvailabilityError, N_PER_DATASET, _source_for_id

A101_REPO = "cvml-nus/assembly101"
EPIC_VAL_CSV_URL = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/"
    "master/EPIC_100_validation.csv"
)
A101_FPS = 30.0
SHORT_ID_RE = re.compile(r"_(\d{4}-[a-z]\d{2}[a-z]?)_")
DOWNLOADED_SLICE = SPLITS_DIR / "downloaded_slice.json"


def run_download(*, dry_run: bool = False) -> dict:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_A101.mkdir(parents=True, exist_ok=True)
    RAW_EPIC.mkdir(parents=True, exist_ok=True)

    a101_ids, a101_short_to_recording = _assembly_val_catalog(dry_run=dry_run)
    epic_ids, epic_val_rows = _epic_val_catalog(dry_run=dry_run)
    ann_root = RAW_A101 / "annotations"

    _write_lines(DEFAULT_PATHS["assembly101"]["val_list"], a101_ids)
    _write_lines(DEFAULT_PATHS["epic_kitchens"]["val_list"], epic_ids)
    _write_a101_coarse_intervals(ann_root, a101_short_to_recording)
    _write_epic_intervals(epic_val_rows)

    if dry_run:
        a101_plan = a101_ids[:N_PER_DATASET]
        epic_plan = epic_ids[:N_PER_DATASET]
        print(f"[dry-run] Assembly101 first {N_PER_DATASET} val recordings: {a101_plan}")
        print(f"[dry-run] EPIC-KITCHENS first {N_PER_DATASET} val videos: {epic_plan}")
        print("[dry-run] No media downloaded.")
        _print_exports()
        return {"dry_run": True, "assembly101_ids": a101_plan, "epic_kitchens_ids": epic_plan}

    _fetch_assembly_videos(a101_ids)
    _fetch_epic_videos(epic_ids)

    complete_a101 = _first_complete("assembly101", a101_ids)
    complete_epic = _first_complete("epic_kitchens", epic_ids)
    payload = {
        "epic_kitchens_ids": complete_epic,
        "assembly101_ids": complete_a101,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paths": {key: rel_to_repo(path) for key, path in ENV_EXPORT.items()},
    }
    DOWNLOADED_SLICE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"downloaded 5+5 -> {rel_to_repo(DOWNLOADED_SLICE)}")
    print(f"  assembly101: {complete_a101}")
    print(f"  epic_kitchens: {complete_epic}")
    _print_exports()
    return payload


def _print_exports() -> None:
    for key, path in ENV_EXPORT.items():
        print(f"export {key}={rel_to_repo(path)}")


def _write_lines(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def _short_id(recording: str) -> str:
    match = SHORT_ID_RE.search(recording)
    return match.group(1) if match else recording


def _assembly_val_catalog(dry_run: bool) -> tuple[list[str], dict[str, str]]:
    """Val recording ids from fine-grained validation.csv; gold labels come from coarse files."""
    ann_root = _ensure_a101_annotations(dry_run=dry_run)
    csv_path = ann_root / "fine-grained-annotations" / "validation.csv"
    if not csv_path.is_file():
        raise DataAvailabilityError(
            f"Assembly101 validation.csv missing at {csv_path}. "
            "Accept the CC BY-NC license on Hugging Face and run hf auth login."
        )
    short_to_recording: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            video = (raw.get("video") or "").strip()
            if not video:
                continue
            recording = video.split("/")[0]
            short = _short_id(recording)
            short_to_recording.setdefault(short, recording)
    ids = sorted(short_to_recording)
    if not ids:
        raise DataAvailabilityError("Assembly101 validation.csv has no video ids")
    return ids, short_to_recording


def _ensure_a101_annotations(dry_run: bool) -> Path:
    local = RAW_A101 / "annotations"
    csv_path = local / "fine-grained-annotations" / "validation.csv"
    if csv_path.is_file():
        return local
    if dry_run:
        raise DataAvailabilityError(
            "Dry-run needs local Assembly101 annotations at "
            f"{csv_path}. Run without --dry-run after hf auth login."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DataAvailabilityError(
            "Install huggingface_hub: pip install -e '.[hf]'"
        ) from exc
    try:
        snapshot_download(
            A101_REPO,
            repo_type="dataset",
            allow_patterns=["annotations/**"],
            local_dir=str(RAW_A101),
        )
    except Exception as exc:  # noqa: BLE001
        raise DataAvailabilityError(
            f"Could not download {A101_REPO} annotations ({exc}). "
            "Accept the dataset license and run hf auth login."
        ) from exc
    if not csv_path.is_file():
        raise DataAvailabilityError(f"HF snapshot did not include {csv_path}")
    return local


def _fetch_assembly_videos(val_ids: list[str]) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise DataAvailabilityError("Install huggingface_hub: pip install -e '.[hf]'") from exc
    api = HfApi()
    try:
        files = api.list_repo_files(A101_REPO, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        raise DataAvailabilityError(
            f"Cannot list {A101_REPO}: {exc}. hf auth login required."
        ) from exc
    complete = 0
    for short_id in val_ids:
        dest = RAW_A101 / f"{short_id}.mp4"
        if dest.is_file():
            complete += 1
            if complete >= N_PER_DATASET:
                return
            continue
        rec_files = [
            path
            for path in files
            if path.startswith("recordings/")
            and short_id in path
            and path.endswith("mono10bit.mp4")
            and "/HMC_" in path
        ]
        rec_files = sorted(rec_files)
        if not rec_files:
            continue
        try:
            downloaded = hf_hub_download(
                A101_REPO,
                rec_files[0],
                repo_type="dataset",
                local_dir=str(RAW_A101 / "_hf"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[assembly101] skip {short_id}: {exc}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, dest)
        complete += 1
        print(f"[assembly101] {short_id} <- {rec_files[0]}")
        if complete >= N_PER_DATASET:
            return


def _write_a101_coarse_intervals(ann_root: Path, short_to_recording: dict[str, str]) -> None:
    from action_ranker.taxonomies import load_dictionary_rows

    labels_dir = ann_root / "coarse-annotations" / "coarse_labels"
    id_lookup = {row.label: (row.verb_id, row.noun_id) for row in load_dictionary_rows("assembly101_coarse")}
    path = DEFAULT_PATHS["assembly101"]["labels"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "start_sec", "end_sec", "gold_action", "verb_id", "noun_id"],
        )
        writer.writeheader()
        for short_id, recording in sorted(short_to_recording.items()):
            for phase in ("assembly", "disassembly"):
                label_path = labels_dir / f"{phase}_{recording}.txt"
                if not label_path.is_file():
                    continue
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split(None, 2)
                    if len(parts) < 3:
                        continue
                    try:
                        start_sec = int(parts[0]) / A101_FPS
                        end_sec = int(parts[1]) / A101_FPS
                    except ValueError:
                        continue
                    action = parts[2].strip()
                    if not action:
                        continue
                    verb_id, noun_id = id_lookup.get(action, (None, None))
                    writer.writerow(
                        {
                            "video_id": short_id,
                            "start_sec": f"{start_sec:.6f}",
                            "end_sec": f"{end_sec:.6f}",
                            "gold_action": action,
                            "verb_id": "" if verb_id is None else verb_id,
                            "noun_id": "" if noun_id is None else noun_id,
                        }
                    )
                    n_rows += 1
    if n_rows == 0:
        raise DataAvailabilityError(
            f"No coarse Assembly101 intervals under {labels_dir}. "
            "Need annotations/coarse-annotations/coarse_labels/{{assembly,disassembly}}_<recording>.txt"
        )


def _epic_val_catalog(dry_run: bool) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = RAW_EPIC / "EPIC_100_validation.csv"
    if not csv_path.is_file():
        if dry_run:
            raise DataAvailabilityError(
                f"Dry-run needs {csv_path}. Run without --dry-run to fetch official annotations."
            )
        _http_download(EPIC_VAL_CSV_URL, csv_path)
    rows: list[dict[str, str]] = []
    ids: list[str] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            video_id = (raw.get("video_id") or "").strip()
            if not video_id:
                continue
            rows.append(raw)
            if video_id not in seen:
                seen.add(video_id)
                ids.append(video_id)
    ids = sorted(set(ids))
    if not ids:
        raise DataAvailabilityError("EPIC_100_validation.csv has no video_id values")
    return ids, rows


def _write_epic_intervals(rows: list[dict[str, str]]) -> None:
    path = DEFAULT_PATHS["epic_kitchens"]["labels"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "start_timestamp",
                "stop_timestamp",
                "prompt_action_label",
                "verb_id",
                "noun_id",
            ],
        )
        writer.writeheader()
        for raw in rows:
            verb = (raw.get("verb") or "").strip()
            noun = (raw.get("noun") or "").strip().replace(":", " ")
            label = f"{verb} {noun}".strip()
            writer.writerow(
                {
                    "video_id": (raw.get("video_id") or "").strip(),
                    "start_timestamp": raw.get("start_timestamp") or "",
                    "stop_timestamp": raw.get("stop_timestamp") or "",
                    "prompt_action_label": label,
                    "verb_id": raw.get("verb_class") or "",
                    "noun_id": raw.get("noun_class") or "",
                }
            )


def _fetch_epic_videos(val_ids: list[str]) -> None:
    base = os.environ.get("ACTION_RANKER_EPIC_VIDEO_BASE", "").rstrip("/")
    complete = 0
    missing: list[str] = []
    for video_id in val_ids:
        dest_mp4 = RAW_EPIC / f"{video_id}.MP4"
        dest_alt = RAW_EPIC / f"{video_id}.mp4"
        if dest_mp4.is_file() or dest_alt.is_file():
            complete += 1
            if complete >= N_PER_DATASET:
                return
            continue
        if not base:
            missing.append(video_id)
            continue
        url = f"{base}/{video_id}.MP4"
        try:
            _http_download(url, dest_mp4)
        except DataAvailabilityError:
            missing.append(video_id)
            continue
        complete += 1
        print(f"[epic] {video_id}")
        if complete >= N_PER_DATASET:
            return
    if complete < N_PER_DATASET:
        need = val_ids[:20]
        raise DataAvailabilityError(
            f"Need {N_PER_DATASET} EPIC-KITCHENS val RGB videos under {rel_to_repo(RAW_EPIC)} "
            f"(found {complete}). Place files named <video_id>.MP4 for official val ids "
            f"such as {need[:5]}. Optional: set ACTION_RANKER_EPIC_VIDEO_BASE to an HTTP "
            "directory of MP4s. EPIC videos are license-gated; annotations were saved."
        )


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise DataAvailabilityError(f"Download failed {url}: {exc}") from exc
    tmp.replace(dest)


def _first_complete(dataset: str, ids: list[str]) -> list[str]:
    chosen: list[str] = []
    for video_id in ids:
        if _source_for_id(dataset, video_id) is None:
            continue
        chosen.append(video_id)
        if len(chosen) == N_PER_DATASET:
            return chosen
    raise DataAvailabilityError(
        f"Need {N_PER_DATASET} complete {dataset} validation videos; found {len(chosen)}. "
        f"Media dir {rel_to_repo(DEFAULT_PATHS[dataset]['video_dir'])}."
    )
