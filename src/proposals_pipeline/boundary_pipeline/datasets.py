"""Local video manifest and ground-truth boundaries.

Assembly101: one egocentric view per recording (first HMC file in sorted
order), coarse boundaries (shared edges between consecutive coarse
segments, per phase) and fine-grained boundaries (starts and ends of every
fine-grained action of that view). EPIC-Kitchens: narration starts and
stops. Evaluation is restricted to the annotated spans of each video.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ANNOTATION_FPS = 30.0
MERGE_WINDOW_S = 0.2

BENCHMARKS = {
    "a101-coarse": ("assembly101", "coarse"),
    "a101-fine": ("assembly101", "fine"),
    "epic": ("epic", "narration"),
}


@dataclass
class GroundTruth:
    boundaries_s: np.ndarray
    spans: list[tuple[float, float]]

    def in_spans(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        keep = np.zeros(len(times), dtype=bool)
        for lo, hi in self.spans:
            keep |= (times >= lo) & (times <= hi)
        return keep


@dataclass
class VideoItem:
    video_id: str
    dataset: str
    video_path: Path
    fps: float
    duration_s: float
    gt: dict[str, GroundTruth] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.video_id.replace("/", "_")


def merge_close(times: list[float] | np.ndarray, window_s: float = MERGE_WINDOW_S) -> np.ndarray:
    """Replace runs of boundaries closer than `window_s` by their mean."""
    times = np.sort(np.asarray(times, dtype=float))
    if len(times) == 0:
        return times
    groups, current = [], [times[0]]
    for t in times[1:]:
        if t - current[-1] <= window_s:
            current.append(t)
        else:
            groups.append(np.mean(current))
            current = [t]
    groups.append(np.mean(current))
    return np.asarray(groups)


def _video_meta(path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    if not fps:
        raise IOError(f"invalid fps for {path}")
    return fps, n / fps


def _coarse_gt(annotations_root: Path, recording: str) -> GroundTruth | None:
    boundaries, spans = [], []
    for phase in ("assembly", "disassembly"):
        path = annotations_root / "coarse-annotations" / "coarse_labels" / f"{phase}_{recording}.txt"
        if not path.is_file():
            continue
        segs = sorted(
            (int(a) / ANNOTATION_FPS, int(b) / ANNOTATION_FPS)
            for a, b, *_ in (line.split(None, 2) for line in path.read_text().splitlines() if line.strip())
        )
        if not segs:
            continue
        edges = [s for s, _ in segs[1:]] + [e for _, e in segs[:-1]]
        boundaries.extend(merge_close(edges))
        spans.append((segs[0][0], segs[-1][1]))
    if not spans:
        return None
    return GroundTruth(np.sort(np.asarray(boundaries)), spans)


def _fine_gt(annotations_root: Path, video_relpath: str) -> GroundTruth | None:
    edges = []
    for split in ("train", "validation", "test"):
        path = annotations_root / "fine-grained-annotations" / f"{split}.csv"
        if not path.is_file():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["video"] == video_relpath:
                    edges.append(int(row["start_frame"]) / ANNOTATION_FPS)
                    edges.append(int(row["end_frame"]) / ANNOTATION_FPS)
    if not edges:
        return None
    return GroundTruth(merge_close(edges), [(min(edges), max(edges))])


def _narration_gt(csv_path: Path, duration_s: float) -> GroundTruth:
    def parse(ts: str) -> float:
        h, m, s = ts.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    edges = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            edges.append(parse(row["start_timestamp"]))
            edges.append(parse(row["stop_timestamp"]))
    return GroundTruth(merge_close(edges), [(0.0, duration_s)])


def _short_recording_id(recording: str) -> str:
    m = re.search(r"_(\d{4}-[a-z]\d{2}[a-z]?)_", recording)
    return m.group(1) if m else recording


def discover(assembly_root: str | Path = "data/assembly101", epic_root: str | Path = "data/epic_kitchens") -> list[VideoItem]:
    items: list[VideoItem] = []
    assembly_root, epic_root = Path(assembly_root), Path(epic_root)

    for rec_dir in sorted(assembly_root.glob("recordings/*/")):
        views = sorted(rec_dir.glob("HMC_*mono10bit.mp4"))
        if not views:
            continue
        video = views[0]
        fps, duration = _video_meta(video)
        item = VideoItem(f"a101/{_short_recording_id(rec_dir.name)}", "assembly101", video, fps, duration)
        ann = assembly_root / "annotations"
        coarse = _coarse_gt(ann, rec_dir.name)
        fine = _fine_gt(ann, f"{rec_dir.name}/{video.name}")
        if coarse is not None:
            item.gt["coarse"] = coarse
        if fine is not None:
            item.gt["fine"] = fine
        items.append(item)

    for video in sorted(epic_root.glob("EPIC-KITCHENS/*/videos/*.MP4")):
        csv_path = epic_root / "annotations" / f"{video.stem}_train_annotations.csv"
        if not csv_path.is_file():
            continue
        fps, duration = _video_meta(video)
        item = VideoItem(f"epic/{video.stem}", "epic", video, fps, duration)
        item.gt["narration"] = _narration_gt(csv_path, duration)
        items.append(item)
    return items


def benchmark_items(items: list[VideoItem], benchmark: str) -> list[tuple[VideoItem, GroundTruth]]:
    dataset, gt_key = BENCHMARKS[benchmark]
    return [(it, it.gt[gt_key]) for it in items if it.dataset == dataset and gt_key in it.gt]


def find_item(items: list[VideoItem], video_id: str) -> VideoItem:
    for it in items:
        if it.video_id == video_id or it.slug == video_id:
            return it
    raise KeyError(f"unknown video id {video_id!r}; known: {[it.video_id for it in items]}")
