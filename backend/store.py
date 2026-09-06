from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.schemas import ActionSegment, ClipInfo, JobStatus, VideoAnnotation, VideoRecord, segment_from_dict
from pipeline.export import load_actions_json, write_actions_csv, write_actions_json
from pipeline.types import Rules

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(os.environ.get("PIPELINE_RUNTIME_DIR", ROOT / "data" / "runtime"))


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or RUNTIME_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create_job(self, filename: str, rules: Rules) -> dict[str, Any]:
        video_id = uuid.uuid4().hex[:12]
        job_id = uuid.uuid4().hex[:12]
        work = self.job_dir(job_id)
        work.mkdir(parents=True, exist_ok=True)
        (work / "clips").mkdir(exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "job_id": job_id,
            "video_id": video_id,
            "name": Path(filename).name,
            "status": "queued",
            "progress": 0,
            "message": "queued",
            "error": None,
            "uploaded_at": now,
            "duration_ms": 0,
            "fps": 0.0,
            "encoder_id": None,
            "segmenter": None,
            "warnings": [],
            "rules": rules.to_dict(),
        }
        self._write_job(record)
        return record

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def source_path(self, job_id: str, name: str | None = None) -> Path:
        suffix = Path(name or "source.mp4").suffix or ".mp4"
        return self.job_dir(job_id) / f"source{suffix}"

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        path = self.job_dir(job_id) / "job.json"
        with self._lock:
            return self._load_job(path)

    def get_by_video_id(self, video_id: str) -> dict[str, Any] | None:
        with self._lock:
            for path in self.root.glob("*/job.json"):
                record = self._load_job(path)
                if record and record.get("video_id") == video_id:
                    return record
        return None

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = self.get_job(job_id)
            if record is None:
                raise KeyError(job_id)
            record.update(fields)
            self._write_job(record)
            return record

    def annotation(self, job_id: str) -> VideoAnnotation:
        record = self.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        actions_path = self.job_dir(job_id) / "actions.json"
        segments = []
        if actions_path.is_file():
            segments = [segment_from_dict(row) for row in load_actions_json(actions_path)]
        return VideoAnnotation(
            video_id=record["video_id"],
            duration_ms=int(record.get("duration_ms") or 0),
            fps=float(record.get("fps") or 0.0),
            segments=segments,
        )

    def save_annotation(self, job_id: str, segments: list[ActionSegment]) -> VideoAnnotation:
        record = self.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        rows = [seg.model_dump() for seg in segments]
        write_actions_json(self.job_dir(job_id) / "actions.json", rows)
        write_actions_csv(self.job_dir(job_id) / "actions.csv", rows)
        return self.annotation(job_id)

    def clip_infos(self, job_id: str, video_id: str) -> list[ClipInfo]:
        clips_path = self.job_dir(job_id) / "clips.json"
        if not clips_path.is_file():
            return []
        payload = json.loads(clips_path.read_text(encoding="utf-8"))
        rows = payload.get("clips") or []
        infos: list[ClipInfo] = []
        for row in rows:
            clip_id = str(row.get("clip_id") or "")
            if not clip_id:
                continue
            infos.append(
                ClipInfo(
                    clip_id=clip_id,
                    start_sec=float(row.get("start_sec") or 0.0),
                    end_sec=float(row.get("end_sec") or 0.0),
                    url=f"/api/videos/{video_id}/clips/{clip_id}",
                )
            )
        return infos

    def clip_file(self, job_id: str, clip_id: str) -> Path | None:
        work = self.job_dir(job_id)
        candidate = work / "clips" / f"{clip_id}.mp4"
        if candidate.is_file():
            return candidate
        clips_path = work / "clips.json"
        if not clips_path.is_file():
            return None
        payload = json.loads(clips_path.read_text(encoding="utf-8"))
        for row in payload.get("clips") or []:
            if str(row.get("clip_id")) != clip_id:
                continue
            rel = row.get("clip_path")
            if rel:
                path = work / rel
                if path.is_file():
                    return path
        return None

    def to_job_status(self, record: dict[str, Any]) -> JobStatus:
        return JobStatus(
            job_id=record["job_id"],
            video_id=record["video_id"],
            status=record["status"],
            progress=int(record.get("progress") or 0),
            message=str(record.get("message") or ""),
            error=record.get("error"),
            encoder_id=record.get("encoder_id"),
            segmenter=record.get("segmenter"),
        )

    def to_video_record(self, record: dict[str, Any]) -> VideoRecord:
        video_id = record["video_id"]
        job_id = record["job_id"]
        return VideoRecord(
            video_id=video_id,
            job_id=job_id,
            name=record["name"],
            duration_ms=int(record.get("duration_ms") or 0),
            status=record["status"],
            uploaded_at=record["uploaded_at"],
            file_url=f"/api/videos/{video_id}/file",
            annotation_url=f"/api/videos/{video_id}/annotation",
            clips=self.clip_infos(job_id, video_id),
        )

    def _write_job(self, record: dict[str, Any]) -> None:
        path = self.job_dir(record["job_id"]) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _load_job(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return None
        return json.loads(raw)
