from __future__ import annotations

import threading
import traceback
from pathlib import Path

from backend.store import JobStore
from pipeline.run import process_video
from pipeline.types import Rules

_PIPELINE_LOCK = threading.Lock()


def run_job(store: JobStore, job_id: str, video_path: Path) -> None:
    record = store.get_job(job_id)
    if record is None:
        return
    rules = Rules.from_dict(record.get("rules") or {})
    store.update_job(job_id, status="processing", progress=5, message="starting pipeline")

    def progress(pct: int, message: str) -> None:
        store.update_job(job_id, status="processing", progress=pct, message=message)

    try:
        with _PIPELINE_LOCK:
            result = process_video(
                video_path,
                store.job_dir(job_id),
                rules,
                video_id=record["video_id"],
                progress=progress,
            )
        store.update_job(
            job_id,
            status="completed",
            progress=100,
            message="done",
            duration_ms=result.duration_ms,
            fps=result.fps,
            encoder_id=result.encoder_id,
            segmenter=result.segmenter,
            warnings=result.warnings,
            error=None,
        )
    except Exception as exc:
        store.update_job(
            job_id,
            status="error",
            message="pipeline failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
