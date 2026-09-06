from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from backend.jobs import run_job
from backend.schemas import AnnotationUpdate, UploadResponse
from backend.store import JobStore
from pipeline.export import segments_to_csv, segments_to_json
from pipeline.types import Rules

STORE = JobStore()
APP_ORIGINS = [
    "http://127.0.0.1:8443",
    "http://localhost:8443",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def create_app() -> FastAPI:
    app = FastAPI(title="Video Action Auto Annotation", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=APP_ORIGINS + os.environ.get("CORS_ORIGINS", "").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/videos", response_model=UploadResponse)
    async def upload_video(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        rules: str = Form(default="{}"),
        model_version: str = Form(default="pipeline-0.1"),
    ) -> UploadResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="missing filename")
        try:
            rules_obj = Rules.from_dict(json.loads(rules or "{}"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid rules JSON: {exc}") from exc
        rules_obj.model_version = model_version or rules_obj.model_version
        record = STORE.create_job(file.filename, rules_obj)
        dest = STORE.source_path(record["job_id"], file.filename)
        dest.write_bytes(await file.read())
        if dest.stat().st_size <= 0:
            STORE.update_job(record["job_id"], status="error", error="empty upload")
            raise HTTPException(status_code=400, detail="empty upload")
        (STORE.job_dir(record["job_id"]) / "rules.json").write_text(
            json.dumps(rules_obj.to_dict(), indent=2), encoding="utf-8"
        )
        background_tasks.add_task(run_job, STORE, record["job_id"], dest)
        return UploadResponse(
            video_id=record["video_id"],
            job_id=record["job_id"],
            status="processing",
            name=record["name"],
        )

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        record = STORE.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return STORE.to_job_status(record)

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str):
        record = STORE.get_by_video_id(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="video not found")
        return STORE.to_video_record(record)

    @app.get("/api/videos/{video_id}/file")
    def get_video_file(video_id: str):
        record = _require_record(video_id)
        path = STORE.source_path(record["job_id"], record["name"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="source video missing")
        return FileResponse(path, media_type="video/mp4", filename=record["name"])

    @app.get("/api/videos/{video_id}/annotation")
    def get_annotation(video_id: str):
        record = _require_record(video_id)
        return STORE.annotation(record["job_id"])

    @app.put("/api/videos/{video_id}/annotation")
    def put_annotation(video_id: str, body: AnnotationUpdate):
        record = _require_record(video_id)
        return STORE.save_annotation(record["job_id"], body.segments)

    @app.get("/api/videos/{video_id}/clips")
    def list_clips(video_id: str):
        record = _require_record(video_id)
        return {"video_id": video_id, "clips": STORE.clip_infos(record["job_id"], video_id)}

    @app.get("/api/videos/{video_id}/clips/{clip_id}")
    def get_clip(video_id: str, clip_id: str):
        record = _require_record(video_id)
        path = STORE.clip_file(record["job_id"], clip_id)
        if path is None:
            raise HTTPException(status_code=404, detail="clip not found")
        return FileResponse(path, media_type="video/mp4", filename=f"{clip_id}.mp4")

    @app.get("/api/videos/{video_id}/export")
    def export_annotation(video_id: str, format: str = "json"):
        record = _require_record(video_id)
        annotation = STORE.annotation(record["job_id"])
        fmt = format.lower().strip()
        stem = Path(record["name"]).stem or video_id
        if fmt == "csv":
            body = segments_to_csv([seg.model_dump() for seg in annotation.segments])
            return Response(
                content=body,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{stem}_actions.csv"'},
            )
        if fmt == "json":
            body = segments_to_json([seg.model_dump() for seg in annotation.segments])
            return Response(
                content=body,
                media_type="application/json; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{stem}_actions.json"'},
            )
        raise HTTPException(status_code=400, detail="format must be json or csv")

    return app


def _require_record(video_id: str):
    record = STORE.get_by_video_id(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="video not found")
    return record


app = create_app()
