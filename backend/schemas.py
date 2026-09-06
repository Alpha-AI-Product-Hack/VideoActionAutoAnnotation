from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionSegment(BaseModel):
    id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    action: str
    object: str | None = None
    keyframe_ms: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    model_version: str = "pipeline-0.1"
    clip_id: str | None = None


class VideoAnnotation(BaseModel):
    video_id: str
    duration_ms: int = 0
    fps: float = 0.0
    version: int = 1
    segments: list[ActionSegment] = Field(default_factory=list)


class ClipInfo(BaseModel):
    clip_id: str
    start_sec: float
    end_sec: float
    url: str | None = None


class JobStatus(BaseModel):
    job_id: str
    video_id: str
    status: Literal["queued", "processing", "completed", "error"]
    progress: int = 0
    message: str = ""
    error: str | None = None
    encoder_id: str | None = None
    segmenter: str | None = None


class VideoRecord(BaseModel):
    video_id: str
    job_id: str
    name: str
    duration_ms: int = 0
    status: str
    uploaded_at: str
    file_url: str
    annotation_url: str
    clips: list[ClipInfo] = Field(default_factory=list)


class UploadResponse(BaseModel):
    video_id: str
    job_id: str
    status: str
    name: str


class AnnotationUpdate(BaseModel):
    segments: list[ActionSegment]


def segment_from_dict(raw: dict[str, Any]) -> ActionSegment:
    return ActionSegment(
        id=str(raw.get("id") or ""),
        start_ms=int(raw.get("start_ms") or 0),
        end_ms=int(raw.get("end_ms") or 0),
        action=str(raw.get("action") or "unknown"),
        object=raw.get("object"),
        keyframe_ms=int(raw.get("keyframe_ms") or 0),
        confidence=float(raw.get("confidence") or 0.0),
        model_version=str(raw.get("model_version") or "pipeline-0.1"),
        clip_id=raw.get("clip_id"),
    )
