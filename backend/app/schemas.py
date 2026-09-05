from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Action(BaseModel):
    id: str
    start_ms: int
    end_ms: int
    action: str
    object: str
    keyframe_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str


class AnnotationMeta(BaseModel):
    pipeline_version: str
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class AnnotationResult(BaseModel):
    video_name: str
    actions: list[Action]
    meta: AnnotationMeta
