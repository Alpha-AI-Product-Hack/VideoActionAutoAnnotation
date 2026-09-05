from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.pipeline_stub import run_pipeline
from app.schemas import AnnotationResult

router = APIRouter()
log = logging.getLogger("backend.annotate")

_REQUIRED_RULE_KEYS = ("actions", "objects", "min_duration_ms", "min_confidence")


def _validate_rules(obj: object) -> dict:
    if not isinstance(obj, dict):
        raise HTTPException(status_code=422, detail="rules must be a JSON object")

    missing = [k for k in _REQUIRED_RULE_KEYS if k not in obj]
    if missing:
        raise HTTPException(status_code=422, detail=f"rules missing required key(s): {', '.join(missing)}")

    actions = obj.get("actions")
    if not isinstance(actions, list) or not all(isinstance(x, str) for x in actions):
        raise HTTPException(status_code=422, detail="rules.actions must be an array of strings")

    objects = obj.get("objects")
    if not isinstance(objects, list) or not all(isinstance(x, str) for x in objects):
        raise HTTPException(status_code=422, detail="rules.objects must be an array of strings")

    min_duration_ms = obj.get("min_duration_ms")
    if not isinstance(min_duration_ms, int) or min_duration_ms < 0:
        raise HTTPException(status_code=422, detail="rules.min_duration_ms must be a non-negative integer")

    min_confidence = obj.get("min_confidence")
    if not isinstance(min_confidence, (int, float)) or not (0.0 <= float(min_confidence) <= 1.0):
        raise HTTPException(status_code=422, detail="rules.min_confidence must be a number in [0, 1]")

    return obj



@router.post("/api/annotate", response_model=AnnotationResult)
async def annotate(
    video: UploadFile = File(...),
    rules: str | None = Form(None),
    duration_ms: int | None = Form(None),
    fps: float | None = Form(None),
    model: str | None = Form(None),
) -> AnnotationResult:
    video_bytes = await video.read()
    if not rules:
        raise HTTPException(
            status_code=422,
            detail="rules is required and must be a JSON string matching docs/Data_Sample.md rules.json",
        )
    try:
        parsed = json.loads(rules)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"invalid rules JSON: {e.msg} (line {e.lineno}, col {e.colno})",
        ) from e

    rules_obj = _validate_rules(parsed)
    log.info(
        "annotate_request video_name=%s size=%d rules_present=%s duration_ms=%s fps=%s model=%s",
        video.filename,
        len(video_bytes),
        bool(rules),
        duration_ms,
        fps,
        model,
    )
    return run_pipeline(
        video_bytes=video_bytes,
        video_name=video.filename or "video",
        rules=rules_obj,
        duration_ms=duration_ms,
        fps=fps,
        model=model,
    )
