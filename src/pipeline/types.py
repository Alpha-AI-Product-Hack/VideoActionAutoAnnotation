from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_ACTIONS = ["pick_up", "put_down", "pour", "open", "close", "move"]
DEFAULT_OBJECTS = ["cup", "glass", "bottle", "drawer", "box"]


@dataclass
class Rules:
    actions: list[str] = field(default_factory=lambda: list(DEFAULT_ACTIONS))
    objects: list[str] = field(default_factory=lambda: list(DEFAULT_OBJECTS))
    min_duration_ms: int = 500
    min_confidence: float = 0.7
    model_version: str = "pipeline-0.1"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Rules:
        raw = dict(data or {})
        actions = _string_list(raw.get("actions"), DEFAULT_ACTIONS)
        objects = _string_list(raw.get("objects"), DEFAULT_OBJECTS)
        min_duration_ms = _as_int(raw.get("min_duration_ms"), 500)
        min_confidence = _as_float(raw.get("min_confidence"), 0.7)
        model_version = str(raw.get("model_version") or "pipeline-0.1")
        return cls(
            actions=actions,
            objects=objects,
            min_duration_ms=max(1, min_duration_ms),
            min_confidence=min(1.0, max(0.0, min_confidence)),
            model_version=model_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClipInterval:
    clip_id: str
    start_sec: float
    end_sec: float
    path: str | None = None


@dataclass
class ActionSegment:
    id: str
    start_ms: int
    end_ms: int
    action: str
    object: str | None
    keyframe_ms: int
    confidence: float
    model_version: str
    clip_id: str | None = None
    topk_labels: list[str] = field(default_factory=list)
    topk_scores: list[float] = field(default_factory=list)

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "action": self.action,
            "object": self.object,
            "keyframe_ms": self.keyframe_ms,
            "confidence": self.confidence,
            "model_version": self.model_version,
        }


@dataclass
class PipelineResult:
    video_id: str
    duration_ms: int
    fps: float
    segments: list[ActionSegment]
    encoder_id: str
    segmenter: str
    clips: list[ClipInterval]
    warnings: list[str] = field(default_factory=list)

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "duration_ms": self.duration_ms,
            "fps": self.fps,
            "encoder_id": self.encoder_id,
            "segmenter": self.segmenter,
            "n_segments": len(self.segments),
            "n_clips": len(self.clips),
            "warnings": self.warnings,
        }


def _string_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        return list(default)
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        label = str(item).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out or list(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
