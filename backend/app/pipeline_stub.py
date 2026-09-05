from __future__ import annotations

import random
import uuid

from app.schemas import Action, AnnotationMeta, AnnotationResult


def run_pipeline(
    video_bytes: bytes,
    video_name: str,
    rules: dict,
    duration_ms: int | None = None,
    fps: float | None = None,
    model: str | None = None,
) -> AnnotationResult:
    seed = (len(video_bytes) << 1) ^ sum(video_bytes[:2048])
    rng = random.Random(seed)

    duration_ms = int(duration_ms or 20_000)
    version = str(model or "mock-0.1.0")
    min_duration_ms = int(rules.get("min_duration_ms", 500))
    min_confidence = float(rules.get("min_confidence", 0.7))
    min_confidence = max(0.0, min(1.0, min_confidence))
    seg_min = max(250, min_duration_ms)
    seg_max = max(seg_min + 400, int(seg_min * 4.5))

    head_gap = min(max(250, rng.randint(350, 1300)), max(250, duration_ms // 10))
    tail_gap = min(max(350, rng.randint(450, 1600)), max(350, duration_ms // 10))
    end_limit = max(0, duration_ms - tail_gap)
    t = min(end_limit, head_gap + rng.randint(0, 250))
    out_actions: list[Action] = []
    i = 0
    actions_space = ["pick_up", "move", "put_down"]
    objects_space = ["cup", "glass"]
    max_actions = max(6, min(30, max(1, duration_ms // max(600, seg_min))))
    while t + seg_min <= end_limit and len(out_actions) < max_actions:
        start_ms = int(t)
        remaining = end_limit - start_ms
        if remaining < seg_min:
            break
        if remaining <= seg_max:
            tail_slack = rng.randint(0, min(280, max(0, remaining - seg_min)))
            length = remaining - tail_slack
        else:
            length = int(rng.randint(seg_min, seg_max))
        end_ms = int(min(end_limit, start_ms + length))
        label = actions_space[i % len(actions_space)]
        obj = objects_space[i % len(objects_space)]
        conf_hi = 0.95
        if min_confidence > conf_hi:
            conf = min_confidence
        else:
            conf = float(min(0.98, rng.uniform(max(0.0, min_confidence), conf_hi)))
        out_actions.append(
            Action(
                id=str(uuid.uuid4()),
                start_ms=start_ms,
                end_ms=end_ms,
                action=label,
                object=obj,
                keyframe_ms=int((start_ms + end_ms) // 2),
                confidence=conf,
                model_version=version,
            )
        )
        gap = rng.randint(60, 240)
        if rng.random() < 0.12:
            gap += rng.randint(250, 800)
        overlap = rng.randint(120, 700) if out_actions and rng.random() < 0.12 else 0
        t = max(0, end_ms + gap - overlap)
        i += 1

    return AnnotationResult(
        video_name=video_name,
        actions=out_actions,
        meta=AnnotationMeta(
            pipeline_version=version,
            warnings=["mock_pipeline"],
            debug={
                "duration_ms": duration_ms,
                "fps": fps,
                "seed": seed,
                "head_gap_ms": head_gap,
                "tail_gap_ms": tail_gap,
                "min_duration_ms": min_duration_ms,
                "min_confidence": min_confidence,
            },
        ),
    )
