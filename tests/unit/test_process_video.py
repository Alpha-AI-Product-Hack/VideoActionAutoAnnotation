from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.run import process_video
from pipeline.types import Rules


def _write_synthetic_mp4(path: Path, n_frames: int = 40, fps: int = 10) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v on this host")
    for i in range(n_frames):
        value = 30 if i < n_frames // 2 else 220
        frame = np.full((64, 64, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("synthetic mp4 was not written")


def test_process_video_stub_returns_actions_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_ENCODER", "stub")
    video = tmp_path / "clip.mp4"
    try:
        _write_synthetic_mp4(video)
    except Exception as exc:
        pytest.skip(f"could not write synthetic video: {exc}")
    result = process_video(
        video,
        tmp_path / "work",
        Rules(actions=["pick_up", "put_down"], objects=["cup"], min_confidence=0.0),
        encoder_name="stub",
        sources=["motion"],
    )
    assert result.duration_ms > 0
    assert result.segments
    assert (tmp_path / "work" / "actions.json").is_file()
    assert (tmp_path / "work" / "actions.csv").is_file()
    assert (tmp_path / "work" / "clips.json").is_file()
    assert result.segments[0].action in {"pick_up", "put_down"}
