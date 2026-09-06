from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.store import JobStore
from pipeline.types import ActionSegment, ClipInterval, PipelineResult, Rules


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_ENCODER", "stub")
    import backend.main as main_mod

    main_mod.STORE = JobStore(tmp_path)

    def fake_process(video_path, work_dir, rules, **kwargs):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        parsed = rules if isinstance(rules, Rules) else Rules.from_dict(rules)
        segment = ActionSegment(
            id="1",
            start_ms=0,
            end_ms=1000,
            action=parsed.actions[0],
            object=parsed.objects[0] if parsed.objects else None,
            keyframe_ms=500,
            confidence=0.9,
            model_version=parsed.model_version,
            clip_id="clip_000",
        )
        (work_dir / "clips").mkdir(exist_ok=True)
        (work_dir / "clips.json").write_text(
            json.dumps(
                {
                    "clips": [
                        {
                            "clip_id": "clip_000",
                            "video_id": Path(video_path).name,
                            "start_sec": 0.0,
                            "end_sec": 1.0,
                            "clip_path": None,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        from pipeline.export import write_actions_csv, write_actions_json

        write_actions_json(work_dir / "actions.json", [segment])
        write_actions_csv(work_dir / "actions.csv", [segment])
        return PipelineResult(
            video_id=kwargs.get("video_id") or "vid",
            duration_ms=1000,
            fps=10.0,
            segments=[segment],
            encoder_id="stub-d64",
            segmenter="uniform",
            clips=[ClipInterval("clip_000", 0.0, 1.0)],
        )

    monkeypatch.setattr("backend.jobs.process_video", fake_process)
    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as test_client:
        yield test_client, main_mod.STORE


def test_health(client) -> None:
    test_client, _store = client
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_run_and_export(client, tmp_path) -> None:
    test_client, store = client
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    rules = json.dumps({"actions": ["pick_up"], "objects": ["cup"], "min_confidence": 0.1})
    response = test_client.post(
        "/api/videos",
        files={"file": ("demo.mp4", video.read_bytes(), "video/mp4")},
        data={"rules": rules, "model_version": "pipeline-0.1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    video_id = body["video_id"]
    job_id = body["job_id"]

    job = test_client.get(f"/api/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["status"] == "completed"

    annotation = test_client.get(f"/api/videos/{video_id}/annotation")
    assert annotation.status_code == 200
    segments = annotation.json()["segments"]
    assert segments[0]["action"] == "pick_up"

    exported = test_client.get(f"/api/videos/{video_id}/export?format=json")
    assert exported.status_code == 200
    assert exported.json()[0]["action"] == "pick_up"

    csv_export = test_client.get(f"/api/videos/{video_id}/export?format=csv")
    assert csv_export.status_code == 200
    assert "pick_up,cup" in csv_export.text

    updated = dict(segments[0])
    updated["action"] = "put_down"
    saved = test_client.put(f"/api/videos/{video_id}/annotation", json={"segments": [updated]})
    assert saved.status_code == 200
    assert saved.json()["segments"][0]["action"] == "put_down"
    assert store.annotation(job_id).segments[0].action == "put_down"
