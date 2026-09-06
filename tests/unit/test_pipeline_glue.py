from __future__ import annotations

from pipeline.classify import match_object
from pipeline.export import segments_to_csv, segments_to_json
from pipeline.segment import boundaries_to_intervals, uniform_boundaries
from pipeline.types import ActionSegment, Rules


def test_boundaries_to_intervals_covers_duration() -> None:
    intervals = boundaries_to_intervals([2.0, 5.5], duration_s=10.0, min_duration_s=0.5)
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == 10.0
    assert all(end > start for start, end in intervals)
    assert all(end - start >= 0.5 for start, end in intervals)


def test_short_boundaries_are_merged() -> None:
    intervals = boundaries_to_intervals([0.1, 0.2, 4.0], duration_s=8.0, min_duration_s=1.0)
    assert intervals[0][0] == 0.0
    assert all(end - start >= 1.0 for start, end in intervals)


def test_no_boundaries_keeps_full_span() -> None:
    assert boundaries_to_intervals([], duration_s=3.0, min_duration_s=0.5) == [(0.0, 3.0)]


def test_uniform_boundaries_on_long_clip() -> None:
    times = uniform_boundaries(20.0, window_s=4.0)
    assert times[0] == 4.0
    assert times[-1] < 20.0


def test_match_object_from_verb_object_label() -> None:
    rules = Rules(actions=["pick_up", "pour"], objects=["cup", "bottle"])
    assert match_object("pour cup", rules) == "cup"
    assert match_object("pick_up", rules) is None


def test_json_and_csv_export_contract() -> None:
    segments = [
        ActionSegment(
            id="1",
            start_ms=1200,
            end_ms=3800,
            action="pick_up",
            object="cup",
            keyframe_ms=2500,
            confidence=0.94,
            model_version="pipeline-0.1",
        )
    ]
    payload = segments_to_json(segments)
    assert '"action": "pick_up"' in payload
    csv_body = segments_to_csv(segments)
    header = csv_body.splitlines()[0]
    assert header == "id,start_ms,end_ms,action,object,keyframe_ms,confidence,model_version"
    assert "pick_up,cup,2500,0.94,pipeline-0.1" in csv_body
