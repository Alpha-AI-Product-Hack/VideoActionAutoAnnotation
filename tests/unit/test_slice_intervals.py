import csv
from pathlib import Path

from action_ranker.slice import load_intervals


def test_load_intervals_logs_unparseable_and_empty(tmp_path: Path):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "start_sec", "end_sec", "gold_action"],
        )
        writer.writeheader()
        writer.writerow({"video_id": "v1", "start_sec": "0", "end_sec": "1", "gold_action": "ok"})
        writer.writerow({"video_id": "v1", "start_sec": "bad", "end_sec": "1", "gold_action": "x"})
        writer.writerow({"video_id": "v1", "start_sec": "0", "end_sec": "1", "gold_action": ""})
        writer.writerow({"video_id": "v1", "start_sec": "2", "end_sec": "1", "gold_action": "rev"})
    rows, skips = load_intervals(path, "v1")
    assert len(rows) == 1
    assert rows[0].gold_action == "ok"
    reasons = {event.reason for event in skips}
    assert reasons == {"unparseable_interval", "empty_action", "invalid_interval"}
