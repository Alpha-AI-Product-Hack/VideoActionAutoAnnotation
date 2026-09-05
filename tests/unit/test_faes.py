import numpy as np

from action_ranker.faes import aggregate_frame_action_scores, rank_faes_scores


def test_top2_mean_uses_best_two_frames_per_action():
    frame_scores = np.array(
        [
            [0.1, 0.4],
            [0.9, 0.2],
            [0.7, 0.6],
        ],
        dtype=np.float32,
    )

    aggregated = aggregate_frame_action_scores(frame_scores, "top2_mean")

    assert np.allclose(aggregated, [0.8, 0.5])


def test_rank_faes_scores_returns_descending_aggregated_scores():
    frame_scores = np.array(
        [
            [0.1, 0.8, 0.2],
            [0.4, 0.7, 0.9],
        ],
        dtype=np.float32,
    )

    ranking, debug = rank_faes_scores(
        frame_scores,
        ["a", "b", "c"],
        clip_id="clip",
        aggregation="mean",
    )

    assert ranking.clip_id == "clip"
    assert ranking.labels == ["b", "c", "a"]
    assert ranking.pred_action == "b"
    assert debug.frame_scores_shape == [2, 3]
    assert debug.top_frame_index == 1


def test_rejects_empty_faes_scores():
    empty = np.zeros((0, 3), dtype=np.float32)

    try:
        aggregate_frame_action_scores(empty, "mean")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
