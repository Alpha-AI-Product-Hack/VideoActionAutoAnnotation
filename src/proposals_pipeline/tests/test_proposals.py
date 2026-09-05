import numpy as np

from boundary_pipeline.proposals import Proposal, nms_1d, normalize_scores


def test_normalize_scores_min_max_and_constant():
    props = [Proposal(0, 1, "a"), Proposal(1, 3, "a"), Proposal(2, 5, "a")]
    np.testing.assert_allclose(normalize_scores(props), [0, 0.5, 1])
    np.testing.assert_allclose(normalize_scores([Proposal(0, 2, "a"), Proposal(1, 2, "a")]), [1, 1])
    assert len(normalize_scores([])) == 0


def test_nms_keeps_highest_and_sums_other_source_support():
    props = [Proposal(10.0, 0.5, "a"), Proposal(10.2, 0.9, "b"), Proposal(10.3, 0.4, "b"), Proposal(20.0, 0.3, "a")]
    kept, fused = nms_1d(props, np.array([p.score for p in props]), window_s=0.5)
    assert [p.time_s for p in kept] == [10.2, 20.0]
    np.testing.assert_allclose(fused, [0.9 + 0.5, 0.3])  # same-source neighbour (10.3) adds nothing


def test_nms_zero_window_keeps_everything():
    props = [Proposal(1.0, 1, "a"), Proposal(1.0, 2, "b")]
    kept, _ = nms_1d(props, np.array([1.0, 2.0]), window_s=0.0)
    assert len(kept) == 1  # identical times collide even at window 0
    kept, _ = nms_1d([Proposal(1.0, 1, "a"), Proposal(1.5, 2, "b")], np.array([1.0, 2.0]), window_s=0.0)
    assert len(kept) == 2
