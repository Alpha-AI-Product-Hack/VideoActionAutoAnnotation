import numpy as np

from boundary_pipeline.datasets import GroundTruth, merge_close
from boundary_pipeline.evaluation import Counts, aggregate, evaluate, match


def test_match_is_one_to_one_and_optimal():
    # Greedy closest-first would pair det 1.0 with gt 1.1 and leave det 1.3 unmatched.
    pairs = match(np.array([1.0, 1.3]), np.array([1.1, 0.6]), tolerance_s=0.5)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_match_respects_tolerance_and_components():
    det = np.array([0.0, 10.0, 10.4, 30.0])
    gt = np.array([0.3, 10.2, 29.0])
    pairs = match(det, gt, tolerance_s=0.5)
    assert (0, 0) in pairs and len(pairs) == 2
    assert all(abs(det[i] - gt[j]) <= 0.5 for i, j in pairs)


def test_evaluate_restricts_to_spans():
    gt = GroundTruth(np.array([5.0, 15.0]), spans=[(0.0, 10.0), (12.0, 20.0)])
    counts = evaluate(np.array([5.1, 11.0, 15.2, 25.0]), gt, tolerances_s=(0.5,))
    c = counts[0.5]
    assert (c.tp, c.n_det, c.n_gt) == (2, 2, 2)
    assert c.f1 == 1.0


def test_aggregate_and_summary():
    total = aggregate([{1.0: Counts(1, 2, 2, [0.1])}, {1.0: Counts(2, 2, 4, [0.2, 0.3])}])
    s = total[1.0].summary()
    assert (s["tp"], s["n_det"], s["n_gt"]) == (3, 4, 6)
    assert abs(s["precision"] - 0.75) < 1e-9 and abs(s["recall"] - 0.5) < 1e-9


def test_merge_close():
    np.testing.assert_allclose(merge_close([1.0, 1.1, 1.15, 3.0, 3.3], window_s=0.2), [1.083333, 3.0, 3.3], atol=1e-5)
    assert len(merge_close([])) == 0
