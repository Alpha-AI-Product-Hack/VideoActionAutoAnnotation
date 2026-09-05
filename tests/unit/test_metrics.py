import pytest

from action_ranker.metrics import compute_action_metrics
from action_ranker.types import PredictionRecord


def _rec(dictionary_id: str, gold: str, pred: str, rank: int | None) -> PredictionRecord:
    return PredictionRecord(
        dataset="assembly101",
        video_id="v",
        start_sec=0.0,
        end_sec=1.0,
        gold_action=gold,
        pred_action=pred,
        topk_labels=[pred],
        topk_scores=[1.0],
        dictionary_id=dictionary_id,
        encoder_id="stub",
        prompt_id="the_action_is",
        gold_rank=rank,
        frame_count=8,
    )


def test_refuses_mixed_banks():
    rows = [
        _rec("assembly101_coarse", "a", "a", 1),
        _rec("epic_kitchens_observed", "b", "b", 1),
    ]
    with pytest.raises(ValueError, match="mix"):
        compute_action_metrics(rows, "assembly101_coarse")


def test_top1_on_kept_only():
    rows = [
        _rec("assembly101_coarse", "a", "a", 1),
        _rec("assembly101_coarse", "b", "z", 4),
        _rec("assembly101_coarse", "missing", "a", None),
    ]
    report = compute_action_metrics(rows, "assembly101_coarse", n_skipped_intervals=1)
    assert report.n_clips == 2
    assert report.n_mismatch == 1
    assert report.action_top1 == 0.5
    assert report.action_top5 == 1.0
    assert report.action_top20 == 1.0
    assert report.action_top50 == 1.0


def test_aggregates_inference_time():
    fast = _rec("assembly101_coarse", "a", "a", 1)
    fast.inference_s = 0.10
    fast.decode_s = 0.06
    fast.encode_s = 0.03
    fast.rank_s = 0.01
    slow = _rec("assembly101_coarse", "b", "z", 4)
    slow.inference_s = 0.30
    slow.decode_s = 0.20
    slow.encode_s = 0.08
    slow.rank_s = 0.02
    report = compute_action_metrics([fast, slow], "assembly101_coarse", warmup_s=1.5)
    assert report.inference_s_mean == pytest.approx(0.20)
    assert report.inference_s_sum == pytest.approx(0.40)
    assert report.decode_s_mean == pytest.approx(0.13)
    assert report.warmup_s == 1.5
    payload = report.to_dict()
    assert "inference_s_mean" in payload
    assert "verb_top1_diagnostic" not in payload
