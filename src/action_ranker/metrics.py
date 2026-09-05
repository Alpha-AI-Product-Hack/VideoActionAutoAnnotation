from __future__ import annotations

from sklearn.metrics import f1_score

from action_ranker.types import MetricsReport, PredictionRecord


def compute_action_metrics(
    records: list[PredictionRecord],
    dictionary_id: str,
    n_skipped_intervals: int = 0,
    warmup_s: float | None = None,
) -> MetricsReport:
    ids = {row.dictionary_id for row in records}
    if any(item != dictionary_id for item in ids):
        raise ValueError("Cannot mix dictionaries into one metrics report")
    mismatch = [row for row in records if row.gold_rank is None]
    kept = [row for row in records if row.gold_rank is not None]
    n_kept = len(kept)
    if n_kept == 0:
        top1 = top3 = top5 = top20 = top50 = macro = 0.0
    else:
        top1 = sum(row.gold_rank == 1 for row in kept) / n_kept
        top3 = sum(row.gold_rank <= 3 for row in kept) / n_kept
        top5 = sum(row.gold_rank <= 5 for row in kept) / n_kept
        top20 = sum(row.gold_rank <= 20 for row in kept) / n_kept
        top50 = sum(row.gold_rank <= 50 for row in kept) / n_kept
        golds = [row.gold_action for row in kept]
        preds = [row.pred_action for row in kept]
        macro = float(f1_score(golds, preds, average="macro", zero_division=0))
    timed = [row for row in kept if row.inference_s is not None]
    inf_mean = inf_sum = dec_mean = enc_mean = rank_mean = None
    if timed:
        inf_sum = float(sum(row.inference_s or 0.0 for row in timed))
        inf_mean = inf_sum / len(timed)
        if all(row.decode_s is not None for row in timed):
            dec_mean = float(sum(row.decode_s or 0.0 for row in timed) / len(timed))
        if all(row.encode_s is not None for row in timed):
            enc_mean = float(sum(row.encode_s or 0.0 for row in timed) / len(timed))
        if all(row.rank_s is not None for row in timed):
            rank_mean = float(sum(row.rank_s or 0.0 for row in timed) / len(timed))
    return MetricsReport(
        dictionary_id=dictionary_id,
        n_clips=n_kept,
        n_mismatch=len(mismatch),
        n_skipped_intervals=n_skipped_intervals,
        action_top1=float(top1),
        action_top3=float(top3),
        action_top5=float(top5),
        action_top20=float(top20),
        action_top50=float(top50),
        action_macro_f1=float(macro),
        inference_s_mean=inf_mean,
        inference_s_sum=inf_sum,
        decode_s_mean=dec_mean,
        encode_s_mean=enc_mean,
        rank_s_mean=rank_mean,
        warmup_s=None if warmup_s is None else float(warmup_s),
    )
