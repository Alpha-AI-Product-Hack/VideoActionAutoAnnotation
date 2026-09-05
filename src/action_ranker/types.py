from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DatasetName = Literal["epic_kitchens", "assembly101"]
SplitName = Literal["validation"]


@dataclass(frozen=True)
class SourceVideo:
    dataset: DatasetName
    video_id: str
    split: SplitName
    media_path: str
    labels_path: str


@dataclass(frozen=True)
class GoldInterval:
    video_id: str
    start_sec: float
    end_sec: float
    gold_action: str
    gold_verb_id: int | None = None
    gold_noun_id: int | None = None


@dataclass
class ActionRanking:
    clip_id: str
    labels: list[str]
    cosine_similarity: list[float]
    cosine_distance: list[float]
    pred_action: str
    identity_keys: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DictionaryRow:
    label: str
    verb_id: int | None = None
    noun_id: int | None = None

    def identity_key(self, dictionary_id: str) -> Any:
        if (
            dictionary_id == "epic_kitchens_observed"
            and self.verb_id is not None
            and self.noun_id is not None
        ):
            return [self.verb_id, self.noun_id]
        return self.label



@dataclass
class PredictionRecord:
    dataset: str
    video_id: str
    start_sec: float
    end_sec: float
    gold_action: str
    pred_action: str
    topk_labels: list[str]
    topk_scores: list[float]
    dictionary_id: str
    encoder_id: str
    prompt_id: str
    gold_rank: int | None
    frame_count: int
    inference_s: float | None = None
    decode_s: float | None = None
    encode_s: float | None = None
    rank_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrozenSliceList:
    epic_kitchens_ids: list[str]
    assembly101_ids: list[str]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsReport:
    dictionary_id: str
    n_clips: int
    n_mismatch: int
    n_skipped_intervals: int
    action_top1: float
    action_top3: float
    action_top5: float
    action_macro_f1: float
    action_top20: float | None = None
    action_top50: float | None = None
    verb_top1_diagnostic: float | None = None
    inference_s_mean: float | None = None
    inference_s_sum: float | None = None
    decode_s_mean: float | None = None
    encode_s_mean: float | None = None
    rank_s_mean: float | None = None
    warmup_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class InferenceTiming:
    clip_duration_s: float
    warmup_s: float
    inference_s: float
    encoder_id: str
    frame_count: int
    decode_s: float | None = None
    encode_s: float | None = None
    rank_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class SkipEvent:
    reason: str
    video_id: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
