import numpy as np

from action_ranker.rank import gold_rank_1based, rank_actions, ranking_from_similarities
from action_ranker.types import ActionRanking


def test_sorts_by_ascending_cosine_distance():
    labels = ["a", "b", "c"]
    clip = np.array([1.0, 0.0], dtype=np.float32)
    texts = np.array(
        [
            [0.2, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ranking = rank_actions(clip, texts, labels)
    assert ranking.pred_action == "b"
    assert ranking.labels[0] == "b"
    dists = ranking.cosine_distance
    assert dists == sorted(dists)
    assert set(ranking.labels) == set(labels)


def test_ranking_from_similarities_sorts_descending_cosine():
    ranking = ranking_from_similarities(
        np.array([0.1, 0.9, 0.3], dtype=np.float32),
        ["a", "b", "c"],
        clip_id="x",
    )
    assert ranking.pred_action == "b"
    assert ranking.labels == ["b", "c", "a"]
    assert ranking.cosine_similarity[0] == 0.9


def test_epic_gold_rank_uses_verb_noun_ids():
    ranking = ActionRanking(
        clip_id="c",
        labels=["add cheese", "add banana"],
        cosine_similarity=[0.9, 0.1],
        cosine_distance=[0.1, 0.9],
        pred_action="add cheese",
        identity_keys=[[46, 32], [46, 146]],
    )
    assert gold_rank_1based(
        ranking,
        "wrong string",
        gold_verb_id=46,
        gold_noun_id=146,
        dictionary_id="epic_kitchens_observed",
    ) == 2
    assert gold_rank_1based(
        ranking,
        "add banana",
        gold_verb_id=99,
        gold_noun_id=99,
        dictionary_id="epic_kitchens_observed",
    ) is None
