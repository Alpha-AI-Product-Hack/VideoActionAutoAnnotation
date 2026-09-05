from __future__ import annotations

import numpy as np

from action_ranker.types import ActionRanking

EPS = 1e-12


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    vecs = np.asarray(matrix, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs[None, :]
        squeeze = True
    else:
        squeeze = False
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    out = vecs / norms
    if squeeze:
        return out[0]
    return out


def rank_actions(
    clip_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list[str],
    clip_id: str = "",
    identity_keys: list | None = None,
) -> ActionRanking:
    if not labels:
        raise ValueError("Empty dictionary is not allowed")
    texts = np.asarray(text_embeddings, dtype=np.float32)
    if texts.ndim != 2:
        raise ValueError("text_embeddings must be [M, D]")
    if texts.shape[0] != len(labels):
        raise ValueError("labels length must match text_embeddings rows")
    keys = identity_keys if identity_keys is not None else list(labels)
    if len(keys) != len(labels):
        raise ValueError("identity_keys length must match labels")
    clip = np.asarray(clip_embedding, dtype=np.float32).reshape(-1)
    if clip.shape[0] != texts.shape[1]:
        raise ValueError("clip and text embedding dims must match")
    clip_hat = l2_normalize(clip)
    text_hat = l2_normalize(texts)
    similarity = text_hat @ clip_hat
    return ranking_from_similarities(
        similarity, labels, clip_id=clip_id, identity_keys=keys
    )


def ranking_from_similarities(
    similarity: np.ndarray,
    labels: list[str],
    clip_id: str = "",
    identity_keys: list | None = None,
) -> ActionRanking:
    if not labels:
        raise ValueError("Empty dictionary is not allowed")
    scores = np.asarray(similarity, dtype=np.float32).reshape(-1)
    if scores.shape[0] != len(labels):
        raise ValueError("similarities length must match labels")
    keys = identity_keys if identity_keys is not None else list(labels)
    if len(keys) != len(labels):
        raise ValueError("identity_keys length must match labels")
    distance = 1.0 - scores
    order = np.argsort(distance, kind="stable")
    ordered_labels = [labels[int(i)] for i in order]
    return ActionRanking(
        clip_id=clip_id,
        labels=ordered_labels,
        cosine_similarity=[float(scores[int(i)]) for i in order],
        cosine_distance=[float(distance[int(i)]) for i in order],
        pred_action=ordered_labels[0],
        identity_keys=[keys[int(i)] for i in order],
    )


def gold_rank_1based(
    ranking: ActionRanking,
    gold_action: str,
    *,
    gold_verb_id: int | None = None,
    gold_noun_id: int | None = None,
    dictionary_id: str = "",
) -> int | None:
    if (
        dictionary_id == "epic_kitchens_observed"
        and gold_verb_id is not None
        and gold_noun_id is not None
    ):
        target = [gold_verb_id, gold_noun_id]
        for index, key in enumerate(ranking.identity_keys):
            if key == target or key == (gold_verb_id, gold_noun_id):
                return index + 1
        return None
    try:
        return ranking.labels.index(gold_action) + 1
    except ValueError:
        return None
