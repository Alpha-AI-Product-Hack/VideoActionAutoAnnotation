from __future__ import annotations

import numpy as np

from action_ranker.encoder_protocol import FrozenEncoder
from action_ranker.prompts import PROMPT_ID, render_action_prompt


def encode_action_batch(
    labels: list[str],
    encoder: FrozenEncoder,
    prompt_id: str = PROMPT_ID,
) -> np.ndarray:
    if not labels:
        raise ValueError("Empty dictionary is not allowed")
    texts = [render_action_prompt(label, prompt_id) for label in labels]
    embeddings = encoder.encode_texts(texts)
    out = np.asarray(embeddings, dtype=np.float32)
    if out.ndim != 2 or out.shape[0] != len(labels):
        raise ValueError(f"encoder.encode_texts must return [M, D] but got {out.shape}")
    return out
