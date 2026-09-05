from pathlib import Path

import numpy as np

from action_ranker.text_cache import load_or_build_text_cache


def test_cache_miss_on_dictionary_change(tmp_path, monkeypatch):
    import action_ranker.text_cache as tc

    monkeypatch.setattr(tc, "CACHE_ROOT", Path(tmp_path))
    calls = {"n": 0}

    def encode(labels):
        calls["n"] += 1
        return np.ones((len(labels), 4), dtype=np.float32)

    a = load_or_build_text_cache("enc", "bank_a", "the_action_is", ["x"], encode)
    b = load_or_build_text_cache("enc", "bank_a", "the_action_is", ["x"], encode)
    c = load_or_build_text_cache("enc", "bank_b", "the_action_is", ["x"], encode)
    assert calls["n"] == 2
    assert a.shape == b.shape == (1, 4)
    assert c.shape == (1, 4)
