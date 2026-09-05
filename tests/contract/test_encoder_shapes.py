import numpy as np

from action_ranker.encode_actions import encode_action_batch
from action_ranker.encode_clips import encode_clip_batch
from action_ranker.stub_encoder import StubEncoder


def test_stub_shapes_match():
    enc = StubEncoder()
    clips = np.zeros((2, enc.num_frames, 3, 16, 16), dtype=np.float32)
    clip_emb = encode_clip_batch(clips, enc)
    text_emb = encode_action_batch(["put", "take"], enc)
    assert clip_emb.shape == (2, enc.dim)
    assert text_emb.shape == (2, enc.dim)
    assert clip_emb.shape[1] == text_emb.shape[1]
