import numpy as np
import pytest

from action_boundaries.boundary_detection import detect_boundaries


def _piecewise_constant_embeddings(segment_lens, dim=32, seed=0):
    """Build embeddings that are near-constant within a segment and jump to a
    fresh random direction at each segment boundary, with small iid noise.
    Returns (embeddings, true_boundary_frame_indices).
    """
    rng = np.random.default_rng(seed)
    chunks = []
    boundaries = []
    frame = 0
    for n in segment_lens:
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        noise = rng.normal(scale=0.02, size=(n, dim))
        chunks.append(base[None, :] + noise)
        frame += n
        boundaries.append(frame)
    boundaries = boundaries[:-1]  # last one is just the end of the sequence
    return np.concatenate(chunks, axis=0), boundaries


def test_detects_boundaries_at_segment_changes():
    stride_s = 0.125
    embeddings, true_boundaries = _piecewise_constant_embeddings([16, 20, 24, 18])
    result = detect_boundaries(embeddings, stride_s=stride_s, delta=2, prominence=1.0)

    assert len(result.boundary_times) == len(true_boundaries)
    detected_frames = result.boundary_indices
    for true_frame, detected_frame in zip(true_boundaries, sorted(detected_frames)):
        assert abs(true_frame - detected_frame) <= 2


def test_constant_embeddings_yield_no_boundaries_at_high_prominence():
    # z-scoring is relative, not absolute: it normalizes noise to unit std
    # regardless of its true magnitude, so a *low* threshold still fires
    # occasionally on pure noise (as it would on real video's noise floor).
    # A conservative threshold should still reject it.
    rng = np.random.default_rng(1)
    base = rng.normal(size=32)
    embeddings = np.tile(base, (50, 1)) + rng.normal(scale=1e-4, size=(50, 32))
    result = detect_boundaries(embeddings, stride_s=0.125, delta=3, prominence=4.0)
    assert len(result.boundary_times) == 0


def test_min_distance_enforced():
    stride_s = 0.125
    # Two boundaries only 2 strides apart (0.25s) -- closer than the 0.5s default minimum.
    embeddings, _ = _piecewise_constant_embeddings([10, 2, 10])
    result = detect_boundaries(embeddings, stride_s=stride_s, delta=1, min_distance_s=0.5)
    if len(result.boundary_times) > 1:
        gaps = np.diff(sorted(result.boundary_indices))
        assert np.all(gaps >= round(0.5 / stride_s))


def test_rejects_too_short_sequence():
    with pytest.raises(ValueError):
        detect_boundaries(np.zeros((2, 8)), stride_s=0.125, delta=3)


def test_boundary_scores_are_positive_prominences():
    embeddings, _ = _piecewise_constant_embeddings([16, 16])
    result = detect_boundaries(embeddings, stride_s=0.125, delta=2)
    assert len(result.boundary_scores) == len(result.boundary_times)
    assert np.all(result.boundary_scores > 0)
