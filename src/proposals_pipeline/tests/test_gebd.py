import numpy as np

from action_boundaries.gebd import _block_frame_offsets


def test_block_frame_offsets_shape_and_symmetry():
    offsets = _block_frame_offsets(frames_per_side=5, ds_frames=3)
    assert len(offsets) == 10  # 2 * frames_per_side
    assert 0 not in offsets  # current frame itself is excluded
    np.testing.assert_array_equal(np.sort(-offsets), np.sort(offsets))  # symmetric around 0


def test_block_frame_offsets_spacing():
    offsets = _block_frame_offsets(frames_per_side=5, ds_frames=3)
    np.testing.assert_array_equal(offsets, [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15])


def test_block_frame_offsets_ds_one():
    offsets = _block_frame_offsets(frames_per_side=5, ds_frames=1)
    np.testing.assert_array_equal(offsets, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])


def test_block_frame_offsets_different_frames_per_side():
    offsets = _block_frame_offsets(frames_per_side=2, ds_frames=1)
    np.testing.assert_array_equal(offsets, [-2, -1, 1, 2])
