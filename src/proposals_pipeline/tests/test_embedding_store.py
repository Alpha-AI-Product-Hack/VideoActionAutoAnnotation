import time

import numpy as np
import pytest

from action_boundaries.embedding_store import EmbeddingKey, EmbeddingStore


@pytest.fixture
def fake_video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"not a real video, just needs to exist for stat()")
    return p


def _key(video_path, **overrides):
    defaults = dict(
        video_path=str(video_path),
        checkpoint="Dev-Jahn/vjepa2.1-vitl-fpc64-384",
        start_s=0.0,
        duration_s=60.0,
        window_s=1.0,
        stride_s=0.125,
        frames_per_window=8,
    )
    defaults.update(overrides)
    return EmbeddingKey(**defaults)


def test_miss_then_put_then_hit(tmp_path, fake_video):
    store = EmbeddingStore(tmp_path / "store")
    key = _key(fake_video)
    assert store.get(key) is None

    embeddings = np.random.rand(481, 1024).astype(np.float32)
    store.put(key, embeddings)

    loaded = store.get(key)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, embeddings)


def test_different_config_is_a_separate_entry(tmp_path, fake_video):
    store = EmbeddingStore(tmp_path / "store")
    key_a = _key(fake_video, start_s=0.0)
    key_b = _key(fake_video, start_s=30.0)

    store.put(key_a, np.zeros((10, 4), dtype=np.float32))
    assert store.get(key_b) is None
    store.put(key_b, np.ones((10, 4), dtype=np.float32))

    np.testing.assert_array_equal(store.get(key_a), np.zeros((10, 4)))
    np.testing.assert_array_equal(store.get(key_b), np.ones((10, 4)))


def test_video_file_change_invalidates_cache(tmp_path, fake_video):
    store = EmbeddingStore(tmp_path / "store")
    key = _key(fake_video)
    store.put(key, np.zeros((5, 4), dtype=np.float32))
    assert store.get(key) is not None

    time.sleep(0.01)
    fake_video.write_bytes(b"different content, different size and mtime")

    assert store.get(key) is None


def test_relative_and_absolute_paths_hit_the_same_entry(tmp_path, fake_video, monkeypatch):
    store = EmbeddingStore(tmp_path / "store")
    monkeypatch.chdir(fake_video.parent)

    abs_key = _key(fake_video)
    store.put(abs_key, np.arange(8, dtype=np.float32).reshape(2, 4))

    rel_key = _key(fake_video.name)
    loaded = store.get(rel_key)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, np.arange(8, dtype=np.float32).reshape(2, 4))


def test_stats_and_vacuum(tmp_path, fake_video):
    store = EmbeddingStore(tmp_path / "store")
    store.put(_key(fake_video, start_s=0.0), np.zeros((10, 4), dtype=np.float32))
    store.put(_key(fake_video, start_s=10.0), np.zeros((20, 4), dtype=np.float32))

    stats = store.stats()
    assert stats["entries"] == 2
    assert stats["total_windows"] == 30

    (store.array_dir / "orphan.npy").write_bytes(b"\x00")
    assert store.vacuum() == 1
    assert store.stats()["entries"] == 2  # index untouched


def test_moving_the_tree_keeps_the_entry(tmp_path):
    src = tmp_path / "a" / "rec" / "clip.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"video bytes")
    store = EmbeddingStore(tmp_path / "store")
    store.put(_key(src), np.ones((3, 4), dtype=np.float32))

    moved = tmp_path / "b" / "rec" / "clip.mp4"
    moved.parent.mkdir(parents=True)
    src.rename(moved)  # rename keeps mtime and size
    (tmp_path / "store").rename(tmp_path / "store2")
    assert EmbeddingStore(tmp_path / "store2").get(_key(moved)) is not None


def test_migrate_rekeys_old_rows(tmp_path, fake_video):
    store = EmbeddingStore(tmp_path / "store")
    store.put(_key(fake_video), np.zeros((2, 4), dtype=np.float32))
    with store._connect() as con:
        con.execute("UPDATE embeddings SET key_hash = 'old', npy_path = ?", (str(store.array_dir / "old.npy"),))
    (store.array_dir / next(store.array_dir.glob("*.npy")).name).rename(store.array_dir / "old.npy")
    assert store.get(_key(fake_video)) is None
    assert store.migrate() == 1
    assert store.get(_key(fake_video)) is not None
