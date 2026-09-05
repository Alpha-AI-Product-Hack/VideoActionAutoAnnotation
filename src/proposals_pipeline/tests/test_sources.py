import numpy as np

from boundary_pipeline import fusion
from boundary_pipeline.sources import abd, motion, sam3


def test_abd_pure_valley_finds_segment_change():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=16), rng.normal(size=16)
    emb = np.concatenate([np.tile(a, (40, 1)), np.tile(b, (40, 1))]) + rng.normal(scale=0.02, size=(80, 16))
    sig = abd.AbdSignal(0.125, emb, np.array([]), np.array([]))
    props = abd.propose(sig, {**abd.DEFAULTS, "mode": "abd", "delta": 2, "prominence": 1.0})
    assert len(props) == 1
    assert abs(props[0].time_s - 40 * 0.125) < 0.5


def test_abd_gebd_modes_use_b_t():
    emb = np.tile(np.ones(8), (80, 1))
    times = np.arange(80) * 0.125
    b = np.zeros(80)
    b[50] = 1.0
    sig = abd.AbdSignal(0.125, emb, times, b)
    for mode in ("gebd", "fused"):
        props = abd.propose(sig, {**abd.DEFAULTS, "mode": mode, "prominence": 1.0})
        assert len(props) == 1 and abs(props[0].time_s - 50 * 0.125) < 0.3


def test_motion_peak_and_valley():
    t = np.arange(0, 20, 0.125)
    r = np.exp(-((t - 10) ** 2) / 0.5) * 20
    sig = motion.MotionSignal(t, np.zeros_like(t), r, np.zeros_like(t), np.ones_like(t))
    peaks = motion.propose(sig, {**motion.DEFAULTS, "signal": "r", "mode": "peak", "sigma": 1, "prominence": 1.0})
    assert len(peaks) == 1 and abs(peaks[0].time_s - 10) < 0.3
    valleys = motion.propose(sig, {**motion.DEFAULTS, "signal": "r", "mode": "valley", "sigma": 1, "prominence": 1.0})
    assert all(abs(p.time_s - 10) > 0.5 for p in valleys)
    d = motion.propose(sig, {**motion.DEFAULTS, "signal": "dr", "mode": "peak", "sigma": 1, "prominence": 1.0, "min_distance_s": 0.25})
    assert len(d) == 2  # rising and falling edge


def _sam3_signal(contact_frames, tracks):
    n = 40
    times = np.arange(n) * 0.2
    frames = np.asarray(contact_frames)
    return sam3.Sam3Signal(
        times_s=times, n_hands=np.ones(n, int), obj_frame=frames, obj_score=np.full(len(frames), 0.9),
        obj_track=np.asarray(tracks), obj_overlap=np.full((len(frames), 3), 0.2), track_len=np.full(len(frames), 10),
    )


def test_sam3_on_off_and_switch_events():
    contact = list(range(10, 20)) + list(range(20, 30))
    tracks = [0] * 10 + [1] * 10
    sig = _sam3_signal(contact, tracks)
    params = {**sam3.DEFAULTS, "min_hold_frames": 1, "min_track_len": 1, "use_switch": 1}
    props = sam3.propose(sig, params)
    kinds = [p.kind for p in props]
    assert kinds == ["on", "switch", "off"]
    assert abs(props[0].time_s - (9.5 * 0.2)) < 1e-9
    assert abs(props[1].time_s - (19.5 * 0.2)) < 1e-9
    no_switch = sam3.propose(sig, {**params, "use_switch": 0})
    assert [p.kind for p in no_switch] == ["on", "off"]


def test_sam3_hysteresis_removes_flicker():
    contact = list(range(10, 20)) + [25]
    sig = _sam3_signal(contact, [0] * 11)
    props = sam3.propose(sig, {**sam3.DEFAULTS, "min_hold_frames": 3, "min_track_len": 1, "use_switch": 0})
    assert [p.kind for p in props] == ["on", "off"]


def test_fusion_weights_and_threshold():
    per_source = {
        "abd": [sam3.Proposal(1.0, 1.0, "abd"), sam3.Proposal(5.0, 0.2, "abd")],
        "sam3": [sam3.Proposal(1.1, 1.0, "sam3")],
    }
    fused = fusion.fuse(per_source, {"weights": {"abd": 1.0, "sam3": 1.0}, "nms_window_s": 0.5, "min_fused_score": 0.0})
    assert [p.time_s for p in fused] == [1.0, 5.0]
    assert fused[0].score == 2.0  # own 1.0 + sam3 support 1.0
    strict = fusion.fuse(per_source, {"weights": {"abd": 1.0, "sam3": 1.0}, "nms_window_s": 0.5, "min_fused_score": 1.5})
    assert [p.time_s for p in strict] == [1.0]
    dropped = fusion.fuse(per_source, {"weights": {"abd": 0.0, "sam3": 1.0}, "nms_window_s": 0.5, "min_fused_score": 0.0})
    assert [p.source for p in dropped] == ["sam3"]
