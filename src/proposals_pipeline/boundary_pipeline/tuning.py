"""Oracle threshold tuning against ground truth.

Every tunable lives in a source's `SPACE` or `fusion.SPACE`. Tuning is a
grid or random search per source (solo F1), then a grid over the fusion
parameters with the sources fixed, then rounds of coordinate descent over
everything. The objective is micro-averaged F1 at one tolerance over all
videos of a benchmark, so the result is an oracle for that data, not a
held-out estimate.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import numpy as np

from boundary_pipeline import fusion
from boundary_pipeline.datasets import GroundTruth, VideoItem
from boundary_pipeline.evaluation import Counts, TOLERANCES_S, aggregate, evaluate
from boundary_pipeline.proposals import Proposal
from boundary_pipeline.sources import get_source


@dataclass
class Config:
    sources: dict[str, dict]
    fusion: dict

    def copy(self) -> "Config":
        return Config({k: dict(v) for k, v in self.sources.items()}, {**self.fusion, "weights": dict(self.fusion["weights"])})

    def to_dict(self) -> dict:
        return {"sources": self.sources, "fusion": self.fusion}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(d["sources"], d["fusion"])


@dataclass
class Video:
    item: VideoItem
    gt: GroundTruth
    signals: dict[str, object] = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    def proposals(self, source: str, params: dict) -> list[Proposal]:
        key = (source, tuple(sorted(params.items())))
        if key not in self.cache:
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[key] = get_source(source).propose(self.signals[source], params)
        return self.cache[key]


def load_videos(pairs: list[tuple[VideoItem, GroundTruth]], sources: list[str], load_kwargs: dict[str, dict] | None = None) -> list[Video]:
    load_kwargs = load_kwargs or {}
    videos = []
    for item, gt in pairs:
        v = Video(item, gt)
        for name in sources:
            sig = get_source(name).load(item, **load_kwargs.get(name, {}))
            if sig is None:
                raise FileNotFoundError(f"no cached {name} signal for {item.video_id}; run `extract` first")
            v.signals[name] = sig
        videos.append(v)
    return videos


def fused_times(video: Video, config: Config) -> list[Proposal]:
    per_source = {name: video.proposals(name, params) for name, params in config.sources.items()}
    return fusion.fuse(per_source, config.fusion)


def score(videos: list[Video], config: Config, tolerances=TOLERANCES_S) -> dict[float, Counts]:
    return aggregate([evaluate([p.time_s for p in fused_times(v, config)], v.gt, tolerances) for v in videos])


def objective(videos: list[Video], config: Config, tol: float) -> float:
    return score(videos, config, (tol,))[tol].f1


def solo_config(source: str, params: dict, others: list[str]) -> Config:
    weights = {s: 0.0 for s in others}
    weights[source] = 1.0
    return Config({source: dict(params)}, {"weights": weights, "nms_window_s": 0.0, "min_fused_score": 0.0})


def _grid(space: dict, budget: int | None, rng: random.Random) -> list[dict]:
    keys = list(space)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    if budget is not None and len(combos) > budget:
        combos = rng.sample(combos, budget)
    return combos


def _valid(source: str, params: dict) -> bool:
    if source == "motion" and params["signal"].startswith("d") and params["mode"] == "valley":
        return False
    return True


def tune_source(videos: list[Video], source: str, tol: float, budget: int | None, rng: random.Random, log=print) -> tuple[dict, float]:
    space = get_source(source).SPACE
    best, best_f1 = None, -1.0
    seen = set()
    for params in _grid(space, budget, rng):
        if not _valid(source, params):
            continue
        if source == "abd" and params["mode"] != "fused":
            params = {**params, "gebd_weight": space["gebd_weight"][0]}
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        f1 = objective(videos, solo_config(source, params, []), tol)
        if f1 > best_f1:
            best, best_f1 = params, f1
    log(f"  [{source}] solo best F1@{tol}s = {best_f1:.3f}  {best}")
    return best, best_f1


def tune_fusion(videos: list[Video], config: Config, tol: float, log=print) -> tuple[dict, float]:
    names = list(config.sources)
    best, best_f1 = None, -1.0
    for weights in itertools.product(fusion.SPACE["weight"], repeat=len(names)):
        if not any(weights):
            continue
        for window in fusion.SPACE["nms_window_s"]:
            for min_score in fusion.SPACE["min_fused_score"]:
                fp = {"weights": dict(zip(names, weights)), "nms_window_s": window, "min_fused_score": min_score}
                f1 = objective(videos, Config(config.sources, fp), tol)
                if f1 > best_f1:
                    best, best_f1 = fp, f1
    log(f"  [fusion] best F1@{tol}s = {best_f1:.3f}  {best}")
    return best, best_f1


def coordinate_descent(videos: list[Video], config: Config, tol: float, rounds: int, log=print) -> tuple[Config, float]:
    config = config.copy()
    current = objective(videos, config, tol)
    for r in range(rounds):
        improved = False
        for source, params in config.sources.items():
            for key, values in get_source(source).SPACE.items():
                for v in values:
                    if v == params[key]:
                        continue
                    trial = config.copy()
                    trial.sources[source][key] = v
                    if not _valid(source, trial.sources[source]):
                        continue
                    f1 = objective(videos, trial, tol)
                    if f1 > current + 1e-9:
                        config, current, improved = trial, f1, True
                        params = config.sources[source]
        for source in config.sources:
            for w in fusion.SPACE["weight"]:
                trial = config.copy()
                trial.fusion["weights"][source] = w
                f1 = objective(videos, trial, tol)
                if f1 > current + 1e-9:
                    config, current, improved = trial, f1, True
        for key in ("nms_window_s", "min_fused_score"):
            for v in fusion.SPACE[key]:
                trial = config.copy()
                trial.fusion[key] = v
                f1 = objective(videos, trial, tol)
                if f1 > current + 1e-9:
                    config, current, improved = trial, f1, True
        log(f"  [descent] round {r + 1}: F1@{tol}s = {current:.3f}")
        if not improved:
            break
    return config, current


def tune(videos: list[Video], sources: list[str], tol: float = 1.0, budget: int | None = 1500, rounds: int = 3, seed: int = 0, log=print) -> tuple[Config, dict]:
    rng = random.Random(seed)
    solo: dict[str, dict] = {}
    source_params = {}
    for s in sources:
        params, f1 = tune_source(videos, s, tol, budget, rng, log)
        source_params[s] = params
        solo[s] = {"params": params, "f1": f1}
    config = Config(source_params, dict(fusion.DEFAULTS, weights={s: 1.0 for s in sources}))
    fp, _ = tune_fusion(videos, config, tol, log)
    config = Config(source_params, fp)
    config, f1 = coordinate_descent(videos, config, tol, rounds, log)
    return config, {"solo": solo, "fused_f1": f1}


def ablation(videos: list[Video], config: Config, tol: float) -> dict:
    """Solo scores at the tuned parameters and leave-one-source-out scores."""
    out = {"fused": {str(t): c.summary() for t, c in score(videos, config).items()}}
    names = list(config.sources)
    for s in names:
        solo = solo_config(s, config.sources[s], [n for n in names if n != s])
        out[f"solo:{s}"] = {str(t): c.summary() for t, c in score(videos, solo).items()}
        if len(names) > 1 and config.fusion["weights"].get(s, 0) > 0:
            loo = config.copy()
            loo.fusion["weights"][s] = 0.0
            out[f"without:{s}"] = {str(t): c.summary() for t, c in score(videos, loo).items()}
    return out
