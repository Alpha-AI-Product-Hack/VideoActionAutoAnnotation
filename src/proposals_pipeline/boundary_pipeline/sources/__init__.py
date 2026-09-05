"""Proposal sources. Each module exposes `NAME`, `SPACE` (tunable
parameters and their candidate values), `DEFAULTS`, `extract(item, ...)`
(expensive, writes a per-video cache), `load(item)` (cached signal or
None) and `propose(signal, params)` (cheap, returns scored proposals)."""

from __future__ import annotations

from importlib import import_module

SOURCE_NAMES = ("abd", "motion", "sam3")


def get_source(name: str):
    if name not in SOURCE_NAMES:
        raise KeyError(f"unknown source {name!r}; choose from {SOURCE_NAMES}")
    return import_module(f"boundary_pipeline.sources.{name}")
