"""End-to-end MVP glue: boundaries → clips → action labels → export JSON."""

from pipeline.run import process_video
from pipeline.types import ActionSegment, PipelineResult, Rules

__all__ = ["ActionSegment", "PipelineResult", "Rules", "process_video"]
