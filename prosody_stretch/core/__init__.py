"""Core module for prosody-stretch."""

from .stretcher import ProsodyStretcher
from .segment import Segment, WordSegment, SilenceSegment
from .report import AdjustmentReport
from .planner import DurationPlanner

__all__ = [
    "ProsodyStretcher",
    "Segment",
    "WordSegment",
    "SilenceSegment", 
    "AdjustmentReport",
    "DurationPlanner",
]
