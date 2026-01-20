"""
Prosody-Stretch: Natural audio duration adjustment for dubbing synchronization.
"""

from .core.stretcher import ProsodyStretcher
from .core.segment import Segment, WordSegment, SilenceSegment
from .core.report import AdjustmentReport

__version__ = "0.1.1"
__all__ = [
    "ProsodyStretcher",
    "Segment",
    "WordSegment", 
    "SilenceSegment",
    "AdjustmentReport",
]
