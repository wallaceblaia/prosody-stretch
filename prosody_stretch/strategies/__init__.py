"""Strategies subpackage for duration adjustment."""

from .pause import PauseManipulator
from .stretch import TimeStretcher
from .vowel import VowelExtender

__all__ = [
    "PauseManipulator",
    "TimeStretcher",
    "VowelExtender",
]
