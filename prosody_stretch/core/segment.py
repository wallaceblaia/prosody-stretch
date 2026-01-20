"""Data structures for audio segments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """Base class for audio segments."""
    start_time: float  # seconds
    end_time: float    # seconds
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time
    
    def to_samples(self, sr: int) -> tuple[int, int]:
        """Convert to sample indices."""
        return int(self.start_time * sr), int(self.end_time * sr)


@dataclass
class WordSegment(Segment):
    """A word segment with optional phoneme information."""
    word: str = ""
    phonemes: List[str] = field(default_factory=list)
    is_extensible: bool = False  # True if ends with vowel/prolongable consonant
    
    def __post_init__(self):
        # Check if word ends with extensible sound
        if self.word:
            extensible_endings = ('a', 'e', 'i', 'o', 'u', 's', 'm', 'n', 'l', 'r')
            self.is_extensible = self.word.lower().endswith(extensible_endings)


@dataclass
class SilenceSegment(Segment):
    """A silence/pause segment."""
    min_duration: float = 0.05  # Minimum natural pause (50ms)
    max_extension: float = 0.5   # Maximum extension allowed (500ms)
    
    @property
    def extensible_amount(self) -> float:
        """How much this pause can be extended."""
        return self.max_extension
    
    @property
    def compressible_amount(self) -> float:
        """How much this pause can be compressed."""
        return max(0, self.duration - self.min_duration)


@dataclass
class AdjustmentOperation:
    """A single adjustment operation to apply."""
    segment: Segment
    strategy: str  # 'pause', 'wsola', 'vowel'
    amount: float  # seconds to add (positive) or remove (negative)
    factor: float = 1.0  # stretch factor for wsola (1.0 = no change)


@dataclass  
class AdjustmentPlan:
    """Complete plan for adjusting audio duration."""
    operations: List[AdjustmentOperation] = field(default_factory=list)
    total_adjustment: float = 0.0
    estimated_quality: float = 1.0
    
    def add_operation(self, op: AdjustmentOperation):
        """Add an operation to the plan."""
        self.operations.append(op)
        self.total_adjustment += op.amount
