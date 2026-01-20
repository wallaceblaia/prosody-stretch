"""Pause manipulation strategy - most natural way to adjust duration."""

import numpy as np
from typing import List, Tuple
from ..core.segment import SilenceSegment
from ..analyzer.audio import AudioAnalyzer


class PauseManipulator:
    """
    Manipulate pauses between words to adjust duration.
    
    This is the most natural strategy as it doesn't alter speech,
    only the silence between words.
    """
    
    def __init__(
        self,
        min_pause_duration: float = 0.05,  # 50ms minimum pause
        max_pause_extension: float = 0.5,   # 500ms max extension per pause
        crossfade_ms: float = 10,           # Crossfade duration
    ):
        """
        Initialize pause manipulator.
        
        Args:
            min_pause_duration: Minimum pause to maintain (seconds)
            max_pause_extension: Maximum extension per pause (seconds)
            crossfade_ms: Crossfade duration for transitions (ms)
        """
        self.min_pause_duration = min_pause_duration
        self.max_pause_extension = max_pause_extension
        self.crossfade_samples = None  # Set based on sample rate
        self.crossfade_ms = crossfade_ms
    
    def extend_pauses(
        self,
        audio: np.ndarray,
        sr: int,
        silences: List[SilenceSegment],
        total_extension: float,
        proportional: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Extend pauses to increase total duration.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            silences: List of detected silences
            total_extension: Total time to add (seconds)
            proportional: If True, distribute proportionally to pause lengths
            
        Returns:
            (modified_audio, actual_extension)
        """
        if not silences or total_extension <= 0:
            return audio, 0.0
        
        self.crossfade_samples = int(self.crossfade_ms * sr / 1000)
        
        # Calculate how much each pause can be extended
        extensible = []
        for s in silences:
            max_ext = min(self.max_pause_extension, s.max_extension)
            extensible.append((s, max_ext))
        
        # Calculate total extensible amount
        total_extensible = sum(ext for _, ext in extensible)
        
        if total_extensible == 0:
            return audio, 0.0
        
        # Distribute the extension
        if proportional:
            extensions = self._distribute_proportional(
                extensible, total_extension
            )
        else:
            extensions = self._distribute_equal(
                extensible, total_extension
            )
        
        # Apply extensions (process from end to start to maintain indices)
        result = audio.copy()
        actual_extension = 0.0
        
        for (silence, _), ext in sorted(
            zip(extensible, extensions),
            key=lambda x: x[0][0].start_time,
            reverse=True
        ):
            if ext > 0:
                result = self._insert_silence(
                    result, sr, silence, ext
                )
                actual_extension += ext
        
        return result, actual_extension
    
    def compress_pauses(
        self,
        audio: np.ndarray,
        sr: int,
        silences: List[SilenceSegment],
        total_compression: float,
        proportional: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Compress pauses to decrease total duration.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            silences: List of detected silences
            total_compression: Total time to remove (seconds, positive value)
            proportional: If True, distribute proportionally
            
        Returns:
            (modified_audio, actual_compression)
        """
        if not silences or total_compression <= 0:
            return audio, 0.0
        
        self.crossfade_samples = int(self.crossfade_ms * sr / 1000)
        
        # Calculate how much each pause can be compressed
        compressible = []
        for s in silences:
            max_comp = s.compressible_amount
            compressible.append((s, max_comp))
        
        # Calculate total compressible amount
        total_compressible = sum(comp for _, comp in compressible)
        
        if total_compressible == 0:
            return audio, 0.0
        
        # Limit compression to available amount
        total_compression = min(total_compression, total_compressible)
        
        # Distribute the compression
        if proportional:
            compressions = self._distribute_proportional(
                compressible, total_compression
            )
        else:
            compressions = self._distribute_equal(
                compressible, total_compression
            )
        
        # Apply compressions (process from end to start)
        result = audio.copy()
        actual_compression = 0.0
        
        for (silence, _), comp in sorted(
            zip(compressible, compressions),
            key=lambda x: x[0][0].start_time,
            reverse=True
        ):
            if comp > 0:
                result = self._remove_silence(
                    result, sr, silence, comp
                )
                actual_compression += comp
        
        return result, actual_compression
    
    def _distribute_proportional(
        self,
        segments: List[Tuple[SilenceSegment, float]],
        total_amount: float
    ) -> List[float]:
        """Distribute amount proportionally to segment capacities."""
        total_capacity = sum(cap for _, cap in segments)
        
        if total_capacity == 0:
            return [0.0] * len(segments)
        
        amounts = []
        remaining = total_amount
        
        for _, capacity in segments:
            portion = (capacity / total_capacity) * total_amount
            amount = min(portion, capacity, remaining)
            amounts.append(amount)
            remaining -= amount
        
        return amounts
    
    def _distribute_equal(
        self,
        segments: List[Tuple[SilenceSegment, float]],
        total_amount: float
    ) -> List[float]:
        """Distribute amount equally across segments."""
        n = len(segments)
        if n == 0:
            return []
        
        per_segment = total_amount / n
        amounts = []
        
        for _, capacity in segments:
            amount = min(per_segment, capacity)
            amounts.append(amount)
        
        return amounts
    
    def _insert_silence(
        self,
        audio: np.ndarray,
        sr: int,
        silence: SilenceSegment,
        duration: float
    ) -> np.ndarray:
        """Insert additional silence at a pause location."""
        insert_point = int((silence.start_time + silence.duration / 2) * sr)
        insert_samples = int(duration * sr)
        
        # Create silence to insert
        silence_audio = np.zeros(insert_samples, dtype=audio.dtype)
        
        # Split and concatenate with crossfade
        before = audio[:insert_point]
        after = audio[insert_point:]
        
        # Simple concatenation with fade
        if len(before) > self.crossfade_samples and len(after) > self.crossfade_samples:
            # Fade out end of before
            fade = np.linspace(1, 0, self.crossfade_samples)
            before_faded = before.copy()
            before_faded[-self.crossfade_samples:] *= fade
            
            # Fade in start of after
            fade = np.linspace(0, 1, self.crossfade_samples)
            after_faded = after.copy()
            after_faded[:self.crossfade_samples] *= fade
            
            return np.concatenate([before_faded, silence_audio, after_faded])
        
        return np.concatenate([before, silence_audio, after])
    
    def _remove_silence(
        self,
        audio: np.ndarray,
        sr: int,
        silence: SilenceSegment,
        duration: float
    ) -> np.ndarray:
        """Remove silence from a pause location."""
        # Calculate region to remove (from center of silence)
        center = (silence.start_time + silence.end_time) / 2
        remove_start = center - duration / 2
        remove_end = center + duration / 2
        
        # Clamp to silence boundaries
        remove_start = max(remove_start, silence.start_time)
        remove_end = min(remove_end, silence.end_time)
        
        start_sample = int(remove_start * sr)
        end_sample = int(remove_end * sr)
        
        before = audio[:start_sample]
        after = audio[end_sample:]
        
        # Crossfade the join
        return AudioAnalyzer.crossfade(
            before, after, self.crossfade_samples
        )
