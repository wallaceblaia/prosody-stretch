"""Vowel extension strategy for subtle duration adjustment."""

import numpy as np
from typing import List, Tuple
from ..core.segment import WordSegment
from ..analyzer.audio import AudioAnalyzer


class VowelExtender:
    """
    Extend vowels at word endings for subtle duration increase.
    
    This mimics natural speech where stressed vowels can be
    slightly elongated without sounding artificial.
    """
    
    def __init__(
        self,
        max_extension_per_word_ms: float = 50,  # 50ms max per word
        min_vowel_duration_ms: float = 50,       # Minimum vowel to extend
        crossfade_ms: float = 10,                # Crossfade for blending
    ):
        """
        Initialize vowel extender.
        
        Args:
            max_extension_per_word_ms: Maximum extension per word ending (ms)
            min_vowel_duration_ms: Minimum vowel duration to consider (ms)
            crossfade_ms: Crossfade duration for blending (ms)
        """
        self.max_extension_per_word_ms = max_extension_per_word_ms
        self.min_vowel_duration_ms = min_vowel_duration_ms
        self.crossfade_ms = crossfade_ms
    
    def extend_endings(
        self,
        audio: np.ndarray,
        sr: int,
        word_segments: List[WordSegment],
        total_extension: float
    ) -> Tuple[np.ndarray, float]:
        """
        Extend word endings to increase duration.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            word_segments: List of word segments
            total_extension: Total time to add (seconds)
            
        Returns:
            (modified_audio, actual_extension)
        """
        if not word_segments or total_extension <= 0:
            return audio, 0.0
        
        # Find extensible words
        extensible = [w for w in word_segments if w.is_extensible]
        
        if not extensible:
            return audio, 0.0
        
        # Calculate extension per word
        max_per_word = self.max_extension_per_word_ms / 1000.0
        per_word = min(total_extension / len(extensible), max_per_word)
        
        # Apply extensions (from end to start to maintain indices)
        result = audio.copy()
        actual_extension = 0.0
        
        for word in sorted(extensible, key=lambda w: w.start_time, reverse=True):
            if actual_extension >= total_extension:
                break
            
            ext = min(per_word, total_extension - actual_extension)
            result, added = self._extend_word_ending(result, sr, word, ext)
            actual_extension += added
        
        return result, actual_extension
    
    def _extend_word_ending(
        self,
        audio: np.ndarray,
        sr: int,
        word: WordSegment,
        duration: float
    ) -> Tuple[np.ndarray, float]:
        """
        Extend the ending of a single word.
        
        Uses frame duplication with crossfading to create
        a natural elongation effect.
        """
        # Find the region to extend (last 10-20% of word)
        word_samples = int(word.duration * sr)
        extend_region = max(int(word_samples * 0.15), int(0.05 * sr))  # 15% or 50ms
        
        word_end_sample = int(word.end_time * sr)
        extend_start = word_end_sample - extend_region
        
        if extend_start < 0 or extend_start >= len(audio):
            return audio, 0.0
        
        # Extract the region to duplicate
        region_to_dup = audio[extend_start:word_end_sample]
        
        # Calculate how many times to repeat (with crossfade)
        extension_samples = int(duration * sr)
        crossfade_samples = int(self.crossfade_ms * sr / 1000)
        
        # Create extended region using loop with crossfade
        extended = self._create_extended_region(
            region_to_dup, extension_samples, crossfade_samples
        )
        
        # Insert the extended region
        before = audio[:extend_start]
        after = audio[word_end_sample:]
        
        # Crossfade insertion
        if len(before) > crossfade_samples:
            result = AudioAnalyzer.crossfade(before, region_to_dup, crossfade_samples)
            result = AudioAnalyzer.crossfade(result, extended, crossfade_samples)
            result = AudioAnalyzer.crossfade(result, after, crossfade_samples)
        else:
            result = np.concatenate([before, region_to_dup, extended, after])
        
        actual_extension = len(extended) / sr
        return result, actual_extension
    
    def _create_extended_region(
        self,
        source: np.ndarray,
        target_samples: int,
        crossfade_samples: int
    ) -> np.ndarray:
        """
        Create extended audio by looping source with crossfade.
        
        This creates a smooth loop of the source audio to achieve
        the target length without obvious repetition artifacts.
        """
        if len(source) == 0 or target_samples <= 0:
            return np.array([], dtype=source.dtype)
        
        # For very short extensions, just duplicate end portion
        if target_samples <= len(source):
            return source[-target_samples:]
        
        # Build extended audio
        result = np.zeros(target_samples, dtype=source.dtype)
        
        # Use the middle portion for looping (more stable)
        loop_start = len(source) // 4
        loop_end = len(source) - len(source) // 4
        loop_region = source[loop_start:loop_end]
        
        if len(loop_region) < crossfade_samples * 2:
            loop_region = source
        
        # Fill with looped content
        pos = 0
        while pos < target_samples:
            remaining = target_samples - pos
            chunk_len = min(len(loop_region), remaining)
            
            # Apply fade in/out at boundaries
            chunk = loop_region[:chunk_len].copy()
            
            if pos == 0 and chunk_len > crossfade_samples:
                # Fade in at start
                fade = np.linspace(0, 1, crossfade_samples)
                chunk[:crossfade_samples] *= fade
            
            if pos + chunk_len >= target_samples and chunk_len > crossfade_samples:
                # Fade out at end
                fade = np.linspace(1, 0, crossfade_samples)
                chunk[-crossfade_samples:] *= fade
            
            result[pos:pos + chunk_len] = chunk
            pos += chunk_len
        
        return result
    
    def estimate_extensible_words(
        self,
        word_segments: List[WordSegment]
    ) -> Tuple[int, float]:
        """
        Estimate how many words can be extended and maximum extension.
        
        Returns:
            (count_extensible_words, max_total_extension_seconds)
        """
        extensible = [w for w in word_segments if w.is_extensible]
        count = len(extensible)
        max_ext = count * (self.max_extension_per_word_ms / 1000.0)
        return count, max_ext
