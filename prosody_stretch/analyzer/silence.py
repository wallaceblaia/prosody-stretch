"""Silence detection module."""

import numpy as np
from typing import List, Tuple
from ..core.segment import SilenceSegment


class SilenceDetector:
    """Detect silence regions in audio."""
    
    def __init__(
        self,
        min_silence_ms: float = 100,
        silence_thresh_db: float = -40,
        min_speech_ms: float = 50,
    ):
        """
        Initialize silence detector.
        
        Args:
            min_silence_ms: Minimum silence duration to detect (ms)
            silence_thresh_db: Threshold below which is considered silence (dB)
            min_speech_ms: Minimum speech duration between silences (ms)
        """
        self.min_silence_ms = min_silence_ms
        self.silence_thresh_db = silence_thresh_db
        self.min_speech_ms = min_speech_ms
    
    def detect(
        self, 
        audio: np.ndarray, 
        sr: int,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> List[SilenceSegment]:
        """
        Detect silence regions in audio.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            frame_length: Frame length for RMS calculation
            hop_length: Hop length between frames
            
        Returns:
            List of SilenceSegment objects
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Calculate RMS energy per frame
        rms = self._calculate_rms(audio, frame_length, hop_length)
        
        # Convert to dB
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Find silence frames
        is_silence = rms_db < self.silence_thresh_db
        
        # Convert to time segments
        segments = self._frames_to_segments(
            is_silence, sr, hop_length
        )
        
        # Filter by minimum duration
        min_dur = self.min_silence_ms / 1000.0
        segments = [s for s in segments if s.duration >= min_dur]
        
        return segments
    
    def _calculate_rms(
        self, 
        audio: np.ndarray, 
        frame_length: int, 
        hop_length: int
    ) -> np.ndarray:
        """Calculate RMS energy per frame."""
        # Pad audio
        pad_length = frame_length // 2
        audio_padded = np.pad(audio, (pad_length, pad_length), mode='constant')
        
        # Calculate number of frames
        n_frames = 1 + (len(audio_padded) - frame_length) // hop_length
        
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio_padded[start:start + frame_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))
        
        return rms
    
    def _frames_to_segments(
        self, 
        is_silence: np.ndarray, 
        sr: int, 
        hop_length: int
    ) -> List[SilenceSegment]:
        """Convert boolean frame array to time segments."""
        segments = []
        
        # Find transitions
        diff = np.diff(is_silence.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if is_silence[0]:
            starts = np.insert(starts, 0, 0)
        if is_silence[-1]:
            ends = np.append(ends, len(is_silence))
        
        # Create segments
        frame_duration = hop_length / sr
        for start, end in zip(starts, ends):
            start_time = start * frame_duration
            end_time = end * frame_duration
            segments.append(SilenceSegment(
                start_time=start_time,
                end_time=end_time,
            ))
        
        return segments
    
    def get_speech_segments(
        self, 
        audio: np.ndarray, 
        sr: int,
        silence_segments: List[SilenceSegment] = None
    ) -> List[Tuple[float, float]]:
        """
        Get speech (non-silence) segments.
        
        Returns:
            List of (start_time, end_time) tuples for speech regions
        """
        if silence_segments is None:
            silence_segments = self.detect(audio, sr)
        
        duration = len(audio) / sr
        speech_segments = []
        
        # Start from 0
        current_time = 0.0
        
        for silence in sorted(silence_segments, key=lambda s: s.start_time):
            if silence.start_time > current_time:
                speech_segments.append((current_time, silence.start_time))
            current_time = silence.end_time
        
        # Add final segment if needed
        if current_time < duration:
            speech_segments.append((current_time, duration))
        
        # Filter by minimum duration
        min_dur = self.min_speech_ms / 1000.0
        speech_segments = [s for s in speech_segments if s[1] - s[0] >= min_dur]
        
        return speech_segments
