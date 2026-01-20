"""Time-stretching strategy using professional WSOLA algorithm from pytsmod."""

import numpy as np
from typing import Optional


class TimeStretcher:
    """
    Time-stretch audio without changing pitch using WSOLA.
    
    Uses pytsmod library for high-quality, artifact-free time-stretching.
    """
    
    def __init__(
        self,
        max_stretch_factor: float = 1.50,  # Max 50% slower
        min_stretch_factor: float = 0.60,  # Max 40% faster
    ):
        """
        Initialize time stretcher.
        
        Args:
            max_stretch_factor: Maximum stretch factor (>1 = slower)
            min_stretch_factor: Minimum stretch factor (<1 = faster)
        """
        self.max_stretch_factor = max_stretch_factor
        self.min_stretch_factor = min_stretch_factor
    
    def stretch(
        self,
        audio: np.ndarray,
        sr: int,
        factor: float,
        method: str = 'wsola'
    ) -> np.ndarray:
        """
        Apply time-stretch to audio.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            factor: Stretch factor (1.0 = no change, 1.5 = 50% longer, 0.5 = 50% shorter)
            method: Algorithm to use ('wsola', 'ola', 'pv' for phase vocoder)
            
        Returns:
            Time-stretched audio
        """
        # Clamp factor to safe range
        factor = max(self.min_stretch_factor, min(self.max_stretch_factor, factor))
        
        if abs(factor - 1.0) < 0.001:
            return audio
        
        try:
            import pytsmod as tsm
            
            # Ensure float64 for pytsmod
            audio_f64 = audio.astype(np.float64)
            
            # pytsmod expects (channels, samples) format
            if audio_f64.ndim == 1:
                audio_f64 = audio_f64.reshape(1, -1)
            
            if method == 'wsola':
                # WSOLA - best for speech, minimal artifacts
                result = tsm.wsola(audio_f64, factor)
            elif method == 'ola':
                # OLA - simpler, faster, but more artifacts
                result = tsm.ola(audio_f64, factor)
            elif method == 'pv':
                # Phase vocoder - good for music, can sound metallic for speech
                result = tsm.phase_vocoder(audio_f64, factor)
            elif method == 'pv_int':
                # Phase vocoder with identity phase locking
                result = tsm.phase_vocoder_int(audio_f64, factor)
            else:
                # Default to WSOLA
                result = tsm.wsola(audio_f64, factor)
            
            # Convert back to original shape and dtype
            if result.ndim > 1:
                result = result.squeeze()
            
            return result.astype(audio.dtype)
            
        except ImportError:
            # Fallback to basic implementation if pytsmod not available
            return self._fallback_stretch(audio, sr, factor)
    
    def stretch_to_duration(
        self,
        audio: np.ndarray,
        sr: int,
        target_duration: float,
        method: str = 'wsola'
    ) -> np.ndarray:
        """
        Stretch audio to a specific duration.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            target_duration: Target duration in seconds
            method: Algorithm to use
            
        Returns:
            Time-stretched audio
        """
        current_duration = len(audio) / sr
        factor = target_duration / current_duration
        return self.stretch(audio, sr, factor, method)
    
    def _fallback_stretch(
        self,
        audio: np.ndarray,
        sr: int,
        factor: float
    ) -> np.ndarray:
        """
        Basic WSOLA implementation as fallback.
        Only used if pytsmod is not available.
        """
        # Frame parameters optimized for speech
        frame_length = int(0.050 * sr)  # 50ms frames
        synthesis_hop = int(0.0125 * sr)  # 12.5ms hop
        analysis_hop = int(synthesis_hop / factor)
        
        # Ensure valid parameters
        frame_length = max(frame_length, 64)
        synthesis_hop = max(synthesis_hop, 16)
        analysis_hop = max(analysis_hop, 8)
        
        # Calculate output length
        n_frames = max(1, (len(audio) - frame_length) // analysis_hop + 1)
        output_length = (n_frames - 1) * synthesis_hop + frame_length
        
        # Initialize output
        output = np.zeros(output_length, dtype=audio.dtype)
        window = np.hanning(frame_length).astype(audio.dtype)
        norm = np.zeros(output_length, dtype=audio.dtype)
        
        # Tolerance for finding optimal overlap
        tolerance = int(frame_length * 0.25)
        
        prev_analysis_pos = 0
        
        for i in range(n_frames):
            # Nominal analysis position
            analysis_pos = i * analysis_hop
            synthesis_pos = i * synthesis_hop
            
            # Find best match within tolerance (except first frame)
            if i > 0 and tolerance > 0:
                best_offset = self._find_best_offset(
                    audio, analysis_pos, frame_length,
                    output, synthesis_pos, tolerance
                )
                analysis_pos = max(0, min(len(audio) - frame_length, analysis_pos + best_offset))
            
            # Bounds check
            if analysis_pos + frame_length > len(audio):
                break
            
            # Extract and window frame
            frame = audio[analysis_pos:analysis_pos + frame_length] * window
            
            # Add to output with overlap-add
            end_pos = min(synthesis_pos + frame_length, output_length)
            frame_len = end_pos - synthesis_pos
            output[synthesis_pos:end_pos] += frame[:frame_len]
            norm[synthesis_pos:end_pos] += window[:frame_len]
        
        # Normalize
        norm[norm < 1e-8] = 1.0
        output /= norm
        
        return output
    
    def _find_best_offset(
        self,
        audio: np.ndarray,
        analysis_pos: int,
        frame_length: int,
        output: np.ndarray,
        synthesis_pos: int,
        tolerance: int
    ) -> int:
        """Find optimal offset for waveform similarity."""
        best_offset = 0
        best_similarity = -np.inf
        
        # Match against end of synthesized output
        match_length = min(32, synthesis_pos, frame_length // 4)
        
        if match_length < 8 or synthesis_pos < match_length:
            return 0
        
        ref_segment = output[synthesis_pos - match_length:synthesis_pos]
        
        for offset in range(-tolerance, tolerance + 1, 4):
            pos = analysis_pos + offset
            if pos < 0 or pos + match_length > len(audio):
                continue
            
            test_segment = audio[pos:pos + match_length]
            similarity = np.dot(ref_segment, test_segment)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_offset = offset
        
        return best_offset
