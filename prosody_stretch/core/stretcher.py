"""Main ProsodyStretcher API - the public interface."""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List

from .segment import WordSegment, SilenceSegment
from .planner import DurationPlanner, PlannerConfig
from .report import AdjustmentReport
from ..analyzer.silence import SilenceDetector
from ..analyzer.audio import AudioAnalyzer
from ..strategies.pause import PauseManipulator
from ..strategies.stretch import TimeStretcher
from ..strategies.vowel import VowelExtender


class ProsodyStretcher:
    """
    Main API for natural audio duration adjustment.
    
    This class orchestrates the analysis, planning, and execution
    of duration adjustments while maintaining audio naturalness.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.6,
        prefer_pauses: bool = True,
        sample_rate: int = 22050,
        max_compression_ratio: float = 0.25,
        max_extension_ratio: float = 0.45,
    ):
        """
        Initialize the prosody stretcher.
        
        Args:
            quality_threshold: Minimum quality score (0-1) to proceed
            prefer_pauses: Prioritize pause manipulation over stretching
            sample_rate: Target sample rate for processing
        """
        self.quality_threshold = quality_threshold
        self.prefer_pauses = prefer_pauses
        self.sample_rate = sample_rate
        self.max_compression_ratio = max_compression_ratio
        self.max_extension_ratio = max_extension_ratio
        
        # Initialize components
        self.silence_detector = SilenceDetector(
            min_silence_ms=30,
            silence_thresh_db=-30,
        )
        
        # Use default config - it now maximizes all strategies to reach target
        self.planner = DurationPlanner()
        
        self.pause_manipulator = PauseManipulator()
        self.time_stretcher = TimeStretcher()
        self.vowel_extender = VowelExtender()
    
    def adjust_duration(
        self,
        audio_path: Union[str, Path],
        target_duration: float,
        text: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[np.ndarray, AdjustmentReport]:
        """
        Adjust audio duration to target value.
        
        Args:
            audio_path: Path to input audio file
            target_duration: Target duration in seconds
            text: Optional transcription text (improves quality)
            output_path: Optional path to save adjusted audio
            
        Returns:
            (adjusted_audio, adjustment_report)
        """
        # Load audio
        audio, sr = AudioAnalyzer.load(audio_path, sr=self.sample_rate)
        current_duration = AudioAnalyzer.get_duration(audio, sr)
        
        # Process
        result, report = self._process(
            audio, sr, current_duration, target_duration, text
        )
        
        # Save if output path provided
        if output_path:
            AudioAnalyzer.save(result, output_path, sr)
        
        return result, report
    
    def adjust_duration_array(
        self,
        audio: np.ndarray,
        sr: int,
        target_duration: float,
        text: Optional[str] = None
    ) -> Tuple[np.ndarray, AdjustmentReport]:
        """
        Adjust duration of audio array directly.
        
        Args:
            audio: Input audio as numpy array
            sr: Sample rate
            target_duration: Target duration in seconds
            text: Optional transcription text
            
        Returns:
            (adjusted_audio, adjustment_report)
        """
        current_duration = AudioAnalyzer.get_duration(audio, sr)
        return self._process(audio, sr, current_duration, target_duration, text)
    
    def match_duration(
        self,
        source_audio: Union[str, Path, np.ndarray],
        reference_audio: Union[str, Path, np.ndarray],
        source_sr: Optional[int] = None,
        reference_sr: Optional[int] = None,
        text: Optional[str] = None
    ) -> Tuple[np.ndarray, AdjustmentReport]:
        """
        Adjust source audio to match reference audio duration.
        
        Useful for dubbing synchronization where you want the
        dubbed audio to match the original audio timing.
        
        Args:
            source_audio: Audio to adjust (path or array)
            reference_audio: Reference for target duration (path or array)
            source_sr: Sample rate if source is array
            reference_sr: Sample rate if reference is array
            text: Optional transcription text
            
        Returns:
            (adjusted_audio, adjustment_report)
        """
        # Load source
        if isinstance(source_audio, (str, Path)):
            source, sr = AudioAnalyzer.load(source_audio, sr=self.sample_rate)
        else:
            source = source_audio
            sr = source_sr or self.sample_rate
        
        # Get reference duration
        if isinstance(reference_audio, (str, Path)):
            ref, _ = AudioAnalyzer.load(reference_audio)
            ref_sr = _
        else:
            ref = reference_audio
            ref_sr = reference_sr or self.sample_rate
        
        target_duration = AudioAnalyzer.get_duration(ref, ref_sr)
        current_duration = AudioAnalyzer.get_duration(source, sr)
        
        return self._process(source, sr, current_duration, target_duration, text)
    
    def _process(
        self,
        audio: np.ndarray,
        sr: int,
        current_duration: float,
        target_duration: float,
        text: Optional[str] = None
    ) -> Tuple[np.ndarray, AdjustmentReport]:
        """Internal processing pipeline."""
        
        report = AdjustmentReport(
            original_duration=current_duration,
            target_duration=target_duration,
        )
        
        # Quick check - if already close enough, return as-is
        delta = abs(target_duration - current_duration)
        if delta < 0.02:  # Within 20ms
            report.final_duration = current_duration
            report.quality_score = 1.0
            return audio, report
        
        # Estimate quality
        quality, desc = self.planner.estimate_quality(current_duration, target_duration)
        
        if quality < self.quality_threshold:
            report.add_warning(
                f"Low quality expected ({quality:.2f}): {desc}. "
                "Consider regenerating TTS with adjusted speed."
            )

        # For larger extensions, use conservative pause extension first,
        # then a smaller full-audio stretch to reduce metallic artifacts.
        extension_ratio = target_duration / current_duration
        if target_duration > current_duration and extension_ratio >= 1.10:
            conservative_detector = SilenceDetector(
                min_silence_ms=120,
                silence_thresh_db=-40,
                min_speech_ms=80,
            )
            silences = conservative_detector.detect(audio, sr)
            long_silences = [s for s in silences if s.duration >= 0.12]
            remaining = target_duration - current_duration
            result = audio

            if long_silences and remaining > 0:
                pause_available = sum(s.extensible_amount for s in long_silences)
                pause_use = min(pause_available, remaining)
                if pause_use > 0:
                    result, actual = self.pause_manipulator.extend_pauses(
                        result, sr, long_silences, pause_use
                    )
                    report.add_strategy('pause', actual)
                    remaining -= actual

            if remaining > 0.01:
                result = self.time_stretcher.stretch_to_duration(
                    result, sr, target_duration, method="wsola"
                )
                report.add_strategy('wsola', target_duration - current_duration)

            report.final_duration = AudioAnalyzer.get_duration(result, sr)
            report.quality_score = quality
            return result, report
        
        # Clamp excessive compression to avoid metallic artifacts
        min_duration = current_duration * (1 - self.max_compression_ratio)
        if target_duration < min_duration:
            report.add_warning(
                f"Target too short; clamped to {min_duration:.3f}s "
                f"(-{self.max_compression_ratio*100:.0f}%)."
            )
            target_duration = min_duration

        # Clamp excessive extension to avoid artifacts
        max_duration = current_duration * (1 + self.max_extension_ratio)
        if target_duration > max_duration:
            report.add_warning(
                f"Target too long; clamped to {max_duration:.3f}s "
                f"(+{self.max_extension_ratio*100:.0f}%)."
            )
            target_duration = max_duration

        # Analyze audio
        silences = self.silence_detector.detect(audio, sr)
        speech_regions = self.silence_detector.get_speech_segments(audio, sr, silences)

        # For moderate extensions, only use long pauses to avoid micro-cuts,
        # and rely on speech-only stretching for the rest.
        silences_for_plan = silences
        if target_duration > current_duration:
            extension_ratio = target_duration / current_duration
            if 1.10 <= extension_ratio < 1.30:
                conservative_detector = SilenceDetector(
                    min_silence_ms=120,
                    silence_thresh_db=-40,
                    min_speech_ms=80,
                )
                silences_for_plan = conservative_detector.detect(audio, sr)
        
        # Create word segments from text if available
        word_segments = self._create_word_segments(text, speech_regions) if text else None
        
        # Plan the adjustment
        plan = self.planner.plan(
            current_duration=current_duration,
            target_duration=target_duration,
            silences=silences_for_plan,
            word_segments=word_segments,
            speech_regions=speech_regions,
        )
        
        # Execute the plan
        result = self._execute_plan(audio, sr, plan, silences, word_segments, report)
        
        # Update report
        report.final_duration = AudioAnalyzer.get_duration(result, sr)
        report.quality_score = plan.estimated_quality
        
        return result, report
    
    def _execute_plan(
        self,
        audio: np.ndarray,
        sr: int,
        plan,  # AdjustmentPlan
        silences: List[SilenceSegment],
        word_segments: Optional[List[WordSegment]],
        report: AdjustmentReport
    ) -> np.ndarray:
        """Execute the adjustment plan."""
        
        result = audio.copy()
        
        # Group operations by strategy
        pause_ops = [op for op in plan.operations if op.strategy == 'pause']
        stretch_ops = [op for op in plan.operations if op.strategy == 'wsola']
        vowel_ops = [op for op in plan.operations if op.strategy == 'vowel']
        
        # Apply pause adjustments first (they're the most natural)
        if pause_ops:
            total_pause_adj = sum(op.amount for op in pause_ops)
            
            if total_pause_adj > 0:
                result, actual = self.pause_manipulator.extend_pauses(
                    result, sr, silences, total_pause_adj
                )
            else:
                result, actual = self.pause_manipulator.compress_pauses(
                    result, sr, silences, abs(total_pause_adj)
                )
                actual = -actual
            
            report.add_strategy('pause', actual)
        
        # Apply time-stretching
        if stretch_ops:
            # Calculate average factor
            factors = [op.factor for op in stretch_ops]
            avg_factor = sum(factors) / len(factors)
            
            # For compression, stretch full audio to avoid segment boundary artifacts.
            # For extension, stretch only speech to avoid amplifying added pauses.
            before = result
            if avg_factor < 1.0:
                # Use phase vocoder with phase locking for heavier compression
                method = 'pv_int' if avg_factor < 0.81 else 'wsola'
                result = self.time_stretcher.stretch(result, sr, avg_factor, method=method)
            else:
                # For large extensions, stretch full audio to avoid word cuts
                if avg_factor >= 1.35:
                    result = self.time_stretcher.stretch(result, sr, avg_factor)
                else:
                    result = self._stretch_speech_regions(result, sr, avg_factor)
            
            stretch_adj = (
                AudioAnalyzer.get_duration(result, sr)
                - AudioAnalyzer.get_duration(before, sr)
            )
            report.add_strategy('wsola', stretch_adj)
        
        # Apply vowel extension
        if vowel_ops and word_segments:
            total_vowel_ext = sum(op.amount for op in vowel_ops if op.amount > 0)
            
            if total_vowel_ext > 0:
                result, actual = self.vowel_extender.extend_endings(
                    result, sr, word_segments, total_vowel_ext
                )
                report.add_strategy('vowel', actual)
        
        return result

    def _stretch_speech_regions(
        self,
        audio: np.ndarray,
        sr: int,
        factor: float
    ) -> np.ndarray:
        """
        Stretch only speech regions, leaving detected silences untouched.
        
        This prevents pause manipulation from being amplified by stretching.
        """
        if abs(factor - 1.0) < 0.001:
            return audio
        
        silences = self.silence_detector.detect(audio, sr)
        speech_regions = self.silence_detector.get_speech_segments(
            audio, sr, silences
        )
        
        if not speech_regions:
            return self.time_stretcher.stretch(audio, sr, factor)
        
        pieces = []
        cursor = 0
        
        for start, end in speech_regions:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample > cursor:
                pieces.append(audio[cursor:start_sample])
            
            speech = audio[start_sample:end_sample]
            stretched = self.time_stretcher.stretch(speech, sr, factor)
            pieces.append(stretched)
            cursor = end_sample
        
        if cursor < len(audio):
            pieces.append(audio[cursor:])
        
        if not pieces:
            return audio
        
        return np.concatenate(pieces)
    
    def _create_word_segments(
        self,
        text: str,
        speech_regions: List[Tuple[float, float]]
    ) -> List[WordSegment]:
        """
        Create approximate word segments from text and speech regions.
        
        This is a simple heuristic when forced alignment is not available.
        """
        words = text.split()
        if not words or not speech_regions:
            return []
        
        # Calculate total speech duration
        total_speech = sum(end - start for start, end in speech_regions)
        
        # Estimate duration per word (rough approximation)
        avg_word_duration = total_speech / len(words)
        
        segments = []
        current_time = speech_regions[0][0] if speech_regions else 0
        
        for word in words:
            # Adjust duration by word length (longer words = more time)
            word_factor = len(word) / 5.0  # 5 chars = average
            word_factor = max(0.5, min(2.0, word_factor))
            duration = avg_word_duration * word_factor
            
            segments.append(WordSegment(
                start_time=current_time,
                end_time=current_time + duration,
                word=word,
            ))
            current_time += duration
        
        return segments
