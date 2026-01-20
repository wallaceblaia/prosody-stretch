"""Duration planner - the brain of prosody-stretch."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .segment import (
    Segment, WordSegment, SilenceSegment,
    AdjustmentOperation, AdjustmentPlan
)


@dataclass
class PlannerConfig:
    """Configuration for the duration planner."""
    
    # Priority order: pause (most natural) -> stretch -> vowel (least natural)
    # These are now preferences, not hard limits
    
    # Limits per strategy
    max_pause_extension: float = 1.0     # 1000ms max per pause
    max_stretch_factor: float = 1.40     # 40% stretch max
    min_stretch_factor: float = 0.60     # 40% compress max
    max_vowel_extension_ms: float = 100  # 100ms per word
    
    # Quality thresholds
    warning_threshold: float = 0.30  # Warn if adjustment > 30%
    limit_threshold: float = 0.60    # Hard limit at 60%


class DurationPlanner:
    """
    Plan how to adjust audio duration to reach target.
    
    Uses all available strategies to reach exact target duration,
    prioritizing natural adjustments (pauses) first.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """Initialize planner with configuration."""
        self.config = config or PlannerConfig()
    
    def plan(
        self,
        current_duration: float,
        target_duration: float,
        silences: List[SilenceSegment],
        word_segments: Optional[List[WordSegment]] = None,
        speech_regions: Optional[List[Tuple[float, float]]] = None
    ) -> AdjustmentPlan:
        """
        Create an adjustment plan to reach target duration.
        
        The planner will use ALL available strategies to reach
        the exact target duration, prioritizing natural adjustments.
        """
        delta = target_duration - current_duration
        
        if abs(delta) < 0.01:  # Less than 10ms difference
            return AdjustmentPlan(estimated_quality=1.0)
        
        plan = AdjustmentPlan()
        
        # Calculate quality based on adjustment ratio
        ratio = abs(delta) / current_duration
        if ratio > self.config.limit_threshold:
            plan.estimated_quality = 0.4
        elif ratio > self.config.warning_threshold:
            plan.estimated_quality = 0.7
        else:
            plan.estimated_quality = 1.0 - (ratio * 0.5)
        
        if delta > 0:
            plan = self._plan_extension(
                delta, silences, word_segments, speech_regions, plan, current_duration
            )
        else:
            plan = self._plan_compression(
                abs(delta), silences, speech_regions, plan, current_duration
            )
        
        return plan
    
    def _plan_extension(
        self,
        total_extension: float,
        silences: List[SilenceSegment],
        word_segments: Optional[List[WordSegment]],
        speech_regions: Optional[List[Tuple[float, float]]],
        plan: AdjustmentPlan,
        current_duration: float
    ) -> AdjustmentPlan:
        """Plan extension using ALL available capacity to reach target."""
        remaining = total_extension
        
        # 1. PAUSE EXTENSION (most natural)
        if silences and remaining > 0:
            # Calculate max available from pauses
            pause_available = sum(
                min(s.extensible_amount, self.config.max_pause_extension)
                for s in silences
            )
            
            # Use as much as needed from pauses
            pause_use = min(pause_available, remaining)
            
            if pause_use > 0 and pause_available > 0:
                for s in silences:
                    if remaining <= 0.001:
                        break
                    
                    max_for_this = min(s.extensible_amount, self.config.max_pause_extension)
                    portion = (max_for_this / pause_available) * pause_use
                    portion = min(portion, max_for_this, remaining)
                    
                    if portion > 0.001:
                        plan.add_operation(AdjustmentOperation(
                            segment=s,
                            strategy='pause',
                            amount=portion,
                        ))
                        remaining -= portion
        
        # 2. TIME-STRETCH (still natural with pytsmod)
        if remaining > 0.001 and speech_regions:
            total_speech = sum(end - start for start, end in speech_regions)
            
            if total_speech > 0:
                # Calculate factor needed
                factor = 1 + (remaining / total_speech)
                factor = min(factor, self.config.max_stretch_factor)
                
                # Actual extension we can achieve
                actual_extension = total_speech * (factor - 1)
                
                for start, end in speech_regions:
                    duration = end - start
                    segment = Segment(start_time=start, end_time=end)
                    extension = duration * (factor - 1)
                    
                    plan.add_operation(AdjustmentOperation(
                        segment=segment,
                        strategy='wsola',
                        amount=extension,
                        factor=factor,
                    ))
                
                remaining -= actual_extension
        
        # 3. VOWEL EXTENSION (subtle, for fine-tuning)
        if remaining > 0.001 and word_segments:
            extensible = [w for w in word_segments if w.is_extensible]
            
            if extensible:
                max_per_word = self.config.max_vowel_extension_ms / 1000
                vowel_available = len(extensible) * max_per_word
                vowel_use = min(vowel_available, remaining)
                
                per_word = vowel_use / len(extensible)
                per_word = min(per_word, max_per_word)
                
                for word in extensible:
                    if remaining <= 0.001:
                        break
                    
                    ext = min(per_word, remaining)
                    if ext > 0.001:
                        plan.add_operation(AdjustmentOperation(
                            segment=word,
                            strategy='vowel',
                            amount=ext,
                        ))
                        remaining -= ext
        
        return plan
    
    def _plan_compression(
        self,
        total_compression: float,
        silences: List[SilenceSegment],
        speech_regions: Optional[List[Tuple[float, float]]],
        plan: AdjustmentPlan,
        current_duration: float
    ) -> AdjustmentPlan:
        """Plan compression using ALL available capacity to reach target."""
        remaining = total_compression
        
        # 1. PAUSE COMPRESSION (most natural)
        if silences and remaining > 0:
            pause_available = sum(s.compressible_amount for s in silences)
            pause_use = min(pause_available, remaining)
            
            if pause_use > 0 and pause_available > 0:
                for s in silences:
                    if remaining <= 0.001:
                        break
                    
                    portion = (s.compressible_amount / pause_available) * pause_use
                    portion = min(portion, s.compressible_amount, remaining)
                    
                    if portion > 0.001:
                        plan.add_operation(AdjustmentOperation(
                            segment=s,
                            strategy='pause',
                            amount=-portion,
                        ))
                        remaining -= portion
        
        # 2. TIME-COMPRESS (speed up speech)
        if remaining > 0.001 and speech_regions:
            total_speech = sum(end - start for start, end in speech_regions)
            
            if total_speech > 0:
                # Calculate factor needed
                factor = 1 - (remaining / total_speech)
                factor = max(factor, self.config.min_stretch_factor)
                
                # Actual compression we can achieve
                actual_compression = total_speech * (1 - factor)
                
                for start, end in speech_regions:
                    duration = end - start
                    segment = Segment(start_time=start, end_time=end)
                    compression = duration * (1 - factor)
                    
                    plan.add_operation(AdjustmentOperation(
                        segment=segment,
                        strategy='wsola',
                        amount=-compression,
                        factor=factor,
                    ))
                
                remaining -= actual_compression
        
        return plan
    
    def estimate_quality(
        self,
        current_duration: float,
        target_duration: float
    ) -> Tuple[float, str]:
        """Estimate quality of adjustment."""
        ratio = abs(target_duration - current_duration) / current_duration
        
        if ratio < 0.10:
            return 0.95, "Excellent - minimal adjustment"
        elif ratio < 0.20:
            return 0.85, "Good - noticeable but natural"
        elif ratio < 0.30:
            return 0.70, "Acceptable - some artifacts possible"
        elif ratio < 0.45:
            return 0.55, "Marginal - consider regenerating TTS"
        else:
            return 0.40, "Poor - regenerate with different speed settings"
