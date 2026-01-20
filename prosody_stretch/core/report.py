"""Adjustment report for tracking what was done."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AdjustmentReport:
    """Report of adjustments made to achieve target duration."""
    
    original_duration: float = 0.0
    target_duration: float = 0.0
    final_duration: float = 0.0
    
    # Quality metrics
    quality_score: float = 1.0  # 0-1, higher is better
    
    # Strategies used and their contributions
    strategies_used: List[str] = field(default_factory=list)
    adjustment_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Warnings/notes
    warnings: List[str] = field(default_factory=list)
    
    @property
    def total_adjustment(self) -> float:
        """Total duration change in seconds."""
        return self.final_duration - self.original_duration
    
    @property
    def adjustment_percent(self) -> float:
        """Adjustment as percentage of original."""
        if self.original_duration == 0:
            return 0.0
        return (self.total_adjustment / self.original_duration) * 100
    
    def add_strategy(self, name: str, amount: float):
        """Record a strategy's contribution."""
        if name not in self.strategies_used:
            self.strategies_used.append(name)
        self.adjustment_breakdown[name] = self.adjustment_breakdown.get(name, 0) + amount
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_duration": self.original_duration,
            "target_duration": self.target_duration,
            "final_duration": self.final_duration,
            "total_adjustment": self.total_adjustment,
            "adjustment_percent": self.adjustment_percent,
            "quality_score": self.quality_score,
            "strategies_used": self.strategies_used,
            "adjustment_breakdown": self.adjustment_breakdown,
            "warnings": self.warnings,
        }
    
    def __str__(self) -> str:
        lines = [
            f"Duration: {self.original_duration:.3f}s → {self.final_duration:.3f}s ({self.adjustment_percent:+.1f}%)",
            f"Quality: {self.quality_score:.2f}",
            f"Strategies: {', '.join(self.strategies_used)}",
        ]
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(lines)
