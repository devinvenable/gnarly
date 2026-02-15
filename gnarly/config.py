"""Configuration dataclasses for gnarly."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CAConfig:
    """Configuration for cellular automata."""

    rule: str = "Conway"
    grid_scale: int = 4
    divisor: int = 2


@dataclass
class OutputConfig:
    """Configuration for output video."""

    fps: int = 15
    frames: int = 300


@dataclass
class EffectConfig:
    """Configuration for effects."""

    blend_alpha: float = 0.3


@dataclass
class ProcessingConfig:
    """Combined configuration for processing."""

    input_path: str
    output_path: str
    ca: CAConfig
    output: OutputConfig
    effect: EffectConfig
    max_dimension: int = 800

    @classmethod
    def from_args(
        cls,
        input_path: str,
        output_path: str,
        rule: str = "Conway",
        grid_scale: int = 4,
        divisor: int = 2,
        blend_alpha: float = 0.3,
        fps: int = 15,
        frames: int = 300,
        max_dimension: int = 800,
    ) -> "ProcessingConfig":
        """Create ProcessingConfig from CLI arguments."""
        return cls(
            input_path=input_path,
            output_path=output_path,
            ca=CAConfig(rule=rule, grid_scale=grid_scale, divisor=divisor),
            output=OutputConfig(fps=fps, frames=frames),
            effect=EffectConfig(blend_alpha=blend_alpha),
            max_dimension=max_dimension,
        )
