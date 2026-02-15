"""Configuration dataclasses for gnarly."""

from dataclasses import dataclass, field
from typing import List, Optional


DREAM_LAYERS = [
    "Mixed_7a.branch3x3_2a.conv",
    "Mixed_7b.branch3x3_2b.conv",
    "Mixed_7c.branch3x3_2c.conv",
]


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
class ZoomConfig:
    """Configuration for continuous zoom effect."""

    enabled: bool = False
    speed: float = 1.02
    min_zoom: float = 1.0
    max_zoom: float = 2.0


@dataclass
class DeepDreamConfig:
    """Configuration for Deep Dream effect."""

    enabled: bool = False
    iterations: int = 30
    learning_rate: float = 0.02
    layers: List[str] = field(default_factory=lambda: list(DREAM_LAYERS))


@dataclass
class DetectionConfig:
    """Configuration for object detection."""

    enabled: bool = False
    model: str = "yolo"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    max_objects: int = 10
    interval: int = 30


@dataclass
class MorphConfig:
    """Configuration for object morphing."""

    enabled: bool = False
    style: str = "random"
    creativity: float = 0.5
    frames: int = 30
    min_blend: float = 0.0
    max_blend: float = 1.0


@dataclass
class FaceConfig:
    """Configuration for face detection."""

    enabled: bool = False


@dataclass
class ProcessingConfig:
    """Combined configuration for processing."""

    input_path: str
    output_path: str
    ca: CAConfig
    output: OutputConfig
    effect: EffectConfig
    zoom: ZoomConfig
    deep_dream: DeepDreamConfig
    detection: DetectionConfig
    morph: MorphConfig
    face: FaceConfig
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
        # Zoom config
        zoom_enabled: bool = False,
        zoom_speed: float = 1.02,
        zoom_min: float = 1.0,
        zoom_max: float = 2.0,
        # Deep Dream config
        deep_dream_enabled: bool = False,
        deep_dream_iterations: int = 30,
        deep_dream_lr: float = 0.02,
        deep_dream_layers: Optional[List[str]] = None,
        # Detection config
        detection_enabled: bool = False,
        detection_model: str = "yolo",
        detection_confidence: float = 0.5,
        detection_iou: float = 0.4,
        detection_max_objects: int = 10,
        detection_interval: int = 30,
        # Morph config
        morph_enabled: bool = False,
        morph_style: str = "random",
        morph_creativity: float = 0.5,
        morph_frames: int = 30,
        morph_min_blend: float = 0.0,
        morph_max_blend: float = 1.0,
        # Face config
        face_enabled: bool = False,
    ) -> "ProcessingConfig":
        """Create ProcessingConfig from CLI arguments."""
        return cls(
            input_path=input_path,
            output_path=output_path,
            ca=CAConfig(rule=rule, grid_scale=grid_scale, divisor=divisor),
            output=OutputConfig(fps=fps, frames=frames),
            effect=EffectConfig(blend_alpha=blend_alpha),
            zoom=ZoomConfig(
                enabled=zoom_enabled,
                speed=zoom_speed,
                min_zoom=zoom_min,
                max_zoom=zoom_max,
            ),
            deep_dream=DeepDreamConfig(
                enabled=deep_dream_enabled,
                iterations=deep_dream_iterations,
                learning_rate=deep_dream_lr,
                layers=deep_dream_layers or list(DREAM_LAYERS),
            ),
            detection=DetectionConfig(
                enabled=detection_enabled,
                model=detection_model,
                confidence_threshold=detection_confidence,
                iou_threshold=detection_iou,
                max_objects=detection_max_objects,
                interval=detection_interval,
            ),
            morph=MorphConfig(
                enabled=morph_enabled,
                style=morph_style,
                creativity=morph_creativity,
                frames=morph_frames,
                min_blend=morph_min_blend,
                max_blend=morph_max_blend,
            ),
            face=FaceConfig(enabled=face_enabled),
            max_dimension=max_dimension,
        )
