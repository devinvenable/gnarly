"""Headless batch processing runner."""

import sys

import cv2
import numpy as np
from tqdm import tqdm

from ..config import ProcessingConfig
from ..core.io import VideoReader, VideoWriter, is_video_file
from ..effects.ca_effect import CAEffect
from ..effects.deep_dream_effect import DeepDreamEffect
from ..effects.morph_effect import MorphEffect
from ..effects.pipeline import EffectPipeline
from ..effects.zoom_effect import ZoomEffect


def check_gpu_availability() -> bool:
    """Check if CUDA GPU is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def validate_gpu_features(config: ProcessingConfig) -> list[str]:
    """Check which GPU features are requested and validate availability.

    Args:
        config: Processing configuration.

    Returns:
        List of error messages for unavailable GPU features.
    """
    errors = []
    gpu_available = check_gpu_availability()

    if config.deep_dream.enabled and not gpu_available:
        errors.append("Deep Dream requires a CUDA-capable GPU (--deep-dream)")

    # Object detection (YOLO/EfficientDet) requires GPU
    # Morphing also implicitly requires detection (unless using --faces)
    needs_gpu_detection = (
        config.detection.enabled or
        (config.morph.enabled and not config.face.enabled)
    )
    if needs_gpu_detection and not gpu_available:
        errors.append(
            f"Object detection with {config.detection.model} requires a CUDA-capable GPU (--detect)"
        )

    if config.morph.enabled and not gpu_available:
        errors.append("Morphing effect (Stable Diffusion) requires a CUDA-capable GPU (--morph)")

    return errors


def create_detector(config: ProcessingConfig):
    """Create detector based on config.

    Detector is created if:
    - face detection is enabled (--faces)
    - object detection is enabled (--detect)
    - morphing is enabled (--morph) - requires detection to work

    Args:
        config: Processing configuration.

    Returns:
        Detector instance or None if detection not needed.
    """
    # Face detection takes priority if explicitly enabled
    if config.face.enabled:
        from ..detection.face import FaceDetector
        print("Initializing face detector...")
        return FaceDetector()

    # Check if detection is needed (either explicitly or for morphing)
    needs_detection = config.detection.enabled or config.morph.enabled

    if not needs_detection:
        return None

    if config.detection.model == "yolo":
        from ..detection.yolo import YOLODetector
        return YOLODetector.from_config(config.detection, device="cuda")
    elif config.detection.model == "efficientdet":
        from ..detection.efficientdet import EfficientDetDetector
        return EfficientDetDetector(
            confidence_threshold=config.detection.confidence_threshold,
            device="cuda",
        )
    else:
        raise ValueError(f"Unknown detection model: {config.detection.model}")


def prepare_frame(frame: np.ndarray, max_dim: int, grid_scale: int) -> np.ndarray:
    """Resize and pad frame for processing.

    Args:
        frame: Input frame.
        max_dim: Maximum dimension.
        grid_scale: Grid scale for padding.

    Returns:
        Prepared frame.
    """
    height, width = frame.shape[:2]

    # Resize if needed
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]

    # Pad to grid scale multiple
    pad_h = (grid_scale - height % grid_scale) % grid_scale
    pad_w = (grid_scale - width % grid_scale) % grid_scale

    if pad_h or pad_w:
        frame = cv2.copyMakeBorder(
            frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
        )

    return frame


def run_headless(config: ProcessingConfig) -> None:
    """Run headless batch processing.

    Args:
        config: Processing configuration.

    Raises:
        SystemExit: If GPU features are requested but unavailable.
    """
    # Validate GPU features before proceeding
    gpu_errors = validate_gpu_features(config)
    if gpu_errors:
        print("GPU features requested but unavailable:", file=sys.stderr)
        for error in gpu_errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nTo fix this, either:", file=sys.stderr)
        print("  1. Run on a system with a CUDA-capable GPU", file=sys.stderr)
        print("  2. Disable GPU features (remove --deep-dream, --detect, --morph)", file=sys.stderr)
        sys.exit(1)

    # Create detector if needed
    detector = create_detector(config)

    # Print enabled features
    features = []
    if config.zoom.enabled:
        features.append("zoom")
    if config.deep_dream.enabled:
        features.append("deep-dream")
    if config.face.enabled:
        features.append("face-detection")
    elif config.detection.enabled or (config.morph.enabled and detector):
        features.append(f"detection ({config.detection.model})")
    if config.morph.enabled:
        features.append(f"morph ({config.morph.style})")
    if features:
        print(f"Enabled features: {', '.join(features)}")

    # Read input
    with VideoReader(config.input_path) as reader:
        # Get first frame to determine dimensions
        ret, first_frame = reader.read_frame()
        if not ret:
            raise RuntimeError(f"Cannot read from {config.input_path}")

        # Prepare first frame
        first_frame = prepare_frame(
            first_frame, config.max_dimension, config.ca.grid_scale
        )
        height, width = first_frame.shape[:2]

        # Determine frame count
        if is_video_file(config.input_path):
            total_frames = reader.frame_count
            fps = reader.fps
        else:
            total_frames = config.output.frames
            fps = config.output.fps

    # Build effect pipeline
    effects = [
        CAEffect(
            width=width,
            height=height,
            grid_scale=config.ca.grid_scale,
            rule=config.ca.rule,
            divisor=config.ca.divisor,
            blend_alpha=config.effect.blend_alpha,
        )
    ]
    if config.zoom.enabled:
        effects.append(
            ZoomEffect(
                speed=config.zoom.speed,
                min_zoom=config.zoom.min_zoom,
                max_zoom=config.zoom.max_zoom,
            )
        )
    deep_dream_effect = None
    if config.deep_dream.enabled:
        deep_dream_effect = DeepDreamEffect(
            iterations=config.deep_dream.iterations,
            learning_rate=config.deep_dream.learning_rate,
            layers=config.deep_dream.layers,
        )
        effects.append(deep_dream_effect)
    if config.morph.enabled:
        effects.append(
            MorphEffect(
                detector=detector,
                detection_interval=config.detection.interval,
                morph_frames=config.morph.frames,
                morph_min_blend=config.morph.min_blend,
                morph_max_blend=config.morph.max_blend,
                max_objects_per_frame=config.detection.max_objects,
                preferred_style=config.morph.style,
                creativity=config.morph.creativity,
                deep_dream_effect=deep_dream_effect,
            )
        )

    effect = EffectPipeline(effects)

    # Process frames
    with VideoWriter(config.output_path, width, height, fps) as writer:
        with VideoReader(config.input_path) as reader:
            # For images, we generate multiple frames from the same image
            if is_video_file(config.input_path):
                # Process video frames
                progress = tqdm(total=total_frames, desc="Processing")
                for frame in reader:
                    frame = prepare_frame(
                        frame, config.max_dimension, config.ca.grid_scale
                    )
                    # Ensure consistent size
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))

                    output = effect.apply(frame)
                    writer.write_frame(output)
                    progress.update(1)
                progress.close()
            else:
                # Process single image multiple times
                ret, frame = reader.read_frame()
                frame = prepare_frame(
                    frame, config.max_dimension, config.ca.grid_scale
                )

                for _ in tqdm(range(total_frames), desc="Processing"):
                    output = effect.apply(frame)
                    writer.write_frame(output)

    print(f"Output saved to: {config.output_path}")
