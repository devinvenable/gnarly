"""Headless batch processing runner."""

import cv2
import numpy as np
from tqdm import tqdm

from ..config import ProcessingConfig
from ..core.io import VideoReader, VideoWriter, is_video_file
from ..effects.ca_effect import CAEffect
from ..effects.deep_dream_effect import DeepDreamEffect
from ..effects.pipeline import EffectPipeline
from ..effects.zoom_effect import ZoomEffect


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
    """
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
    if config.deep_dream.enabled:
        effects.append(
            DeepDreamEffect(
                iterations=config.deep_dream.iterations,
                learning_rate=config.deep_dream.learning_rate,
                layers=config.deep_dream.layers,
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
