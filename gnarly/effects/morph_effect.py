"""Object morphing effect using detector boxes and Stable Diffusion."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

from ..detection.base import Detection, Detector
from .base import Effect

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency path
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from diffusers import StableDiffusionPipeline
except ImportError as exc:  # pragma: no cover - optional dependency path
    StableDiffusionPipeline = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None

if TYPE_CHECKING:
    from .deep_dream_effect import DeepDreamEffect


ARTISTIC_STYLES = [
    {
        "name": "photorealistic",
        "prompts": [
            "a photorealistic {obj} with intricate details",
            "a high-resolution photograph of a {obj}",
            "a detailed realistic {obj} in natural lighting",
        ],
        "guidance_scale": 7.0,
    },
    {
        "name": "surrealistic",
        "prompts": [
            "a surrealistic {obj} in the style of Salvador Dali",
            "a dreamlike surreal {obj} with impossible physics",
            "a melting surrealistic {obj} in a bizarre landscape",
        ],
        "guidance_scale": 9.0,
    },
    {
        "name": "impressionist",
        "prompts": [
            "an impressionist painting of a {obj} in the style of Monet",
            "a vibrant impressionist {obj} with visible brushstrokes",
            "a light-filled impressionist scene featuring a {obj}",
        ],
        "guidance_scale": 8.0,
    },
    {
        "name": "cyberpunk",
        "prompts": [
            "a cyberpunk {obj} with neon lights and chrome",
            "a futuristic {obj} in a high-tech cyberpunk setting",
            "a {obj} reimagined as a cyber-enhanced device",
        ],
        "guidance_scale": 8.5,
    },
    {
        "name": "abstract",
        "prompts": [
            "an abstract interpretation of a {obj} with geometric shapes",
            "a Kandinsky-style abstract painting of a {obj}",
            "a colorful abstract {obj} with bold forms",
        ],
        "guidance_scale": 9.0,
    },
]

STYLE_NAMES = {style["name"] for style in ARTISTIC_STYLES}


@dataclass
class MorphState:
    """Tracks one active morph region across frames."""

    obj_id: str
    region: tuple[int, int, int, int]
    generated_image: Image.Image
    frame_start: int
    frame_count: int = 0
    frames: int = 60
    min_blend: float = 0.1
    max_blend: float = 0.7

    def get_blend_factor(self) -> float:
        progress = min(1.0, self.frame_count / max(self.frames, 1))
        blend = self.min_blend + (self.max_blend - self.min_blend) * progress
        return float(np.clip(blend, 0.0, 1.0))

    def update(self) -> bool:
        self.frame_count += 1
        return self.frame_count < self.frames


def load_stable_diffusion_pipeline(device: str | None = None):
    """Load Stable Diffusion pipeline for morph image generation."""
    if StableDiffusionPipeline is None:
        raise RuntimeError(
            "Morph effect requires diffusers. Install it from requirements.txt."
        ) from _DIFFUSERS_IMPORT_ERROR
    if torch is None:
        raise RuntimeError(
            "Morph effect requires torch. Install torch to enable morphing."
        ) from _TORCH_IMPORT_ERROR

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
    ).to(resolved_device)
    return pipeline


def _pick_style(preferred_style: str | None) -> dict:
    if preferred_style and preferred_style != "random":
        for style in ARTISTIC_STYLES:
            if style["name"] == preferred_style:
                return style
        raise ValueError(f"Unknown morph style '{preferred_style}'.")
    return random.choice(ARTISTIC_STYLES)


def generate_image(
    obj_name: str,
    pipeline,
    preferred_style: str | None = None,
    creativity: float = 0.5,
) -> Image.Image | None:
    """Generate one stylized image for a detected object label."""
    if torch is None:
        raise RuntimeError(
            "Morph effect requires torch. Install torch to enable morphing."
        ) from _TORCH_IMPORT_ERROR

    style = _pick_style(preferred_style)
    prompts = style["prompts"]
    prompt_index = random.randint(0, len(prompts) - 1) if creativity > 0 else 0
    prompt = prompts[prompt_index].format(obj=obj_name)

    guidance = float(style["guidance_scale"])
    if creativity > 0:
        guidance += (random.random() - 0.5) * max(0.0, min(creativity, 1.0)) * 2.0
    guidance = max(1.0, guidance)

    autocast_ctx = (
        torch.autocast("cuda") if torch.cuda.is_available() else contextlib.nullcontext()
    )
    with autocast_ctx:
        generated_image = pipeline(prompt, guidance_scale=guidance).images[0]
    return generated_image


def morph_detected_objects(
    current_frame: np.ndarray,
    detected_objects: list[Detection],
    pipeline,
    deep_dream_effect: "DeepDreamEffect | None",
    morph_states: list[MorphState],
    frame_counter: int,
    morph_frames: int = 60,
    morph_min_blend: float = 0.1,
    morph_max_blend: float = 0.7,
    max_objects_per_frame: int = 3,
    preferred_style: str | None = None,
    creativity: float = 0.5,
) -> np.ndarray:
    """Update and render ongoing morphs; spawn morphs for new detections."""
    pil_frame = Image.fromarray(current_frame)
    active_morphs: list[MorphState] = []

    # Continue already-active morphs.
    for morph in list(morph_states):
        if morph.update():
            region = pil_frame.crop(morph.region)
            blend_factor = morph.get_blend_factor()
            mask = Image.new("L", region.size, int(255 * blend_factor))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
            blended_region = Image.composite(morph.generated_image, region, mask)
            pil_frame.paste(blended_region, (morph.region[0], morph.region[1]))
            active_morphs.append(morph)

    if detected_objects:
        frame_h, frame_w = current_frame.shape[:2]
        for index, detection in enumerate(detected_objects[:max_objects_per_frame]):
            x1, y1, x2, y2 = [int(v) for v in detection.box.tolist()]
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))
            if x2 <= x1 or y2 <= y1:
                continue

            overlap = False
            for morph in active_morphs:
                old_x1, old_y1, old_x2, old_y2 = morph.region
                if x1 < old_x2 and x2 > old_x1 and y1 < old_y2 and y2 > old_y1:
                    overlap = True
                    break
            if overlap:
                continue

            generated_image = generate_image(
                detection.label,
                pipeline=pipeline,
                preferred_style=preferred_style,
                creativity=creativity,
            )
            if generated_image is None:
                continue

            region = pil_frame.crop((x1, y1, x2, y2))
            generated_resized = generated_image.resize(region.size)
            generated_array = np.array(generated_resized)
            if deep_dream_effect is not None:
                generated_array = deep_dream_effect.apply(generated_array)
            generated_dreamed_image = Image.fromarray(generated_array)

            morph_state = MorphState(
                obj_id=f"{detection.label}_{index}_{frame_counter}",
                region=(x1, y1, x2, y2),
                generated_image=generated_dreamed_image,
                frame_start=frame_counter,
                frames=morph_frames,
                min_blend=morph_min_blend,
                max_blend=morph_max_blend,
            )
            active_morphs.append(morph_state)

    morph_states.clear()
    morph_states.extend(active_morphs)
    return np.array(pil_frame)


class MorphEffect(Effect):
    """Object-based morph effect driven by detection results."""

    def __init__(
        self,
        detector: Detector | None = None,
        detection_interval: int = 5,
        morph_frames: int = 60,
        morph_min_blend: float = 0.1,
        morph_max_blend: float = 0.7,
        max_objects_per_frame: int = 3,
        preferred_style: str | None = None,
        creativity: float = 0.5,
        deep_dream_effect: "DeepDreamEffect | None" = None,
        pipeline=None,
    ):
        self.detector = detector
        self.detection_interval = max(1, detection_interval)
        self.morph_frames = morph_frames
        self.morph_min_blend = morph_min_blend
        self.morph_max_blend = morph_max_blend
        self.max_objects_per_frame = max_objects_per_frame
        self.preferred_style = preferred_style
        self.creativity = float(np.clip(creativity, 0.0, 1.0))
        self.deep_dream_effect = deep_dream_effect

        self.frame_counter = 0
        self.morph_states: list[MorphState] = []
        self._last_detections: list[Detection] = []
        self._pipeline = pipeline

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = load_stable_diffusion_pipeline()
        return self._pipeline

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply morphing to current frame using active detections and morph states."""
        if self.detector is None:
            self.frame_counter += 1
            return frame

        if self.frame_counter % self.detection_interval == 0:
            self._last_detections = self.detector.detect(frame)

        if not self._last_detections and not self.morph_states:
            self.frame_counter += 1
            return frame

        output = morph_detected_objects(
            current_frame=frame,
            detected_objects=self._last_detections,
            pipeline=self._get_pipeline(),
            deep_dream_effect=self.deep_dream_effect,
            morph_states=self.morph_states,
            frame_counter=self.frame_counter,
            morph_frames=self.morph_frames,
            morph_min_blend=self.morph_min_blend,
            morph_max_blend=self.morph_max_blend,
            max_objects_per_frame=self.max_objects_per_frame,
            preferred_style=self.preferred_style,
            creativity=self.creativity,
        )
        self.frame_counter += 1
        return output

    def reset(self) -> None:
        """Reset morph state while keeping loaded model instances for reuse."""
        self.frame_counter = 0
        self.morph_states.clear()
        self._last_detections = []
