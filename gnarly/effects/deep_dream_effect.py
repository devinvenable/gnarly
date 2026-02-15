"""Deep Dream effect implementation."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .base import Effect

try:
    import torch
    from torchvision import models, transforms
    from torchvision.models import Inception_V3_Weights
except ImportError as exc:  # pragma: no cover - optional dependency path
    torch = None
    models = None
    transforms = None
    Inception_V3_Weights = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_DREAM_LAYERS = [
    "Mixed_7a.branch3x3_2a.conv",
    "Mixed_7b.branch3x3_2b.conv",
    "Mixed_7c.branch3x3_2c.conv",
]


def preprocess_image_for_deep_dream(image: Image.Image, device: "torch.device") -> "torch.Tensor":
    """Preprocess PIL image for InceptionV3 input."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


def postprocess_deep_dream_output(tensor: "torch.Tensor") -> np.ndarray:
    """Convert normalized tensor back to uint8 RGB image."""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    return np.array(transforms.ToPILImage()(tensor))


def load_deep_dream_model(device: "torch.device") -> "torch.nn.Module":
    """Load and return InceptionV3 model for Deep Dream."""
    return models.inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        aux_logits=True,
    ).to(device).eval()


def deep_dream_processing(
    image: np.ndarray,
    model: "torch.nn.Module",
    layers: list[str],
    device: "torch.device",
    iterations: int = 20,
    lr: float = 0.02,
) -> np.ndarray:
    """Apply Deep Dream optimization to one RGB frame."""
    original_size = image.shape[:2][::-1]
    min_size = 224
    pil_img = Image.fromarray(image)
    if pil_img.width < min_size or pil_img.height < min_size:
        pil_img = pil_img.resize(
            (max(pil_img.width, min_size), max(pil_img.height, min_size)),
            Image.LANCZOS,
        )
        resized_for_processing = True
    else:
        resized_for_processing = False

    input_tensor = preprocess_image_for_deep_dream(pil_img, device)
    activations: list[torch.Tensor] = []

    def hook_fn(module, hook_input, output):
        del module, hook_input
        activations.append(output)

    hooks = []
    for name, module in model.named_modules():
        if name in layers:
            hooks.append(module.register_forward_hook(hook_fn))

    if not hooks:
        return image

    input_tensor.requires_grad_(True)
    optimizer = torch.optim.Adam([input_tensor], lr=lr, weight_decay=1e-4)

    try:
        for _ in range(iterations):
            optimizer.zero_grad()
            activations.clear()
            output = model(input_tensor)
            if isinstance(output, tuple):
                _ = output[0]
            if not activations:
                break
            loss = torch.stack([activation.norm() for activation in activations]).sum()
            loss.backward()
            optimizer.step()
    finally:
        for hook in hooks:
            hook.remove()

    output_image = postprocess_deep_dream_output(input_tensor)
    if resized_for_processing:
        output_image = cv2.resize(output_image, original_size, interpolation=cv2.INTER_LINEAR)
    return output_image


class DeepDreamEffect(Effect):
    """Per-frame Deep Dream effect."""

    def __init__(
        self,
        iterations: int = 30,
        learning_rate: float = 0.02,
        layers: list[str] | None = None,
    ):
        """Initialize Deep Dream with a single shared model instance."""
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Deep Dream dependencies are missing. Install torch and torchvision."
            ) from _IMPORT_ERROR
        if not torch.cuda.is_available():
            raise RuntimeError("Deep Dream requires a CUDA-capable GPU.")

        self.device = torch.device("cuda")
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.layers = list(layers) if layers else list(DEFAULT_DREAM_LAYERS)
        self.model = load_deep_dream_model(self.device)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply Deep Dream to a frame."""
        return deep_dream_processing(
            image=frame,
            model=self.model,
            layers=self.layers,
            device=self.device,
            iterations=self.iterations,
            lr=self.learning_rate,
        )

    def reset(self) -> None:
        """Reset effect state (stateless effect)."""
        return None
