"""EfficientDet detector implementation."""

from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from effdet import create_model, DetBenchPredict
except ImportError:
    create_model = None
    DetBenchPredict = None

from .base import Detection, Detector


class EfficientDetDetector(Detector):
    """EfficientDet object detector."""

    def __init__(
        self,
        model_name: str = "tf_efficientdet_d0",
        confidence_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize EfficientDet detector.

        Args:
            model_name: Name of the EfficientDet model to load.
            confidence_threshold: Confidence threshold for detections.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        if create_model is None:
            raise ImportError(
                "effdet is not installed. Please install it with 'pip install effdet'."
            )

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load EfficientDet model for inference.

        Ported from main.py:299-312.
        """
        try:
            model = create_model(model_name, pretrained=True)
            # Wrap the model with DetBenchPredict so that it returns post-processed predictions.
            model = DetBenchPredict(model)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading EfficientDet model: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.

        Ported from main.py:340-444.
        """
        try:
            pil_image = Image.fromarray(frame)
            original_width, original_height = pil_image.size
            target_size = 512
            pil_image_resized = pil_image.resize((target_size, target_size), Image.LANCZOS)

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            img_tensor = transform(pil_image_resized).unsqueeze(0).to(self.device)

            with torch.no_grad():
                raw_outputs = self.model(img_tensor)

            outputs = None
            # Handle direct tensor output of shape [batch, num_detections, 6]
            if torch.is_tensor(raw_outputs) and raw_outputs.ndim == 3 and raw_outputs.shape[-1] >= 6:
                scores = raw_outputs[..., 4]
                mask = scores > self.confidence_threshold
                if mask.any():
                    outputs = {
                        "boxes": raw_outputs[..., :4][mask],
                        "scores": scores[mask],
                        "labels": raw_outputs[..., 5][mask]
                    }
            # Handle dictionary output
            elif isinstance(raw_outputs, dict):
                if "detections" in raw_outputs:
                    detections = raw_outputs["detections"]
                    if torch.is_tensor(detections) and detections.shape[-1] >= 6:
                        outputs = {
                            "boxes": detections[..., :4],
                            "scores": detections[..., 4],
                            "labels": detections[..., 5]
                        }
                else:
                    required_keys = {"boxes", "scores", "labels"}
                    if all(key in raw_outputs for key in required_keys):
                        outputs = raw_outputs

            if outputs is None:
                return []

            def convert_output(x):
                if torch.is_tensor(x):
                    return x.cpu().numpy()
                elif isinstance(x, list):
                    try:
                        return np.array(x)
                    except Exception:
                        return np.vstack(x)
                else:
                    return np.array(x)

            boxes = convert_output(outputs["boxes"])
            scores = convert_output(outputs["scores"])
            labels = convert_output(outputs["labels"])

            if boxes.ndim != 2:
                boxes = boxes.reshape(-1, 4)

            scale_x = original_width / target_size
            scale_y = original_height / target_size

            detections = []
            min_size = 20  # Default from main.py
            for i in range(len(boxes)):
                if scores[i] < self.confidence_threshold:
                    continue

                x_min, y_min, x_max, y_max = boxes[i]

                # Filtering by size as in main.py
                if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
                    continue

                # Rescale boxes to original frame size
                x_min *= scale_x
                x_max *= scale_x
                y_min *= scale_y
                y_max *= scale_y

                detections.append(
                    Detection(
                        box=np.array([x_min, y_min, x_max, y_max]),
                        score=float(scores[i]),
                        label=str(int(labels[i])),
                        class_id=int(labels[i])
                    )
                )

            return detections

        except Exception as e:
            # Replicating print from main.py
            print(f"Error during EfficientDet object detection: {e}")
            return []
