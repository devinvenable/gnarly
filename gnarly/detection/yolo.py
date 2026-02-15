"""YOLO-based object detection."""

from typing import List

import numpy as np
import torch
from PIL import Image

from ..config import DetectionConfig
from .base import Detection


class YOLODetector:
    """YOLO object detector implementing the Detector protocol.

    Attributes:
        model: YOLOv5 model loaded from torch hub.
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IOU threshold for NMS.
        max_objects: Maximum number of objects to return per frame.
        max_size_ratio: Maximum object size as fraction of frame area.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        max_objects: int = 10,
        max_size_ratio: float = 0.25,
        device: str = "cuda",
    ):
        """Initialize YOLODetector.

        Args:
            confidence_threshold: Minimum confidence score (0.0 to 1.0).
            iou_threshold: IOU threshold for non-max suppression.
            max_objects: Maximum detections to return per frame.
            max_size_ratio: Max object area as fraction of frame area.
            device: Device to run inference on ('cuda' or 'cpu').
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_objects = max_objects
        self.max_size_ratio = max_size_ratio
        self.device = device
        self.model = self._load_model()

    @classmethod
    def from_config(cls, config: DetectionConfig, device: str = "cuda") -> "YOLODetector":
        """Create YOLODetector from DetectionConfig.

        Args:
            config: Detection configuration.
            device: Device to run inference on.

        Returns:
            Configured YOLODetector instance.
        """
        return cls(
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            max_objects=config.max_objects,
            device=device,
        )

    def _load_model(self) -> torch.nn.Module:
        """Load YOLOv5 model from torch hub.

        Returns:
            Loaded YOLOv5 model.

        Raises:
            RuntimeError: If model loading fails.
        """
        print("Loading YOLOv5 model for object detection...")
        try:
            model = torch.hub.load(
                "ultralytics/yolov5", "yolov5x", pretrained=True
            )
            model = model.to(self.device).eval()
            model.conf = self.confidence_threshold
            model.iou = self.iou_threshold
            print("YOLOv5 model loaded successfully!")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5 model: {e}") from e

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.

        Args:
            frame: Input frame as RGB numpy array (H, W, 3).

        Returns:
            List of Detection objects, sorted by confidence.
        """
        pil_image = Image.fromarray(frame)
        width, height = pil_image.size
        frame_area = width * height
        max_area = frame_area * self.max_size_ratio

        # Update model thresholds
        self.model.conf = self.confidence_threshold
        self.model.iou = self.iou_threshold

        results = self.model(pil_image, size=500)
        detections_df = results.pandas().xyxy[0]

        if detections_df.empty:
            return []

        # Calculate object dimensions
        detections_df["obj_width"] = detections_df["xmax"] - detections_df["xmin"]
        detections_df["obj_height"] = detections_df["ymax"] - detections_df["ymin"]
        detections_df["area"] = detections_df["obj_width"] * detections_df["obj_height"]

        # Filter by size
        detections_df = detections_df[detections_df["area"] < max_area]

        if detections_df.empty:
            return []

        # Keep top N by confidence
        detections_df = detections_df.nlargest(self.max_objects, "confidence")

        # Convert to Detection objects
        detections = []
        for _, row in detections_df.iterrows():
            box = np.array([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            detection = Detection(
                box=box,
                score=float(row["confidence"]),
                label=str(row["name"]),
                class_id=int(row["class"]),
            )
            detections.append(detection)

        if detections:
            labels = ", ".join(set(d.label for d in detections))
            scores = ", ".join(f"{d.score:.2f}" for d in detections)
            print(f"Detected {len(detections)} objects (YOLO): {labels}")
            print(f"Confidence scores: {scores}")

        return detections
