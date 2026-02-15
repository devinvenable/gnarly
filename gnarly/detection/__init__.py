"""Detection module for object and face detection."""

from .base import Detection, Detector
from .yolo import YOLODetector

__all__ = ["Detection", "Detector", "YOLODetector"]
