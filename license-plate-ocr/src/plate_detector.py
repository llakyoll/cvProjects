"""Adapt Ultralytics YOLO ONNX predictions to the project detection type.

The adapter keeps backend-specific result objects at the detector boundary so
the rest of the pipeline can consume stable integer boxes and Python floats.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - exercised only without runtime deps.
    YOLO = None

from .types import PlateDetection

if TYPE_CHECKING:
    import numpy as np


class PlateDetector:
    """Run a YOLOv11 ONNX model and normalize its plate detections."""

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "cuda:0",
        model: object | None = None,
    ) -> None:
        """Initialize the detector backend and inference settings.

        Args:
            model_path: Path to the supplied YOLO ONNX model.
            confidence: Minimum confidence retained from backend results.
            image_size: Inference image size forwarded to Ultralytics.
            device: Inference device forwarded to Ultralytics.
            model: Optional backend override used by dependency-free tests.

        Raises:
            ImportError: If production construction is requested without
                ``ultralytics`` installed.
        """
        self._confidence = confidence
        self._image_size = image_size
        self._device = device

        if model is not None:
            self._model = model
            return

        if YOLO is None:
            raise ImportError(
                "ultralytics is required to construct PlateDetector without a model"
            )
        self._model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        """Detect license plates in one frame.

        Args:
            frame: Image frame passed unchanged to the YOLO backend.

        Returns:
            Normalized detections with integer pixel coordinates and Python
            float confidence values.
        """
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            verbose=False,
        )
        return self._normalize_results(results)

    def track(self, frame: np.ndarray) -> list[PlateDetection]:
        """Detect plates and associate them with persistent ByteTrack IDs."""
        results = self._model.track(
            source=frame,
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        return self._normalize_results(results, include_track_ids=True)

    def _normalize_results(
        self, results: object, include_track_ids: bool = False
    ) -> list[PlateDetection]:
        """Normalize backend prediction results at the adapter boundary."""
        if not results:
            return []

        result = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        coordinates = getattr(boxes, "xyxy", [])
        confidences = getattr(boxes, "conf", [])
        identifiers = getattr(boxes, "id", None)
        detections: list[PlateDetection] = []
        for index, (box, confidence) in enumerate(zip(coordinates, confidences)):
            score = float(confidence)
            if score < self._confidence:
                continue
            detections.append(
                PlateDetection(
                    bbox=tuple(int(coordinate) for coordinate in box),
                    confidence=score,
                    track_id=(
                        int(identifiers[index])
                        if include_track_ids and identifiers is not None
                        else None
                    ),
                )
            )
        return detections
