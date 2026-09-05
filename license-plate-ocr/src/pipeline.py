"""Compose plate detection, padded cropping, and batched OCR per frame.

The pipeline deliberately owns no temporal state or presentation behavior so a
caller can process each frame independently and decide how to visualize or
track the returned readings.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from .ocr import PlateOCR
from .plate_detector import PlateDetector
from .types import PlateDetection

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class PlateResult:
    """Combine one detection with its corresponding OCR reading.

    Attributes:
        bbox: Original detector coordinates in ``(x1, y1, x2, y2)`` order.
        detection_confidence: Confidence reported by the detector.
        text: Normalized plate text reported by OCR.
        ocr_confidence: OCR confidence, when the backend provides one.
        region: Optional region code reported by OCR.
    """

    bbox: tuple[int, int, int, int]
    detection_confidence: float
    text: str
    ocr_confidence: float | None
    region: str | None


def crop_with_padding(
    frame: np.ndarray,
    detection: PlateDetection,
    padding_ratio: float = 0.08,
) -> np.ndarray | None:
    """Return a clipped padded crop for one detection.

    Args:
        frame: Image array with height and width available through ``shape``.
        detection: Detection whose box is expressed as ``(x1, y1, x2, y2)``.
        padding_ratio: Fraction of the box width and height added on each
            side.

    Returns:
        A view of the padded crop, or ``None`` when the frame, padding, or
        clipped box is invalid or empty.
    """
    try:
        height, width = (int(dimension) for dimension in frame.shape[:2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None

    if height <= 0 or width <= 0 or not isfinite(padding_ratio) or padding_ratio < 0:
        return None

    x1, y1, x2, y2 = detection.bbox
    if x2 <= x1 or y2 <= y1:
        return None

    padding_x = int((x2 - x1) * padding_ratio)
    padding_y = int((y2 - y1) * padding_ratio)
    padded_x1 = max(0, x1 - padding_x)
    padded_y1 = max(0, y1 - padding_y)
    padded_x2 = min(width, x2 + padding_x)
    padded_y2 = min(height, y2 + padding_y)

    if padded_x2 <= padded_x1 or padded_y2 <= padded_y1:
        return None

    crop = frame[padded_y1:padded_y2, padded_x1:padded_x2]
    crop_size = getattr(crop, "size", None)
    if crop_size is not None:
        if crop_size == 0:
            return None
    else:
        try:
            if len(crop) == 0:
                return None
        except TypeError:
            pass
    return crop


class PlatePipeline:
    """Run stateless detection-to-OCR processing for individual frames."""

    def __init__(
        self,
        detector: PlateDetector,
        ocr: PlateOCR,
        padding_ratio: float = 0.08,
    ) -> None:
        """Initialize a pipeline with detector and OCR collaborators.

        Args:
            detector: Object that detects plates in a frame.
            ocr: Object that recognizes an ordered batch of plate crops.
            padding_ratio: Fraction of each detection box added around crops.
        """
        self._detector = detector
        self._ocr = ocr
        self._padding_ratio = padding_ratio

    def process_frame(self, frame: np.ndarray) -> list[PlateResult]:
        """Detect and recognize all valid plates in one frame.

        Args:
            frame: Image frame passed to the detector and cropped for OCR.

        Returns:
            Ordered plate results, or an empty list when no valid crop exists.

        Raises:
            ValueError: If OCR returns a different number of readings than
                valid crops.
        """
        detections = self._detector.detect(frame)
        if not detections:
            return []

        valid_detections: list[PlateDetection] = []
        crops: list[np.ndarray] = []
        for detection in detections:
            crop = crop_with_padding(frame, detection, self._padding_ratio)
            if crop is None:
                continue
            valid_detections.append(detection)
            crops.append(crop)

        if not crops:
            return []

        readings = self._ocr.recognize_batch(crops)
        if len(readings) != len(valid_detections):
            raise ValueError(
                "OCR returned "
                f"{len(readings)} readings for {len(valid_detections)} crops"
            )

        return [
            PlateResult(
                bbox=detection.bbox,
                detection_confidence=detection.confidence,
                text=reading.text,
                ocr_confidence=reading.confidence,
                region=reading.region,
            )
            for detection, reading in zip(valid_detections, readings, strict=True)
        ]
