"""Shared immutable value types used by the license-plate OCR pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlateDetection:
    """Represent one detector output for a license plate.

    Attributes:
        bbox: Integer pixel coordinates in ``(x1, y1, x2, y2)`` order.
        confidence: Detector confidence score for the bounding box.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
