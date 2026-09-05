"""Adapt FastPlateOCR predictions to stable license-plate readings.

The adapter owns image color conversion and prediction normalization so the
pipeline can consume dependency-independent ``PlateReading`` values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only without runtime deps.
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without runtime deps.
    np = None

try:
    from fast_plate_ocr import LicensePlateRecognizer
except ImportError:  # pragma: no cover - exercised only without runtime deps.
    LicensePlateRecognizer = None

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class PlateReading:
    """Represent one normalized OCR result for a license plate.

    Attributes:
        text: Uppercase plate text with surrounding whitespace removed.
        confidence: Mean character probability, when provided by the backend.
        region: Optional region code exposed by the backend prediction.
    """

    text: str
    confidence: float | None
    region: str | None


class PlateOCR:
    """Run FastPlateOCR in batches and normalize its predictions."""

    def __init__(
        self,
        model_name: str = "cct-xs-v2-global-model",
        device: str = "cuda",
        recognizer: object | None = None,
    ) -> None:
        """Initialize the OCR recognizer.

        Args:
            model_name: FastPlateOCR model identifier used in production.
            device: Inference device forwarded to FastPlateOCR.
            recognizer: Optional backend override used by dependency-free tests.

        Raises:
            ImportError: If production construction is requested without
                ``fast-plate-ocr`` installed.
        """
        if recognizer is not None:
            self._recognizer = recognizer
            return

        if LicensePlateRecognizer is None:
            raise ImportError(
                "fast-plate-ocr is required to construct PlateOCR without a recognizer"
            )
        self._recognizer = LicensePlateRecognizer(
            model_name,
            device=_normalize_fast_plate_ocr_device(device),
        )

    def recognize_batch(self, crops: list[np.ndarray]) -> list[PlateReading]:
        """Recognize an ordered batch of OpenCV BGR plate crops.

        Args:
            crops: OpenCV BGR crops in channels-last image layout.

        Returns:
            One normalized reading per input crop, preserving input order.

        Raises:
            ImportError: If image dependencies are unavailable when inference
                is requested.
            ValueError: If the backend returns a different number of readings
                than input crops.
        """
        if not crops:
            return []
        if cv2 is None or np is None:
            raise ImportError("opencv-python and numpy are required for OCR inference")

        rgb_crops = [
            np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            for crop in crops
        ]
        backend_result = self._recognizer.run(rgb_crops, return_confidence=True)
        predictions = _unpack_predictions(backend_result, len(crops))
        return [_to_reading(prediction) for prediction in predictions]


def _unpack_predictions(result: Any, expected_count: int) -> list[Any]:
    """Normalize the supported FastPlateOCR batch result shapes."""
    if isinstance(result, tuple) and len(result) == 2:
        texts, character_probabilities = result
        if _is_batch(texts, expected_count) and _is_batch(
            character_probabilities, expected_count
        ):
            return [
                _merge_text_and_confidence(text, probabilities)
                for text, probabilities in zip(
                    texts, character_probabilities, strict=True
                )
            ]

    if isinstance(result, Mapping) and "predictions" in result:
        result = result["predictions"]
    elif hasattr(result, "predictions"):
        result = result.predictions

    if isinstance(result, str):
        predictions = [result]
    else:
        try:
            predictions = list(result)
        except TypeError:
            predictions = [result]

    if len(predictions) != expected_count:
        raise ValueError(
            "FastPlateOCR returned "
            f"{len(predictions)} readings for {expected_count} crops"
        )
    return predictions


def _is_batch(value: Any, expected_count: int) -> bool:
    """Return whether a backend value is a batch with the expected length."""
    return isinstance(value, (list, tuple)) and len(value) == expected_count


def _merge_text_and_confidence(text: Any, confidence: Any) -> Any:
    """Combine tuple-style text and character-probability batch outputs."""
    if isinstance(confidence, Mapping):
        merged = dict(confidence)
        merged.setdefault("text", text)
        return merged
    return {"text": text, "char_probs": confidence}


def _field(prediction: Any, names: tuple[str, ...], default: Any = None) -> Any:
    """Read a named field from either a mapping or an object prediction."""
    if isinstance(prediction, Mapping):
        for name in names:
            if name in prediction:
                return prediction[name]
        return default

    for name in names:
        if hasattr(prediction, name):
            return getattr(prediction, name)
    return default


def _to_reading(prediction: Any) -> PlateReading:
    """Convert one backend prediction into a ``PlateReading`` value."""
    if isinstance(prediction, str):
        text = prediction
        char_probs = None
        region = None
    elif isinstance(prediction, (list, tuple)):
        text = prediction[0] if prediction else ""
        char_probs = prediction[1] if len(prediction) > 1 else None
        region = prediction[2] if len(prediction) > 2 else None
    else:
        text = _field(prediction, ("text", "plate", "prediction", "label"), "")
        char_probs = _field(prediction, ("char_probs",), None)
        region = _field(prediction, ("region",), None)

    confidence = _mean_probability(char_probs)
    return PlateReading(
        text=str(text).strip().upper(),
        confidence=confidence,
        region=None if region is None else str(region),
    )


def _mean_probability(char_probs: Any) -> float | None:
    """Return the mean of character probabilities when available."""
    if char_probs is None:
        return None
    try:
        probabilities = [float(probability) for probability in char_probs]
    except TypeError:
        probabilities = [float(char_probs)]
    if not probabilities:
        return None
    return float(sum(probabilities) / len(probabilities))


def _normalize_fast_plate_ocr_device(device: str) -> str:
    """Map indexed CUDA devices to the device names accepted by FastPlateOCR."""
    if device.startswith("cuda:"):
        return "cuda"
    return device
