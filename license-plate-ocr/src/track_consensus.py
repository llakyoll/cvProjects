"""Keep stable OCR readings and display crops for ByteTrack plate identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pipeline import PlateResult

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class TrackedPlateRecord:
    """One persistent tracked plate summary for the display panel."""

    track_id: int
    text: str
    consensus_confidence: float
    detection_confidence: float
    crop: np.ndarray
    last_seen_frame: int


class TrackConsensusStore:
    """Accumulate confidence-weighted OCR votes for recent tracker IDs."""

    def __init__(self, max_records: int = 5) -> None:
        """Create an in-memory consensus store with a bounded recent list."""
        self._max_records = max_records
        self._frame_index = 0
        self._states: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []

    def update(self, frame: np.ndarray, results: list[PlateResult]) -> list[TrackedPlateRecord]:
        """Add one frame's tracked OCR readings and return current records."""
        self._frame_index += 1
        for result in results:
            if result.track_id is None or not result.text:
                continue
            crop = _crop(frame, result.bbox)
            if crop is None:
                continue
            weight = max(0.0, (result.ocr_confidence or 0.0) * result.detection_confidence)
            state = self._states.setdefault(
                result.track_id,
                {"votes": defaultdict(float), "best": {}, "detection": 0.0},
            )
            state["votes"][result.text] += weight
            previous = state["best"].get(result.text)
            if previous is None or weight > previous[0]:
                state["best"][result.text] = (weight, crop)
            state["detection"] = result.detection_confidence
            if result.track_id in self._order:
                self._order.remove(result.track_id)
            self._order.insert(0, result.track_id)
        self._order = self._order[: self._max_records]
        self._states = {track_id: self._states[track_id] for track_id in self._order}
        return self.records()

    def records(self) -> list[TrackedPlateRecord]:
        """Return recent records ordered from newest to oldest."""
        records = []
        for track_id in self._order:
            state = self._states[track_id]
            text = max(state["votes"], key=state["votes"].get)
            total = sum(state["votes"].values())
            best_weight, crop = state["best"][text]
            records.append(
                TrackedPlateRecord(
                    track_id, text, state["votes"][text] / total if total else 0.0,
                    state["detection"], crop, self._frame_index,
                )
            )
        return records


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Copy a valid clipped bbox crop for a durable panel record."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if getattr(crop, "size", 0) else None
