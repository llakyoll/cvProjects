"""Resolve detector observations into stable application-level plate identities.

Ultralytics tracker IDs remain the strongest identity signal, while a
motion-and-text fallback keeps observations usable when the backend returns no
ID for fast-moving or temporarily unconfirmed tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from math import hypot, log

from .pipeline import PlateResult


@dataclass
class _TrackState:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    velocity: tuple[float, float]
    text: str
    last_seen_frame: int


class PlateAssociator:
    """Assign stable IDs using tracker output, motion, size, and OCR text.

    Args:
        max_missed_frames: Number of absent frames before an identity expires.
        max_center_distance: Maximum center displacement normalized by the
            larger bounding-box diagonal.
        min_size_ratio: Smallest accepted current-to-previous box area ratio.
        max_size_ratio: Largest accepted current-to-previous box area ratio.
    """

    def __init__(
        self,
        max_missed_frames: int = 30,
        max_center_distance: float = 3.0,
        min_size_ratio: float = 0.4,
        max_size_ratio: float = 2.5,
    ) -> None:
        self._max_missed_frames = max_missed_frames
        self._max_center_distance = max_center_distance
        self._min_size_ratio = min_size_ratio
        self._max_size_ratio = max_size_ratio
        self._frame_index = 0
        self._next_id = 1
        self._states: dict[int, _TrackState] = {}
        self._external_ids: dict[int, int] = {}

    def update(self, results: list[PlateResult]) -> list[PlateResult]:
        """Return observations carrying stable application-level IDs.

        Args:
            results: OCR observations from one frame, optionally carrying
                Ultralytics tracker IDs.

        Returns:
            Results in their original order with a non-null application ID.
        """
        self._frame_index += 1
        self._expire_stale_states()
        assignments: dict[int, int] = {}
        used_track_ids: set[int] = set()

        for result_index, result in enumerate(results):
            external_id = result.track_id
            internal_id = self._external_ids.get(external_id) if external_id is not None else None
            if internal_id in self._states and internal_id not in used_track_ids:
                assignments[result_index] = internal_id
                used_track_ids.add(internal_id)

        candidates: list[tuple[float, int, int]] = []
        for result_index, result in enumerate(results):
            if result_index in assignments:
                continue
            for internal_id, state in self._states.items():
                if internal_id in used_track_ids:
                    continue
                cost = self._match_cost(state, result)
                if cost is not None:
                    candidates.append((cost, result_index, internal_id))

        for _cost, result_index, internal_id in sorted(candidates):
            if result_index in assignments or internal_id in used_track_ids:
                continue
            assignments[result_index] = internal_id
            used_track_ids.add(internal_id)

        resolved: list[PlateResult] = []
        for result_index, result in enumerate(results):
            internal_id = assignments.get(result_index)
            if internal_id is None:
                internal_id = self._allocate_id(result.track_id)
            if result.track_id is not None:
                self._external_ids[result.track_id] = internal_id
            self._update_state(internal_id, result)
            resolved.append(replace(result, track_id=internal_id))
        return resolved

    def _match_cost(self, state: _TrackState, result: PlateResult) -> float | None:
        center = _center(result.bbox)
        frame_gap = max(1, self._frame_index - state.last_seen_frame)
        predicted_center = (
            state.center[0] + state.velocity[0] * frame_gap,
            state.center[1] + state.velocity[1] * frame_gap,
        )
        normalizer = max(_diagonal(state.bbox), _diagonal(result.bbox), 1.0)
        center_distance = hypot(
            center[0] - predicted_center[0], center[1] - predicted_center[1]
        ) / normalizer
        if center_distance > self._max_center_distance:
            return None

        previous_area = _area(state.bbox)
        current_area = _area(result.bbox)
        size_ratio = current_area / previous_area if previous_area else 0.0
        if not self._min_size_ratio <= size_ratio <= self._max_size_ratio:
            return None

        previous_text = _normalize_text(state.text)
        current_text = _normalize_text(result.text)
        text_similarity = (
            SequenceMatcher(None, previous_text, current_text).ratio()
            if previous_text and current_text
            else 0.5
        )
        return center_distance + 0.5 * abs(log(size_ratio)) + 0.75 * (1.0 - text_similarity)

    def _allocate_id(self, preferred_id: int | None) -> int:
        if preferred_id is not None and preferred_id > 0 and preferred_id not in self._states:
            internal_id = preferred_id
        else:
            while self._next_id in self._states:
                self._next_id += 1
            internal_id = self._next_id
            self._next_id += 1
        self._next_id = max(self._next_id, internal_id + 1)
        return internal_id

    def _update_state(self, internal_id: int, result: PlateResult) -> None:
        center = _center(result.bbox)
        previous = self._states.get(internal_id)
        if previous is None:
            velocity = (0.0, 0.0)
        else:
            frame_gap = max(1, self._frame_index - previous.last_seen_frame)
            velocity = (
                (center[0] - previous.center[0]) / frame_gap,
                (center[1] - previous.center[1]) / frame_gap,
            )
        self._states[internal_id] = _TrackState(
            bbox=result.bbox,
            center=center,
            velocity=velocity,
            text=result.text or (previous.text if previous is not None else ""),
            last_seen_frame=self._frame_index,
        )

    def _expire_stale_states(self) -> None:
        stale_ids = {
            internal_id
            for internal_id, state in self._states.items()
            if self._frame_index - state.last_seen_frame > self._max_missed_frames
        }
        for internal_id in stale_ids:
            self._states.pop(internal_id, None)
        self._external_ids = {
            external_id: internal_id
            for external_id, internal_id in self._external_ids.items()
            if internal_id in self._states
        }


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _diagonal(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return hypot(max(0, x2 - x1), max(0, y2 - y1))


def _normalize_text(text: str) -> str:
    return "".join(character for character in text.upper() if character.isalnum())
