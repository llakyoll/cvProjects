"""Track-aware, two-gate turnstile passage counting."""

from collections import defaultdict

import numpy as np


ENTRY_DIRECTIONS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def segment_intersection_progress(
    movement_start: tuple[float, float], movement_end: tuple[float, float], gate_start: tuple[float, float], gate_end: tuple[float, float]
) -> float | None:
    """Return where a movement segment crosses a finite gate line, from 0 to 1."""
    movement = (movement_end[0] - movement_start[0], movement_end[1] - movement_start[1])
    gate = (gate_end[0] - gate_start[0], gate_end[1] - gate_start[1])
    denominator = _cross(movement, gate)
    if abs(denominator) < 1e-8:
        return None
    offset = (gate_start[0] - movement_start[0], gate_start[1] - movement_start[1])
    movement_progress = _cross(offset, gate) / denominator
    gate_progress = _cross(offset, movement) / denominator
    if 0 < movement_progress <= 1 and 0 <= gate_progress <= 1:
        return movement_progress
    return None


class LanePassageCounter:
    """Count a track only after it crosses both calibrated lane gates in sequence."""

    def __init__(self, lanes: list[dict]):
        self.lanes = lanes
        self.stats = {lane["id"]: {"entry": 0, "exit": 0} for lane in lanes}
        self.track_lane_state: dict[int, dict[str, dict]] = defaultdict(dict)
        self.previous_points: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _gate_midpoint(gate: list[list[int]]) -> tuple[float, float]:
        return (gate[0][0] + gate[1][0]) / 2, (gate[0][1] + gate[1][1]) / 2

    def _bidirectional_entry_gate(self, lane: dict) -> str:
        """Map the configured visual entry direction to gate A or B."""
        entry_dx, entry_dy = ENTRY_DIRECTIONS[lane["entry_direction"]]
        midpoint_a = self._gate_midpoint(lane["gates"]["a"])
        midpoint_b = self._gate_midpoint(lane["gates"]["b"])
        gate_dx, gate_dy = midpoint_b[0] - midpoint_a[0], midpoint_b[1] - midpoint_a[1]
        return "a" if gate_dx * entry_dx + gate_dy * entry_dy > 0 else "b"

    def _flow_for_sequence(self, lane: dict, first_gate: str) -> str | None:
        flow = lane.get("flow")
        if flow in {"entry", "exit"}:
            # Gate A is calibrated on the expected approaching side for a
            # single-direction lane; the reverse sequence is discarded.
            return flow if first_gate == "a" else None
        if flow == "bidirectional":
            return "entry" if first_gate == self._bidirectional_entry_gate(lane) else "exit"
        return None

    @staticmethod
    def _advance_state(state: dict, gate_name: str) -> str | None:
        """Advance a two-gate state machine and return the first crossed gate on completion."""
        stage = state["stage"]
        if stage is None:
            state["stage"] = gate_name
            return None
        if stage == gate_name:
            return None
        state["stage"] = None
        return stage

    def update(self, track_id: int, track_point: tuple[float, float]) -> list[dict]:
        """Update one tracked box-centre point and return verified two-gate passage events."""
        events = []
        previous = self.previous_points.get(track_id)
        lane_states = self.track_lane_state[track_id]

        if previous is None:
            self.previous_points[track_id] = track_point
            return events

        for lane in self.lanes:
            lane_id = lane["id"]
            state = lane_states.setdefault(lane_id, {"stage": None, "counted": False})
            if state["counted"]:
                continue

            crossings = []
            for gate_name in ("a", "b"):
                gate_start, gate_end = lane["gates"][gate_name]
                progress = segment_intersection_progress(previous, track_point, tuple(gate_start), tuple(gate_end))
                if progress is not None:
                    crossings.append((progress, gate_name))

            # A fast track can cross both lines inside one frame; processing by
            # progress preserves the actual A→B or B→A order.
            for _progress, gate_name in sorted(crossings):
                first_gate = self._advance_state(state, gate_name)
                if first_gate is None:
                    continue
                event_flow = self._flow_for_sequence(lane, first_gate)
                if event_flow is not None:
                    self.stats[lane_id][event_flow] += 1
                    state["counted"] = True
                    events.append({
                        "lane_id": lane_id,
                        "flow": event_flow,
                        "track_id": track_id,
                        "sequence": f"{first_gate.upper()}→{gate_name.upper()}",
                    })

        self.previous_points[track_id] = track_point
        return events

    def totals(self) -> dict[str, int]:
        return {
            "entry": sum(lane_stats["entry"] for lane_stats in self.stats.values()),
            "exit": sum(lane_stats["exit"] for lane_stats in self.stats.values()),
        }
