"""Two-line corridor counter for tracked people.

Counts a track only once it has crossed both lines of a corridor in
sequence, not just touched a single line. This filters out people who
merely approach the counting area and turn away, or jitter back and forth
across one line without actually passing through.
"""


class CorridorCounter:
    """Counts objects that cross both lines of a two-line corridor, in order.

    The line with the smaller pixel coordinate is treated as the "near"
    line and the one with the larger coordinate as the "far" line. A track
    is counted as an "entry" once it crosses near -> far (increasing
    coordinate), and an "exit" once it crosses far -> near (decreasing
    coordinate). Crossing only one of the two lines does not count.

    Args:
        line1: Pixel coordinate (y for horizontal, x for vertical) of the
            first corridor line.
        line2: Pixel coordinate of the second corridor line.
        orientation: "horizontal" or "vertical".
    """

    def __init__(self, line1: int, line2: int, orientation: str = "horizontal"):
        self.line_near, self.line_far = sorted((line1, line2))
        self.orientation = orientation
        self.track_history: dict[int, tuple[float, float]] = {}
        self.track_state: dict[int, str] = {}
        self.entries = 0
        self.exits = 0

    @staticmethod
    def _crossing_direction(prev_value: float, current_value: float, line_position: int) -> str | None:
        if prev_value < line_position <= current_value:
            return "forward"
        if prev_value > line_position >= current_value:
            return "backward"
        return None

    def update(self, track_id: int, centroid: tuple[float, float]) -> None:
        """Updates a track's position and counts it once it clears the corridor.

        Args:
            track_id: Stable id assigned by the tracker.
            centroid: (x, y) center of the current detection box.
        """
        axis = 1 if self.orientation == "horizontal" else 0
        current = centroid[axis]
        prev = self.track_history.get(track_id)
        state = self.track_state.get(track_id, "none")

        if prev is not None and state != "counted":
            prev_value = prev[axis]
            cross_near = self._crossing_direction(prev_value, current, self.line_near)
            cross_far = self._crossing_direction(prev_value, current, self.line_far)

            if state == "none" and cross_near == "forward":
                state = "entered_near"
            elif state == "none" and cross_far == "backward":
                state = "entered_far"
            elif state == "entered_near" and cross_far == "forward":
                self.entries += 1
                state = "counted"
            elif state == "entered_far" and cross_near == "backward":
                self.exits += 1
                state = "counted"

        self.track_history[track_id] = centroid
        self.track_state[track_id] = state
