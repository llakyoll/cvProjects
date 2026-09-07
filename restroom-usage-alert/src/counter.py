"""Direction-aware counter for a user-drawn line segment."""


class EntranceCounter:
    """Count a short-lived track once when it crosses a selected line segment."""

    def __init__(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        entry_direction: str = "both",
    ):
        if start == end:
            raise ValueError("The two selected points must be different.")
        self.start = start
        self.end = end
        self.entry_direction = entry_direction
        self.track_history: dict[int, tuple[float, float]] = {}
        self.counted_tracks: set[int] = set()
        self.entry_count = 0

    def _side(self, point: tuple[float, float]) -> float:
        """Return which side of the directed start -> end line a point occupies."""
        x1, y1 = self.start
        x2, y2 = self.end
        x, y = point
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def _crosses_segment(self, previous: tuple[float, float], current: tuple[float, float]) -> bool:
        """Return whether a movement segment intersects the selected line."""
        previous_side = self._side(previous)
        current_side = self._side(current)
        # A centroid can land exactly on a horizontal/vertical line for one
        # frame. Treat that contact as a crossing candidate instead of losing
        # the event on the following frame.
        if (previous_side == 0 and current_side == 0) or previous_side * current_side > 0:
            return False

        x1, y1 = self.start
        x2, y2 = self.end
        px, py = previous
        cx, cy = current
        movement_side_1 = (cx - px) * (y1 - py) - (cy - py) * (x1 - px)
        movement_side_2 = (cx - px) * (y2 - py) - (cy - py) * (x2 - px)
        return movement_side_1 == 0 or movement_side_2 == 0 or movement_side_1 * movement_side_2 < 0

    def update(self, track_id: int, centroid: tuple[float, float]) -> bool:
        """Update one tracked person and return ``True`` when an entry is counted."""
        previous = self.track_history.get(track_id)
        entered = False

        if previous is not None and track_id not in self.counted_tracks and self._crosses_segment(previous, centroid):
            previous_side = self._side(previous)
            current_side = self._side(centroid)
            moving_forward = previous_side <= 0 <= current_side and (previous_side < 0 or current_side > 0)
            moving_backward = previous_side >= 0 >= current_side and (previous_side > 0 or current_side < 0)
            if self.entry_direction == "both" or (
                self.entry_direction == "forward" and moving_forward
            ) or (self.entry_direction == "backward" and moving_backward):
                self.counted_tracks.add(track_id)
                self.entry_count += 1
                entered = True

        self.track_history[track_id] = centroid
        return entered
