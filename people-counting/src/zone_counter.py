"""Entry/exit zone counter for tracked people.

Counts a track only once per transition by comparing whether its centroid
was inside a rectangular zone on the previous frame versus the current one.
"""


class ZoneCounter:
    """Counts objects entering and exiting a rectangular zone.

    Args:
        zone: (x1, y1, x2, y2) pixel rectangle defining the zone.
    """

    def __init__(self, zone: tuple[int, int, int, int]):
        self.zone = zone
        self.inside_state: dict[int, bool] = {}
        self.entries = 0
        self.exits = 0

    def _is_inside(self, centroid: tuple[float, float]) -> bool:
        x1, y1, x2, y2 = self.zone
        x, y = centroid
        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, track_id: int, centroid: tuple[float, float]) -> None:
        """Updates a track's zone membership and counts entries/exits.

        Args:
            track_id: Stable id assigned by the tracker.
            centroid: (x, y) center of the current detection box.
        """
        currently_inside = self._is_inside(centroid)
        was_inside = self.inside_state.get(track_id, False)

        if currently_inside and not was_inside:
            self.entries += 1
        elif was_inside and not currently_inside:
            self.exits += 1

        self.inside_state[track_id] = currently_inside
