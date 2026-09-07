"""Person detection and short-lived tracking for the entrance feed."""

from ultralytics import YOLO


class PersonDetector:
    """Runs person-only YOLO tracking without persisting identity data."""

    PERSON_CLASS = [0]

    def __init__(
        self,
        model_path: str = "yolo26l.pt",
        conf_threshold: float = 0.4,
        tracker: str = "botsort.yaml",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker = tracker

    def track(self, frame):
        """Return tracked person detections for one BGR video frame."""
        return self.model.track(
            frame,
            conf=self.conf_threshold,
            classes=self.PERSON_CLASS,
            persist=True,
            tracker=self.tracker,
            verbose=False,
        )[0]
