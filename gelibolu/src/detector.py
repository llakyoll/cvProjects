"""Person-only YOLO26 detection without tracking or counting."""

from ultralytics import YOLO


class PersonDetector:
    """Run a YOLO detection model while retaining only COCO person detections."""

    PERSON_CLASS = [0]

    def __init__(self, model_path: str = "yolo26l.pt", confidence: float = 0.4, image_size: int = 1280):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):
        """Return detections for a single BGR frame; no tracking state is kept."""
        return self.model(frame, classes=self.PERSON_CLASS, conf=self.confidence, imgsz=self.image_size, verbose=False)[0]

    def track(self, frame, tracker: str = "botsort.yaml"):
        """Return person detections with short-lived IDs; no counts are updated."""
        return self.model.track(
            frame,
            classes=self.PERSON_CLASS,
            conf=self.confidence,
            imgsz=self.image_size,
            tracker=tracker,
            persist=True,
            verbose=False,
        )[0]
