"""Person detection wrapper around a YOLO model.

Isolated from the zone/counting logic so the detection backend can be
swapped (e.g. YOLOv8 -> TensorRT export) without touching the counting code.
"""

from ultralytics import YOLO


class PersonDetector:
    """Runs YOLO inference on a frame and returns person detections + track ids.

    Args:
        model_path: Path or name of the YOLO weights to load.
        conf_threshold: Minimum confidence to keep a detection.
    """

    PERSON_CLASS = [0]

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track(self, frame):
        """Runs detection plus tracking on a single frame.

        Uses Ultralytics' built-in tracker (ByteTrack by default) to assign
        a stable id to each detection across frames, which the zone counter
        needs to tell entries from exits.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            Ultralytics Results object with box.id populated per detection.
        """
        return self.model.track(frame, conf=self.conf_threshold, classes=self.PERSON_CLASS,
                                 persist=True, verbose=False)[0]
