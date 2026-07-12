"""Vehicle detection wrapper around a YOLO model.

Isolated from the counting/tracking logic so the detection backend can be
swapped (e.g. YOLOv8 -> TensorRT export) without touching the counting code.
"""

from ultralytics import YOLO


class VehicleDetector:
    """Runs YOLO inference on a frame and returns vehicle detections.

    Args:
        model_path: Path or name of the YOLO weights to load.
        conf_threshold: Minimum confidence to keep a detection.
        classes: COCO class ids to keep (defaults to vehicle classes:
            car, motorcycle, bus, truck).
    """

    VEHICLE_CLASSES = [2, 3, 5, 7]

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.4,
                 classes: list[int] | None = None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.classes = classes or self.VEHICLE_CLASSES

    def detect(self, frame):
        """Runs detection on a single frame.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            Ultralytics Results object for the frame.
        """
        return self.model(frame, conf=self.conf_threshold, classes=self.classes, verbose=False)[0]

    def track(self, frame):
        """Runs detection plus tracking on a single frame.

        Uses Ultralytics' built-in tracker (ByteTrack by default) to assign
        a stable id to each detection across frames, which the counter
        needs to avoid double-counting the same vehicle.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            Ultralytics Results object with box.id populated per detection.
        """
        return self.model.track(frame, conf=self.conf_threshold, classes=self.classes,
                                 persist=True, verbose=False)[0]
