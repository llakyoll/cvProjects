"""Behavioral tests for the YOLO license-plate detector adapter."""

from pathlib import Path

from src import plate_detector
from src.plate_detector import PlateDetector
from src.types import PlateDetection


class FakeBoxes:
    """Small stand-in for the Ultralytics boxes container."""

    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf


class FakeResult:
    """Small stand-in for one Ultralytics prediction result."""

    def __init__(self, xyxy, conf):
        self.boxes = FakeBoxes(xyxy, conf)


class FakeModel:
    """Record predictions while returning a controlled fake result."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def test_detect_converts_fake_yolo_boxes_to_plate_detections():
    model = FakeModel(
        FakeResult(
            xyxy=[[11.9, 22.1, 133.8, 244.7]],
            conf=[0.91],
        )
    )
    detector = PlateDetector(Path("detector.onnx"), model=model)

    detections = detector.detect(object())

    assert detections == [PlateDetection(bbox=(11, 22, 133, 244), confidence=0.91)]
    assert type(detections[0].confidence) is float


def test_detect_filters_boxes_below_configured_confidence():
    model = FakeModel(
        FakeResult(
            xyxy=[[1.0, 2.0, 30.0, 40.0], [5.0, 6.0, 50.0, 60.0]],
            conf=[0.34, 0.35],
        )
    )
    detector = PlateDetector(Path("detector.onnx"), confidence=0.35, model=model)

    detections = detector.detect(object())

    assert detections == [PlateDetection(bbox=(5, 6, 50, 60), confidence=0.35)]


def test_detect_returns_empty_list_when_result_has_no_boxes():
    model = FakeModel(FakeResult(xyxy=[], conf=[]))
    detector = PlateDetector(Path("detector.onnx"), model=model)

    assert detector.detect(object()) == []


def test_detect_passes_required_arguments_to_model_predict():
    frame = object()
    model = FakeModel(FakeResult(xyxy=[], conf=[]))
    detector = PlateDetector(
        Path("detector.onnx"),
        confidence=0.6,
        image_size=512,
        device="cpu",
        model=model,
    )

    detector.detect(frame)

    assert model.calls == [
        {
            "source": frame,
            "conf": 0.6,
            "imgsz": 512,
            "device": "cpu",
            "verbose": False,
        }
    ]


def test_constructor_loads_onnx_path_through_ultralytics(monkeypatch):
    loaded_paths = []

    def fake_yolo(model_path):
        loaded_paths.append(model_path)
        return FakeModel(FakeResult(xyxy=[], conf=[]))

    monkeypatch.setattr(plate_detector, "YOLO", fake_yolo)

    PlateDetector(Path("detector.onnx"))

    assert loaded_paths == [Path("detector.onnx")]
