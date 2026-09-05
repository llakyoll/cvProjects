"""Dependency-free behavioral tests for the detection-to-OCR pipeline."""

from src.ocr import PlateReading
from src.pipeline import PlatePipeline, PlateResult, crop_with_padding
from src.types import PlateDetection


class FakeCrop:
    """Represent one crop returned by a fake frame."""


class EmptyCrop:
    """Represent a geometrically sliced crop with zero elements."""

    size = 0


class FakeFrame:
    """Record NumPy-style crop slices without importing NumPy."""

    def __init__(self, height, width):
        self.shape = (height, width, 3)
        self.slices = []

    def __getitem__(self, key):
        self.slices.append(key)
        return FakeCrop()


class EmptyCropFrame(FakeFrame):
    """Return an empty crop despite receiving valid slice bounds."""

    def __getitem__(self, key):
        self.slices.append(key)
        return EmptyCrop()


class FakeDetector:
    """Return controlled detections and record detector calls."""

    def __init__(self, detections):
        self.detections = detections
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame)
        return self.detections

    def track(self, frame):
        self.frames.append(frame)
        return self.detections


class FakeOCR:
    """Return controlled readings and record crop batches."""

    def __init__(self, readings):
        self.readings = readings
        self.batches = []

    def recognize_batch(self, crops):
        self.batches.append(crops)
        return self.readings


def test_crop_with_padding_expands_bbox_and_clips_to_frame_bounds():
    frame = FakeFrame(height=10, width=12)
    detection = PlateDetection(bbox=(2, 1, 10, 9), confidence=0.91)

    crop = crop_with_padding(frame, detection, padding_ratio=0.25)

    assert isinstance(crop, FakeCrop)
    assert frame.slices == [(slice(0, 10), slice(0, 12))]


def test_crop_with_padding_returns_none_for_empty_or_invalid_boxes():
    frame = FakeFrame(height=10, width=12)

    assert crop_with_padding(
        frame, PlateDetection(bbox=(4, 2, 4, 8), confidence=0.8)
    ) is None
    assert crop_with_padding(
        frame, PlateDetection(bbox=(2, 8, 10, 2), confidence=0.8)
    ) is None
    assert frame.slices == []


def test_crop_with_padding_rejects_a_sliced_crop_with_zero_elements():
    frame = EmptyCropFrame(height=10, width=12)
    detection = PlateDetection(bbox=(2, 2, 8, 8), confidence=0.8)

    crop = crop_with_padding(frame, detection)

    assert crop is None
    assert frame.slices == [(slice(2, 8), slice(2, 8))]


def test_pipeline_skips_zero_sized_sliced_crop_without_calling_ocr():
    frame = EmptyCropFrame(height=10, width=12)
    detector = FakeDetector(
        [PlateDetection(bbox=(2, 2, 8, 8), confidence=0.8)]
    )
    ocr = FakeOCR([])

    assert PlatePipeline(detector, ocr).process_frame(frame) == []
    assert ocr.batches == []


def test_pipeline_pairs_ordered_readings_with_detection_metadata():
    frame = FakeFrame(height=20, width=30)
    detections = [
        PlateDetection(bbox=(2, 3, 10, 9), confidence=0.91),
        PlateDetection(bbox=(14, 5, 25, 12), confidence=0.82),
    ]
    readings = [
        PlateReading(text="ABC 123", confidence=0.88, region="TR"),
        PlateReading(text="XYZ 987", confidence=None, region=None),
    ]
    detector = FakeDetector(detections)
    ocr = FakeOCR(readings)

    results = PlatePipeline(detector, ocr).process_frame(frame)

    assert detector.frames == [frame]
    assert len(ocr.batches) == 1
    assert len(ocr.batches[0]) == 2
    assert results == [
        PlateResult(
            bbox=(2, 3, 10, 9),
            detection_confidence=0.91,
            text="ABC 123",
            ocr_confidence=0.88,
            region="TR",
        ),
        PlateResult(
            bbox=(14, 5, 25, 12),
            detection_confidence=0.82,
            text="XYZ 987",
            ocr_confidence=None,
            region=None,
        ),
    ]


def test_pipeline_preserves_tracker_id_in_plate_result():
    frame = FakeFrame(height=20, width=30)
    detector = FakeDetector([PlateDetection((2, 3, 10, 9), 0.91, track_id=11)])
    ocr = FakeOCR([PlateReading(text="ABC123", confidence=0.88, region=None)])

    results = PlatePipeline(detector, ocr).process_frame(frame)

    assert results[0].track_id == 11


def test_pipeline_skips_invalid_detections_and_returns_empty_without_valid_crops():
    frame = FakeFrame(height=20, width=30)
    detector = FakeDetector(
        [PlateDetection(bbox=(4, 4, 4, 9), confidence=0.7)]
    )
    ocr = FakeOCR([])

    assert PlatePipeline(detector, ocr).process_frame(frame) == []
    assert ocr.batches == []


def test_pipeline_returns_empty_for_no_detections_without_calling_ocr():
    detector = FakeDetector([])
    ocr = FakeOCR([])

    assert PlatePipeline(detector, ocr).process_frame(FakeFrame(20, 30)) == []
    assert ocr.batches == []
