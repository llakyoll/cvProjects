"""Dependency-free behavioral tests for the FastPlateOCR adapter."""

from unittest.mock import patch

from src import ocr
from src.ocr import PlateOCR, PlateReading


class FakeCrop:
    """Represent a BGR crop without requiring NumPy."""

    def __init__(self, pixels):
        self.pixels = pixels


class FakeImage:
    """Represent an RGB image after the fake OpenCV conversion."""

    def __init__(self, pixels, dtype="int32"):
        self.pixels = pixels
        self.dtype = dtype
        self.layout = "channels-last"


class FakeCV2:
    """Record BGR-to-RGB conversions made by the adapter."""

    COLOR_BGR2RGB = "bgr2rgb"

    def __init__(self):
        self.calls = []

    def cvtColor(self, crop, conversion):
        self.calls.append((crop, conversion))
        rgb_pixels = tuple(reversed(crop.pixels))
        return FakeImage(rgb_pixels)


class FakeNumPy:
    """Provide only the array conversion needed by the adapter."""

    uint8 = "uint8"

    @staticmethod
    def asarray(image, dtype):
        image.dtype = dtype
        return image


class FakeRecognizer:
    """Record one batch call and return controlled prediction records."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def run(self, crops, return_confidence):
        self.calls.append((crops, return_confidence))
        return self.predictions


def test_recognize_batch_converts_bgr_crops_and_forwards_uint8_channels_last_batch():
    cv2_backend = FakeCV2()
    recognizer = FakeRecognizer(
        [
            {"text": " ab 12 ", "char_probs": [0.8, 0.9], "region": "TR"},
            {"text": " xy ", "char_probs": None},
        ]
    )
    crops = [FakeCrop((1, 2, 3)), FakeCrop((4, 5, 6))]

    with patch.object(ocr, "cv2", cv2_backend), patch.object(ocr, "np", FakeNumPy()):
        readings = PlateOCR(recognizer=recognizer).recognize_batch(crops)

    assert cv2_backend.calls == [
        (crops[0], FakeCV2.COLOR_BGR2RGB),
        (crops[1], FakeCV2.COLOR_BGR2RGB),
    ]
    forwarded_crops, return_confidence = recognizer.calls[0]
    assert [crop.pixels for crop in forwarded_crops] == [(3, 2, 1), (6, 5, 4)]
    assert [crop.dtype for crop in forwarded_crops] == [FakeNumPy.uint8, FakeNumPy.uint8]
    assert [crop.layout for crop in forwarded_crops] == ["channels-last", "channels-last"]
    assert return_confidence is True


def test_recognize_batch_normalizes_text_aggregates_confidence_and_forwards_region():
    recognizer = FakeRecognizer(
        [
            {"text": "  ab 12\n", "char_probs": [0.8, 0.9], "region": "TR"},
        ]
    )
    crop = FakeCrop((1, 2, 3))

    with patch.object(ocr, "cv2", FakeCV2()), patch.object(ocr, "np", FakeNumPy()):
        readings = PlateOCR(recognizer=recognizer).recognize_batch([crop])

    assert readings[0].text == "AB 12"
    assert abs(readings[0].confidence - 0.85) < 1e-12
    assert readings[0].region == "TR"


def test_recognize_batch_returns_one_reading_without_confidence_when_probabilities_are_missing():
    recognizer = FakeRecognizer([{"text": " xy ", "char_probs": None}])

    with patch.object(ocr, "cv2", FakeCV2()), patch.object(ocr, "np", FakeNumPy()):
        readings = PlateOCR(recognizer=recognizer).recognize_batch([FakeCrop((1, 2, 3))])

    assert readings == [PlateReading(text="XY", confidence=None, region=None)]


def test_recognize_batch_skips_backend_for_empty_input():
    class FailingRecognizer:
        def run(self, *args, **kwargs):
            raise AssertionError("empty input must not call the recognizer")

    assert PlateOCR(recognizer=FailingRecognizer()).recognize_batch([]) == []


def test_production_constructor_creates_fast_plate_recognizer_with_requested_settings():
    created = []

    class FakeLicensePlateRecognizer:
        def __init__(self, model_name, *, device):
            created.append((model_name, device))

    with patch.object(ocr, "LicensePlateRecognizer", FakeLicensePlateRecognizer):
        PlateOCR(model_name="custom-model", device="cpu")

    assert created == [("custom-model", "cpu")]


def test_production_constructor_normalizes_indexed_cuda_for_fast_plate_ocr():
    created = []

    class FakeLicensePlateRecognizer:
        def __init__(self, model_name, *, device):
            created.append((model_name, device))

    with patch.object(ocr, "LicensePlateRecognizer", FakeLicensePlateRecognizer):
        PlateOCR(model_name="custom-model", device="cuda:0")

    assert created == [("custom-model", "cuda")]
