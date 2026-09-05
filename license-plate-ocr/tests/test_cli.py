"""Dependency-light tests for the command-line and visualization contracts."""

from __future__ import annotations

import sys
from types import SimpleNamespace


def _assert_parse_rejects(argv: list[str]) -> None:
    """Assert argparse rejects one invalid CLI invocation."""
    from main import parse_args

    try:
        parse_args(argv)
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError(f"parse_args accepted invalid arguments: {argv}")


def test_parse_source_converts_numeric_webcam_index() -> None:
    from main import parse_source

    assert parse_source("0") == 0
    assert parse_source(" 2 ") == 2


def test_parse_source_preserves_image_and_stream_sources() -> None:
    from main import parse_source

    assert parse_source("assets/car.jpg") == "assets/car.jpg"
    assert parse_source("rtsp://camera.example/live") == "rtsp://camera.example/live"


def test_parse_args_exposes_documented_defaults() -> None:
    from main import parse_args

    args = parse_args(["--source", "0"])

    assert args.source == 0
    assert args.detector_model == "models/license-plate-finetune-v1l.onnx"
    assert args.ocr_model == "cct-xs-v2-global-model"
    assert args.conf == 0.35
    assert args.imgsz == 640
    assert args.device == "cuda:0"
    assert args.output is None
    assert args.polygon is None


def test_parse_polygon_accepts_integer_points() -> None:
    from main import parse_polygon

    assert parse_polygon("100,200;900,200;900,600;100,600") == (
        (100, 200),
        (900, 200),
        (900, 600),
        (100, 600),
    )


def test_format_polygon_returns_cli_value() -> None:
    from main import format_polygon

    assert format_polygon(((100, 200), (900, 200), (900, 600))) == (
        "100,200;900,200;900,600"
    )


def test_parse_args_accepts_polygon_and_rejects_invalid_shapes() -> None:
    from main import parse_args

    args = parse_args(["--source", "0", "--polygon", "0,0;10,0;10,10"])
    assert args.polygon == ((0, 0), (10, 0), (10, 10))
    _assert_parse_rejects(["--source", "0", "--polygon", "0,0;10,0"])
    _assert_parse_rejects(["--source", "0", "--polygon", "0,0;bad,0;10,10"])
    _assert_parse_rejects(["--source", "0", "--polygon", "0,0;10,0;20,0"])


def test_filter_results_by_polygon_uses_bbox_center_and_includes_boundary() -> None:
    from main import filter_results_by_polygon

    results = [
        SimpleNamespace(bbox=(2, 2, 4, 4)),
        SimpleNamespace(bbox=(10, 10, 12, 12)),
        SimpleNamespace(bbox=(0, 4, 2, 6)),
    ]
    polygon = ((0, 0), (10, 0), (10, 10), (0, 10))

    assert filter_results_by_polygon(results, polygon) == [results[0], results[2]]
    assert filter_results_by_polygon(results, None) == results


def test_parse_args_accepts_custom_runtime_values() -> None:
    from main import parse_args

    args = parse_args(
        [
            "--source",
            "input.mp4",
            "--detector-model",
            "custom.onnx",
            "--ocr-model",
            "custom-ocr",
            "--conf",
            "0.6",
            "--imgsz",
            "960",
            "--device",
            "cuda:1",
            "--output",
            "annotated.mp4",
        ]
    )

    assert args.source == "input.mp4"
    assert args.detector_model == "custom.onnx"
    assert args.ocr_model == "custom-ocr"
    assert args.conf == 0.6
    assert args.imgsz == 960
    assert args.device == "cuda:1"
    assert args.output == "annotated.mp4"


def test_parse_args_rejects_non_finite_confidence() -> None:
    _assert_parse_rejects(["--source", "0", "--conf", "nan"])


def test_parse_args_rejects_confidence_outside_inclusive_unit_interval() -> None:
    _assert_parse_rejects(["--source", "0", "--conf", "1.01"])
    _assert_parse_rejects(["--source", "0", "--conf", "-0.01"])


def test_parse_args_rejects_non_positive_image_size() -> None:
    _assert_parse_rejects(["--source", "0", "--imgsz", "0"])
    _assert_parse_rejects(["--source", "0", "--imgsz", "-1"])


def test_parse_args_accepts_inclusive_confidence_bounds() -> None:
    from main import parse_args

    assert parse_args(["--source", "0", "--conf", "0"]).conf == 0.0
    assert parse_args(["--source", "0", "--conf", "1"]).conf == 1.0


def test_invalid_capture_fps_uses_fallback_but_valid_fps_is_preserved() -> None:
    from main import _valid_fps

    assert _valid_fps(29.97) == 29.97
    assert _valid_fps(0) == 25.0
    assert _valid_fps(float("nan")) == 25.0
    assert _valid_fps(float("inf")) == 25.0


def test_resize_for_display_scales_only_wide_frames_with_preserved_aspect_ratio() -> None:
    from main import resize_for_display

    resize_calls = []

    class FakeCV2:
        INTER_AREA = "area"

        @staticmethod
        def resize(frame, size, interpolation):
            resize_calls.append((frame, size, interpolation))
            return "resized-frame"

    class WideFrame:
        shape = (720, 1920, 3)

    class NarrowFrame:
        shape = (720, 960, 3)

    wide_frame = WideFrame()
    narrow_frame = NarrowFrame()

    assert resize_for_display(FakeCV2, wide_frame) == "resized-frame"
    assert resize_calls == [(wide_frame, (1280, 480), "area")]
    assert resize_for_display(FakeCV2, narrow_frame) is narrow_frame


def test_failed_video_writer_is_released_before_reporting_error() -> None:
    from main import _create_writer

    class FailedWriter:
        def __init__(self) -> None:
            self.release_calls = 0

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            self.release_calls += 1

    failed_writer = FailedWriter()

    class FakeCV2:
        @staticmethod
        def VideoWriter_fourcc(*args: str) -> str:
            return "fourcc"

        @staticmethod
        def VideoWriter(*args: object) -> FailedWriter:
            return failed_writer

    class FakeFrame:
        shape = (12, 24, 3)

    try:
        _create_writer(FakeCV2, "annotated.mp4", 25.0, FakeFrame())
    except RuntimeError as error:
        assert str(error) == "Could not open output writer: annotated.mp4"
    else:
        raise AssertionError("an unopened writer must raise RuntimeError")

    assert failed_writer.release_calls == 1


def test_annotate_frame_draws_bbox_and_label_with_cv2() -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 7

        @staticmethod
        def rectangle(*args: object, **kwargs: object) -> None:
            calls.append(("rectangle", args, kwargs))

        @staticmethod
        def putText(*args: object, **kwargs: object) -> None:
            calls.append(("putText", args, kwargs))

    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = FakeCV2
    try:
        from src.visualization import annotate_frame

        frame = object()
        result = annotate_frame(
            frame,
            [
                SimpleNamespace(
                    bbox=(10, 20, 110, 80),
                    detection_confidence=0.91,
                    text="34ABC123",
                    ocr_confidence=0.87,
                    region=None,
                )
            ],
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert result is frame
    assert [call[0] for call in calls] == ["rectangle", "putText"]
    rectangle_args = calls[0][1]
    assert rectangle_args[1:4] == ((10, 20), (110, 80), (0, 255, 0))
    label = calls[1][1][1]
    assert "34ABC123" in label
    assert "D:0.91" in label
    assert "O:0.87" in label


def test_annotate_frame_draws_polygon_when_provided() -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 7

        @staticmethod
        def rectangle(*args: object, **kwargs: object) -> None:
            calls.append(("rectangle", args, kwargs))

        @staticmethod
        def putText(*args: object, **kwargs: object) -> None:
            calls.append(("putText", args, kwargs))

        @staticmethod
        def line(*args: object, **kwargs: object) -> None:
            calls.append(("line", args, kwargs))

    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = FakeCV2
    try:
        from src.visualization import annotate_frame

        annotate_frame(object(), [], ((1, 2), (10, 2), (10, 8)))
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert [call[0] for call in calls] == ["line", "line", "line"]


def test_annotate_frame_handles_missing_ocr_confidence() -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 7

        @staticmethod
        def rectangle(*args: object, **kwargs: object) -> None:
            calls.append(("rectangle", args, kwargs))

        @staticmethod
        def putText(*args: object, **kwargs: object) -> None:
            calls.append(("putText", args, kwargs))

    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = FakeCV2
    try:
        from src.visualization import annotate_frame

        annotate_frame(
            object(),
            [
                SimpleNamespace(
                    bbox=(1, 2, 3, 4),
                    detection_confidence=0.5,
                    text="UNKNOWN",
                    ocr_confidence=None,
                    region="TR",
                )
            ],
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert "D:0.50" in calls[1][1][1]
    assert "O:n/a" in calls[1][1][1]


def test_compose_plate_panel_adds_fixed_header_and_scene_below() -> None:
    import sys

    from src.visualization import compose_plate_panel

    class FakeArray:
        def __init__(self, shape: tuple[int, int, int]) -> None:
            self.shape = shape
            self.dtype = "uint8"
            self.size = shape[0] * shape[1] * shape[2]

        def __getitem__(self, key: object) -> "FakeArray":
            if isinstance(key, tuple) and len(key) == 2:
                y_slice, x_slice = key
                height = len(range(*y_slice.indices(self.shape[0])))
                width = len(range(*x_slice.indices(self.shape[1])))
                return FakeArray((height, width, self.shape[2]))
            return self

        def __setitem__(self, key: object, value: object) -> None:
            pass

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 7
        INTER_AREA = 3

        @staticmethod
        def putText(*args: object, **kwargs: object) -> None:
            pass

        @staticmethod
        def resize(crop: FakeArray, size: tuple[int, int], **kwargs: object) -> FakeArray:
            return FakeArray((size[1], size[0], crop.shape[2]))

    class FakeNumpy:
        uint8 = "uint8"

        @staticmethod
        def zeros(shape: tuple[int, int, int], dtype: object) -> FakeArray:
            return FakeArray(shape)

        @staticmethod
        def vstack(values: tuple[FakeArray, FakeArray]) -> FakeArray:
            return FakeArray((sum(value.shape[0] for value in values), values[0].shape[1], 3))

    frame = FakeArray((20, 40, 3))
    result = SimpleNamespace(
        bbox=(5, 4, 25, 12),
        detection_confidence=0.91,
        text="34ABC123",
        ocr_confidence=0.87,
    )

    previous_cv2 = sys.modules.get("cv2")
    previous_numpy = sys.modules.get("numpy")
    sys.modules["cv2"] = FakeCV2
    sys.modules["numpy"] = FakeNumpy
    try:
        composed = compose_plate_panel(frame, [result], panel_height=10)
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy

    assert composed.shape == (30, 40, 3)


def test_compose_plate_panel_renders_empty_state() -> None:
    import sys

    from src.visualization import compose_plate_panel

    calls: list[str] = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 7
        INTER_AREA = 3

        @staticmethod
        def putText(*args: object, **kwargs: object) -> None:
            calls.append(str(args[1]))

    class FakeNumpy:
        @staticmethod
        def zeros(shape: tuple[int, int, int], dtype: object) -> object:
            return SimpleNamespace(shape=shape, dtype=dtype)

        @staticmethod
        def vstack(values: tuple[object, object]) -> object:
            return object()

    frame = SimpleNamespace(shape=(20, 40, 3), dtype="uint8")

    previous_cv2 = sys.modules.get("cv2")
    previous_numpy = sys.modules.get("numpy")
    sys.modules["cv2"] = FakeCV2
    sys.modules["numpy"] = FakeNumpy
    try:
        compose_plate_panel(frame, [], panel_height=140)
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy

    assert "No plates detected" in calls


def test_compose_plate_panel_skips_invalid_boxes_and_labels_valid_plate() -> None:
    import sys

    from src.visualization import compose_plate_panel

    calls: list[str] = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 7
        INTER_AREA = 3

        @staticmethod
        def putText(*args: object, **kwargs: object) -> None:
            calls.append(str(args[1]))

        @staticmethod
        def resize(*args: object, **kwargs: object) -> object:
            return SimpleNamespace(shape=(40, 100, 3))

    class FakeFrame:
        shape = (100, 400, 3)
        dtype = "uint8"
        size = 120000

        def __getitem__(self, key: object) -> object:
            if isinstance(key, tuple) and len(key) == 2:
                return self
            return self

        def __setitem__(self, key: object, value: object) -> None:
            pass

    previous_cv2 = sys.modules.get("cv2")
    previous_numpy = sys.modules.get("numpy")
    sys.modules["cv2"] = FakeCV2
    sys.modules["numpy"] = SimpleNamespace(
        zeros=lambda shape, dtype: FakeFrame(),
        vstack=lambda values: FakeFrame(),
    )
    try:
        compose_plate_panel(
            FakeFrame(),
            [
                SimpleNamespace(
                    bbox=(10, 10, 10, 30),
                    detection_confidence=0.1,
                    text="bad",
                    ocr_confidence=0.1,
                ),
                SimpleNamespace(
                    bbox=(-5, 10, 40, 30),
                    detection_confidence=0.91,
                    text="34ABC123",
                    ocr_confidence=0.87,
                ),
            ],
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy

    assert any("#1" in label for label in calls)
    assert "34ABC123" in calls
    assert not any("bad" in label for label in calls)


def test_compose_plate_panel_shows_header_and_overflow_count() -> None:
    import sys

    from src.visualization import compose_plate_panel

    labels: list[str] = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 7
        INTER_AREA = 3

        @staticmethod
        def putText(*args: object, **kwargs: object) -> None:
            labels.append(str(args[1]))

        @staticmethod
        def rectangle(*args: object, **kwargs: object) -> None:
            pass

        @staticmethod
        def resize(*args: object, **kwargs: object) -> object:
            return SimpleNamespace(shape=(40, 100, 3))

    class FakeFrame:
        shape = (100, 320, 3)
        dtype = "uint8"
        size = 96000

        def __getitem__(self, key: object) -> object:
            return self

        def __setitem__(self, key: object, value: object) -> None:
            pass

    results = [
        SimpleNamespace(
            bbox=(10, 10, 40, 30),
            detection_confidence=0.9,
            text=f"34ABC{index}",
            ocr_confidence=0.8,
        )
        for index in range(7)
    ]
    previous_cv2 = sys.modules.get("cv2")
    previous_numpy = sys.modules.get("numpy")
    sys.modules["cv2"] = FakeCV2
    sys.modules["numpy"] = SimpleNamespace(
        zeros=lambda shape, dtype: FakeFrame(),
        vstack=lambda values: FakeFrame(),
    )
    try:
        compose_plate_panel(FakeFrame(), results)
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy

    assert "RECENT DETECTIONS" in labels
    assert "+2 more" in labels
