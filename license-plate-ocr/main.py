"""Command-line entry point for GPU license-plate detection and OCR."""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from src.visualization import annotate_frame, compose_plate_panel

if TYPE_CHECKING:
    import numpy as np

    from src.pipeline import PlatePipeline
    from src.plate_associator import PlateAssociator


IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
DEFAULT_DETECTOR_MODEL = "models/license-plate-finetune-v1l.pt"
DEFAULT_OCR_MODEL = "cct-xs-v2-global-model"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IMAGE_SIZE = 640
DEFAULT_DEVICE = "cuda:0"
DEFAULT_OUTPUT_FPS = 25.0
DEFAULT_DISPLAY_MAX_WIDTH = 1280
Polygon = tuple[tuple[int, int], ...]


class ProcessingFpsMeter:
    """Smooth completed-frame throughput for a stable UI reading.

    Args:
        smoothing: Weight assigned to the newest instantaneous FPS value.
    """

    def __init__(self, smoothing: float = 0.2) -> None:
        self.smoothing = smoothing
        self.value: float | None = None

    def observe(self, elapsed_seconds: float) -> float:
        """Record one completed frame duration and return smoothed FPS.

        Args:
            elapsed_seconds: End-to-end processing duration for one frame.

        Returns:
            The exponentially smoothed processing rate in frames per second.
        """
        instantaneous_fps = 1.0 / elapsed_seconds
        if self.value is None:
            self.value = instantaneous_fps
        else:
            self.value += self.smoothing * (instantaneous_fps - self.value)
        return self.value


def parse_source(source: str) -> str | int:
    """Convert a numeric source string to a webcam index.

    Args:
        source: Image/video path, stream URL, or webcam index text.

    Returns:
        An integer webcam index for non-negative numeric text; otherwise the
        original source string.
    """
    normalized = source.strip()
    if normalized.isdigit():
        return int(normalized)
    return source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the documented license-plate OCR command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect and read license plates from an image or stream."
    )
    parser.add_argument("--source", required=True, help="Image, video, RTSP URL, or webcam index")
    parser.add_argument(
        "--detector-model",
        default=DEFAULT_DETECTOR_MODEL,
        help="Local path for the pinned detector PyTorch model",
    )
    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help="FastPlateOCR model name",
    )
    parser.add_argument("--conf", type=_confidence_type, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=_positive_int_type, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--output", help="Optional annotated image/video output path")
    parser.add_argument("--debug", action="store_true", help="Print tracker, OCR, and panel diagnostics")
    parser.add_argument(
        "--polygon",
        type=parse_polygon,
        help="ROI polygon as x1,y1;x2,y2;...; omit to draw it interactively",
    )
    args = parser.parse_args(argv)
    args.source = parse_source(args.source)
    return args


def is_image_source(source: str | int) -> bool:
    """Return whether a string source has a supported still-image suffix."""
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def _confidence_type(value: str) -> float:
    """Parse a finite detector confidence in the inclusive range ``[0, 1]``."""
    try:
        confidence = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--conf must be a number in [0, 1]") from error
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise argparse.ArgumentTypeError("--conf must be a finite number in [0, 1]")
    return confidence


def _positive_int_type(value: str) -> int:
    """Parse a strictly positive integer image size."""
    try:
        image_size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--imgsz must be a positive integer") from error
    if image_size <= 0:
        raise argparse.ArgumentTypeError("--imgsz must be a positive integer")
    return image_size


def parse_polygon(value: str) -> Polygon:
    """Parse a semicolon-separated polygon made of integer ``x,y`` points."""
    points: list[tuple[int, int]] = []
    try:
        for raw_point in value.split(";"):
            coordinates = raw_point.strip().split(",")
            if len(coordinates) != 2:
                raise ValueError
            points.append((int(coordinates[0].strip()), int(coordinates[1].strip())))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "--polygon must contain integer points in x1,y1;x2,y2;... format"
        ) from error
    if len(points) < 3:
        raise argparse.ArgumentTypeError("--polygon must contain at least 3 points")
    area_twice = sum(
        points[index - 1][0] * point[1]
        - points[index - 1][1] * point[0]
        for index, point in enumerate(points)
    )
    if area_twice == 0:
        raise argparse.ArgumentTypeError("--polygon must enclose a non-zero area")
    return tuple(points)


def format_polygon(polygon: Polygon) -> str:
    """Format polygon points for reuse with the ``--polygon`` option."""
    return ";".join(f"{x},{y}" for x, y in polygon)


def point_in_polygon(point: tuple[float, float], polygon: Polygon) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""
    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[index - 1]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if cross == 0 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def filter_results_by_polygon(results: list[Any], polygon: Polygon | None) -> list[Any]:
    """Keep detections whose bounding-box center lies inside the ROI."""
    if polygon is None:
        return results
    filtered = []
    for result in results:
        x1, y1, x2, y2 = result.bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if point_in_polygon(center, polygon):
            filtered.append(result)
    return filtered


def _associate_results(
    frame: np.ndarray,
    results: list[Any],
    associator: PlateAssociator,
    consensus: Any,
) -> tuple[list[Any], list[Any]]:
    """Resolve application IDs before updating persistent panel records."""
    resolved_results = associator.update(results)
    return resolved_results, consensus.update(frame, resolved_results)


def run(args: argparse.Namespace) -> None:
    """Run the detector/OCR pipeline for an image or frame stream.

    Args:
        args: Parsed command-line namespace from :func:`parse_args`.

    Raises:
        RuntimeError: If OpenCV cannot open a source, read an image, or open
            the requested output writer.
    """
    cv2 = _load_cv2()
    source = args.source
    capture: Any | None = None
    writer: Any | None = None
    try:
        from src.model_manager import ensure_detector_model
        from src.ocr import PlateOCR
        from src.pipeline import PlatePipeline
        from src.plate_associator import PlateAssociator
        from src.plate_detector import PlateDetector
        from src.track_consensus import TrackConsensusStore

        model_path = ensure_detector_model(Path(args.detector_model))
        detector = PlateDetector(
            model_path,
            confidence=args.conf,
            image_size=args.imgsz,
            device=args.device,
            debug=args.debug,
        )
        ocr = PlateOCR(model_name=args.ocr_model, device=args.device)
        pipeline = PlatePipeline(detector, ocr, debug=args.debug)
        associator = PlateAssociator()
        consensus = TrackConsensusStore(max_records=5)

        if is_image_source(source):
            _run_image(
                cv2,
                pipeline,
                source,
                args.output,
                args.polygon,
                associator,
                consensus,
                args.debug,
            )
            return

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open capture source: {source}")

        fps = _valid_fps(capture.get(cv2.CAP_PROP_FPS))
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Could not read first frame from source: {source}")
        polygon = args.polygon or select_polygon(cv2, frame)
        display = args.output is None
        fps_meter = ProcessingFpsMeter()
        processing_fps: float | None = None
        while True:
            frame_started_at = perf_counter()
            observations = filter_results_by_polygon(pipeline.process_frame(frame), polygon)
            backend_ids = sum(item.track_id is not None for item in observations)
            results, records = _associate_results(
                frame, observations, associator, consensus
            )
            if args.debug:
                print(
                    f"[debug] results={len(results)} backend_ids={backend_ids} "
                    f"resolved_ids={sum(item.track_id is not None for item in results)} "
                    f"records={len(records)}"
                )
            consensus_text = {record.track_id: record.text for record in records}
            annotated_scene = annotate_frame(frame.copy(), results, polygon, consensus_text)
            annotated = compose_plate_panel(
                frame,
                records,
                scene=annotated_scene,
                processing_fps=processing_fps,
                source_fps=fps,
            )
            processing_fps = fps_meter.observe(perf_counter() - frame_started_at)
            if writer is None and args.output is not None:
                writer = _create_writer(cv2, args.output, fps, annotated)
            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("License Plate OCR", resize_for_display(cv2, annotated))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            success, frame = capture.read()
            if not success:
                break
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def _run_image(
    cv2: Any,
    pipeline: PlatePipeline,
    source: str,
    output: str | None,
    polygon: Polygon | None,
    associator: PlateAssociator,
    consensus: Any,
    debug: bool,
) -> None:
    """Process and optionally save one still image."""
    frame = cv2.imread(source)
    if frame is None:
        raise RuntimeError(f"Could not read image source: {source}")
    selected_polygon = polygon or select_polygon(cv2, frame)
    observations = filter_results_by_polygon(
        pipeline.process_frame(frame), selected_polygon
    )
    backend_ids = sum(item.track_id is not None for item in observations)
    results, records = _associate_results(
        frame, observations, associator, consensus
    )
    if debug:
        print(
            f"[debug] results={len(results)} backend_ids={backend_ids} "
            f"resolved_ids={sum(item.track_id is not None for item in results)} "
            f"records={len(records)}"
        )
    consensus_text = {record.track_id: record.text for record in records}
    annotated_scene = annotate_frame(frame.copy(), results, selected_polygon, consensus_text)
    annotated = compose_plate_panel(frame, records, scene=annotated_scene)
    if output is not None and not cv2.imwrite(output, annotated):
        raise RuntimeError(f"Could not write annotated image: {output}")
    if output is None:
        cv2.imshow("License Plate OCR", resize_for_display(cv2, annotated))
        cv2.waitKey(0)


def select_polygon(cv2: Any, frame: np.ndarray) -> Polygon:
    """Interactively select an ROI and return points in original-frame coordinates."""
    window_name = "Select license-plate ROI"
    original_height, original_width = frame.shape[:2]
    preview = resize_for_display(cv2, frame)
    preview_width = preview.shape[1]
    scale = preview_width / original_width
    points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(
                (
                    min(original_width - 1, max(0, round(x / scale))),
                    min(original_height - 1, max(0, round(y / scale))),
                )
            )
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, getattr(cv2, "WINDOW_NORMAL", 0))
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            canvas = preview.copy()
            display_points = [(round(x * scale), round(y * scale)) for x, y in points]
            for index, point in enumerate(display_points):
                cv2.circle(canvas, point, 5, (0, 255, 255), -1)
                if index:
                    cv2.line(canvas, display_points[index - 1], point, (255, 0, 0), 2)
            if len(display_points) > 2:
                cv2.line(canvas, display_points[-1], display_points[0], (255, 0, 0), 2)
            cv2.putText(
                canvas,
                "Left click: add | Right click: undo | Enter: confirm | Esc/q: cancel",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                getattr(cv2, "LINE_AA", 16),
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                raise RuntimeError("Polygon selection cancelled")
            if key in (10, 13):
                if len(points) < 3:
                    continue
                selected_polygon = tuple(points)
                print(f"Selected ROI polygon: {format_polygon(selected_polygon)}")
                print(
                    "Reuse with: --polygon "
                    f'"{format_polygon(selected_polygon)}"'
                )
                return selected_polygon
    finally:
        destroy_window = getattr(cv2, "destroyWindow", None)
        if destroy_window is not None:
            destroy_window(window_name)


def resize_for_display(
    cv2: Any,
    frame: np.ndarray,
    max_width: int = DEFAULT_DISPLAY_MAX_WIDTH,
) -> np.ndarray:
    """Shrink a frame for display while retaining its original aspect ratio.

    Args:
        cv2: OpenCV-compatible module that provides ``resize``.
        frame: Annotated image frame to show in the preview window.
        max_width: Largest display width in pixels.

    Returns:
        The original frame when it fits, otherwise a proportionally resized one.
    """
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scaled_height = max(1, round(height * max_width / width))
    return cv2.resize(
        frame,
        (max_width, scaled_height),
        interpolation=cv2.INTER_AREA,
    )


def _valid_fps(value: Any) -> float:
    """Return source FPS or the fallback only for non-finite/non-positive FPS."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return DEFAULT_OUTPUT_FPS
    return fps if isfinite(fps) and fps > 0 else DEFAULT_OUTPUT_FPS


def _create_writer(cv2: Any, output: str, fps: float, frame: np.ndarray) -> Any:
    """Create and validate an output writer sized to the first frame."""
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (int(width), int(height)))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open output writer: {output}")
    return writer


def _load_cv2() -> Any:
    """Import OpenCV at runtime so argument parsing stays dependency-light."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise ImportError("opencv-python is required to run license-plate OCR") from error
    return cv2


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and execute the license-plate OCR application."""
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
