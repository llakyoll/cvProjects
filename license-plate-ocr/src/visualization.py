"""OpenCV rendering helpers for annotated scenes and OCR result panels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from .pipeline import PlateResult


DEFAULT_PANEL_HEIGHT = 270
PANEL_MARGIN = 12
PANEL_HEADER_HEIGHT = 34
PANEL_CARD_GAP = 10
PANEL_MIN_CARD_WIDTH = 220
PANEL_BACKGROUND = (24, 26, 33)
PANEL_CARD_BACKGROUND = (42, 45, 55)
PANEL_ACCENT = (0, 210, 255)


def annotate_frame(
    frame: np.ndarray,
    results: list[PlateResult],
    polygon: tuple[tuple[int, int], ...] | None = None,
    consensus_text: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw plate boxes and confidence labels onto one frame.

    Args:
        frame: BGR image array modified in place.
        results: Plate results produced for this frame.

    Returns:
        The same frame object, after drawing each result.

    Raises:
        ImportError: If OpenCV is not installed when rendering is requested.
    """
    cv2 = _load_cv2()
    if polygon is not None:
        for index, start in enumerate(polygon):
            cv2.line(
                frame,
                start,
                polygon[(index + 1) % len(polygon)],
                (255, 0, 0),
                2,
            )
    for result in results:
        x1, y1, x2, y2 = result.bbox
        detection_label = f"D:{result.detection_confidence:.2f}"
        ocr_label = (
            "O:n/a"
            if result.ocr_confidence is None
            else f"O:{result.ocr_confidence:.2f}"
        )
        region_label = f" [{result.region}]" if result.region else ""
        track_id = getattr(result, "track_id", None)
        text = (consensus_text or {}).get(track_id, result.text) if track_id is not None else result.text
        track_label = f" T#{track_id}" if track_id is not None else ""
        label = f"{text or 'UNKNOWN'}{track_label} {detection_label} {ocr_label}{region_label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_origin = (x1, max(20, y1 - 8))
        cv2.putText(
            frame,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            getattr(cv2, "LINE_AA", 16),
        )
    return frame


def compose_plate_panel(
    frame: np.ndarray,
    results: list[PlateResult],
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    scene: np.ndarray | None = None,
    processing_fps: float | None = None,
    source_fps: float | None = None,
) -> np.ndarray:
    """Add a fixed-height plate crop and OCR summary panel above a frame.

    Args:
        frame: Original BGR image used as the source for clean plate crops.
        results: Plate results whose crops and readings should be summarized.
        panel_height: Height in pixels reserved above the scene.
        scene: Optional annotated scene to place below the panel. When omitted,
            ``frame`` is used.
        processing_fps: Smoothed completed-frame processing rate, if available.
        source_fps: FPS reported by the video or camera source, if available.

    Returns:
        A new image with the panel above the unchanged scene. Invalid boxes
        are omitted from the panel without interrupting frame processing.

    Raises:
        ValueError: If ``panel_height`` is not positive or the frame shape is
            not a valid image shape.
    """
    cv2 = _load_cv2()
    import numpy as np

    if panel_height <= 0:
        raise ValueError("panel_height must be positive")
    try:
        height, width = (int(value) for value in frame.shape[:2])
        channels = int(frame.shape[2])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("frame must be a valid HxWxC image") from error
    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError("frame must be a valid HxWxC image")

    panel = np.zeros((panel_height, width, channels), dtype=frame.dtype)
    if hasattr(panel, "__setitem__"):
        panel[:] = PANEL_BACKGROUND
    _draw_panel_header(cv2, panel, width, processing_fps, source_fps)
    valid_results = []
    for result in results:
        crop = getattr(result, "crop", None)
        if crop is None:
            crop = _safe_crop(frame, result.bbox)
        if crop is not None:
            valid_results.append((result, crop))

    if not valid_results:
        cv2.putText(
            panel,
            "No plates detected",
            (PANEL_MARGIN, min(panel_height - PANEL_MARGIN, PANEL_HEADER_HEIGHT + 42)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            getattr(cv2, "LINE_AA", 16),
        )
    else:
        visible_results = valid_results[:5]
        columns = 5
        card_width = max(
            1,
            (width - 2 * PANEL_MARGIN - PANEL_CARD_GAP * (columns - 1)) // columns,
        )
        card_height = max(
            1,
            panel_height - PANEL_HEADER_HEIGHT - PANEL_MARGIN,
        )
        draw_rectangle = getattr(cv2, "rectangle", None)
        hidden_count = len(valid_results) - len(visible_results)
        if hidden_count:
            cv2.putText(
                panel,
                f"+{hidden_count} more",
                (max(PANEL_MARGIN, width - 92), 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                getattr(cv2, "LINE_AA", 16),
            )
        for index, (result, crop) in enumerate(visible_results, start=1):
            card_index = index - 1
            x = PANEL_MARGIN + card_index * (card_width + PANEL_CARD_GAP)
            y = PANEL_HEADER_HEIGHT
            if draw_rectangle is not None:
                draw_rectangle(
                    panel,
                    (x, y),
                    (x + card_width, y + card_height),
                    PANEL_CARD_BACKGROUND,
                    -1,
                )
            thumbnail_width = max(1, card_width - 2 * PANEL_MARGIN)
            thumbnail_height = max(1, card_height - 76)
            thumbnail = _fit_thumbnail(
                cv2, crop, thumbnail_width, thumbnail_height
            )
            thumbnail_x = x + (card_width - thumbnail.shape[1]) // 2
            thumbnail_y = y + PANEL_MARGIN
            panel[
                thumbnail_y : thumbnail_y + thumbnail.shape[0],
                thumbnail_x : thumbnail_x + thumbnail.shape[1],
            ] = thumbnail
            text = result.text or "UNKNOWN"
            confidence = getattr(result, "consensus_confidence", None)
            if confidence is None:
                confidence = getattr(result, "ocr_confidence", None)
            ocr = "n/a" if confidence is None else f"{confidence:.0%}"
            text_x = x + PANEL_MARGIN
            track_id = getattr(result, "track_id", None)
            cv2.putText(
                panel,
                text,
                (text_x, y + card_height - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                getattr(cv2, "LINE_AA", 16),
            )
            cv2.putText(
                panel,
                f"T#{track_id if track_id is not None else index}  OCR {ocr}  "
                f"DET {result.detection_confidence:.0%}",
                (text_x, y + card_height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                PANEL_ACCENT,
                1,
                getattr(cv2, "LINE_AA", 16),
            )

    return np.vstack((panel, frame if scene is None else scene))


def _draw_panel_header(
    cv2: Any,
    panel: np.ndarray,
    width: int,
    processing_fps: float | None,
    source_fps: float | None,
) -> None:
    """Draw the panel title and optional throughput indicators."""
    cv2.putText(
        panel,
        "RECENT DETECTIONS",
        (PANEL_MARGIN, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        PANEL_ACCENT,
        1,
        getattr(cv2, "LINE_AA", 16),
    )
    if processing_fps is None and source_fps is None:
        return
    process_label = "--.-" if processing_fps is None else f"{processing_fps:.1f}"
    source_label = "--.-" if source_fps is None else f"{source_fps:.1f}"
    metrics = f"PROCESS {process_label} FPS | SOURCE {source_label} FPS"
    get_text_size = getattr(cv2, "getTextSize", None)
    if get_text_size is None:
        text_width = len(metrics) * 8
    else:
        text_width = get_text_size(
            metrics, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )[0][0]
    cv2.putText(
        panel,
        metrics,
        (max(PANEL_MARGIN, width - PANEL_MARGIN - text_width), 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        getattr(cv2, "LINE_AA", 16),
    )


def _safe_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Clip a bbox to the frame and return a non-empty crop when possible."""
    try:
        height, width = (int(value) for value in frame.shape[:2])
        x1, y1, x2, y2 = (int(value) for value in bbox)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if getattr(crop, "size", 0) else None


def _fit_thumbnail(cv2: Any, crop: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize a crop into a panel slot while preserving its aspect ratio."""
    crop_height, crop_width = crop.shape[:2]
    scale = min(max_width / crop_width, max_height / crop_height)
    size = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def _load_cv2() -> Any:
    """Import OpenCV only when a frame actually needs rendering."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise ImportError("opencv-python is required for frame annotation") from error
    return cv2
