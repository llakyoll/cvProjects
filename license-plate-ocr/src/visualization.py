"""OpenCV rendering helpers for annotated scenes and OCR result panels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from .pipeline import PlateResult


DEFAULT_PANEL_HEIGHT = 160
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
        label = f"{result.text or 'UNKNOWN'} {detection_label} {ocr_label}{region_label}"

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
) -> np.ndarray:
    """Add a fixed-height plate crop and OCR summary panel above a frame.

    Args:
        frame: Original BGR image used as the source for clean plate crops.
        results: Plate results whose crops and readings should be summarized.
        panel_height: Height in pixels reserved above the scene.
        scene: Optional annotated scene to place below the panel. When omitted,
            ``frame`` is used.

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
    valid_results = []
    for result in results:
        crop = _safe_crop(frame, result.bbox)
        if crop is not None:
            valid_results.append((result, crop))

    if not valid_results:
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
        max_cards = max(1, (width - 2 * PANEL_MARGIN + PANEL_CARD_GAP) // (
            PANEL_MIN_CARD_WIDTH + PANEL_CARD_GAP
        ))
        visible_results = valid_results[:max_cards]
        card_width = max(
            1,
            (width - 2 * PANEL_MARGIN - PANEL_CARD_GAP * (len(visible_results) - 1))
            // len(visible_results),
        )
        draw_rectangle = getattr(cv2, "rectangle", None)
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
        thumbnail_height = max(1, panel_height - PANEL_HEADER_HEIGHT - 2 * PANEL_MARGIN)
        for index, (result, crop) in enumerate(visible_results, start=1):
            x = PANEL_MARGIN + (index - 1) * (card_width + PANEL_CARD_GAP)
            if draw_rectangle is not None:
                draw_rectangle(
                    panel,
                    (x, PANEL_HEADER_HEIGHT),
                    (x + card_width, panel_height - PANEL_MARGIN),
                    PANEL_CARD_BACKGROUND,
                    -1,
                )
            thumbnail_width = min(max(1, card_width // 2), 112)
            thumbnail = _fit_thumbnail(cv2, crop, thumbnail_width, thumbnail_height)
            thumbnail_y = PANEL_HEADER_HEIGHT + (thumbnail_height - thumbnail.shape[0]) // 2
            panel[
                thumbnail_y : thumbnail_y + thumbnail.shape[0],
                x + PANEL_MARGIN : x + PANEL_MARGIN + thumbnail.shape[1],
            ] = thumbnail
            text = result.text or "UNKNOWN"
            ocr = "n/a" if result.ocr_confidence is None else f"{result.ocr_confidence:.0%}"
            text_x = x + PANEL_MARGIN + thumbnail_width + 8
            cv2.putText(
                panel,
                f"#{index}",
                (text_x, PANEL_HEADER_HEIGHT + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                PANEL_ACCENT,
                1,
                getattr(cv2, "LINE_AA", 16),
            )
            cv2.putText(
                panel,
                text,
                (text_x, PANEL_HEADER_HEIGHT + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                getattr(cv2, "LINE_AA", 16),
            )
            cv2.putText(
                panel,
                f"OCR {ocr}",
                (text_x, PANEL_HEADER_HEIGHT + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (190, 230, 190),
                1,
                getattr(cv2, "LINE_AA", 16),
            )
            cv2.putText(
                panel,
                f"DET {result.detection_confidence:.0%}",
                (text_x, PANEL_HEADER_HEIGHT + 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (210, 210, 210),
                1,
                getattr(cv2, "LINE_AA", 16),
            )

    return np.vstack((panel, frame if scene is None else scene))


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
