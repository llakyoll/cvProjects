"""Interactive per-lane ROI calibration for two turnstile camera videos.

This is deliberately a calibration-only tool. It does not detect, track, or
count people; it only stores lane polygons for the next project phase.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CONFIG = Path("config/rois.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select fixed turnstile lane ROIs for two camera videos.")
    parser.add_argument("--source-a", required=True, help="Representative video for turnstile A.")
    parser.add_argument("--source-b", required=True, help="Representative video for turnstile B.")
    parser.add_argument("--camera-a", default="turnstile-a", help="Stable config key for the first camera.")
    parser.add_argument("--camera-b", default="turnstile-b", help="Stable config key for the second camera.")
    parser.add_argument("--seek-seconds", type=float, default=5.0, help="Frame timestamp used for calibration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path of the saved ROI JSON configuration.")
    parser.add_argument("--replace", action="store_true", help="Replace an existing ROI configuration without asking.")
    return parser.parse_args()


def read_calibration_frame(source: str, seek_seconds: float):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    capture.set(cv2.CAP_PROP_POS_MSEC, max(seek_seconds, 0) * 1000)
    ok, frame = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if not ok:
        raise RuntimeError(f"Could not read a frame at {seek_seconds:g}s from: {source}")
    return frame, {"width": width, "height": height, "fps": fps}


def select_lanes(camera_name: str, source: str, seek_seconds: float) -> dict:
    frame, video_info = read_calibration_frame(source, seek_seconds)
    window_name = f"ROI calibration - {camera_name}"
    polygons: list[list[tuple[int, int]]] = []
    active_polygon: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = None

    def on_mouse(event, x, y, _flags, _params) -> None:
        nonlocal cursor
        if event == cv2.EVENT_LBUTTONDOWN:
            active_polygon.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            cursor = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Display one frame before registering the callback. This avoids the Qt6
    # null-handler failure raised by cv2.selectROIs on some Wayland sessions.
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        preview = frame.copy()
        cv2.putText(
            preview,
            "Left click: add vertex | ENTER: finish booth | SPACE: finish camera | U: undo | R: reset | Q/ESC: abort",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
        for index, polygon in enumerate(polygons, start=1):
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [points], True, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, f"GISE {index}", polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        if active_polygon:
            active_points = np.array(active_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [active_points], False, (255, 255, 255), 1, cv2.LINE_AA)
            for point in active_polygon:
                cv2.circle(preview, point, 4, (255, 255, 255), -1)
            if cursor is not None:
                cv2.line(preview, active_polygon[-1], cursor, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(active_polygon) >= 3:
            polygons.append(active_polygon.copy())
            active_polygon.clear()
        if key == 32:
            if len(active_polygon) >= 3:
                polygons.append(active_polygon.copy())
                active_polygon.clear()
            if polygons:
                break
        if key in (ord("u"), ord("U")):
            if active_polygon:
                active_polygon.pop()
            elif polygons:
                polygons.pop()
        if key in (ord("r"), ord("R")):
            polygons.clear()
            active_polygon.clear()
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError(f"ROI selection cancelled for {camera_name}.")

    cv2.destroyWindow(window_name)
    if len(polygons) == 0:
        raise RuntimeError(f"No lane ROI selected for {camera_name}.")

    lanes = []
    for index, polygon in enumerate(polygons, start=1):
        lanes.append({
            "id": f"gise-{index}",
            "polygon": [[int(x), int(y)] for x, y in polygon],
        })
    if not lanes:
        raise RuntimeError(f"Only empty lane polygons were selected for {camera_name}.")

    print(f"{camera_name}: saved {len(lanes)} gişe ROI(s)")
    for lane in lanes:
        print(f"  {lane['id']}: {lane['polygon']}")
    return {
        "source": str(Path(source)),
        "calibration_time_seconds": seek_seconds,
        "video": video_info,
        "lanes": lanes,
    }


def save_config(config_path: Path, cameras: dict[str, dict], replace: bool) -> None:
    if config_path.exists() and not replace:
        raise FileExistsError(
            f"{config_path} already exists. Review it or re-run with --replace to overwrite it."
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cameras": cameras,
    }
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nFixed ROI configuration written to: {config_path}")


def main() -> None:
    args = parse_args()
    if args.seek_seconds < 0:
        raise ValueError("--seek-seconds cannot be negative.")
    if args.camera_a == args.camera_b:
        raise ValueError("--camera-a and --camera-b must be different.")

    cameras = {
        args.camera_a: select_lanes(args.camera_a, args.source_a, args.seek_seconds),
        args.camera_b: select_lanes(args.camera_b, args.source_b, args.seek_seconds),
    }
    save_config(args.config, cameras, args.replace)


if __name__ == "__main__":
    main()
