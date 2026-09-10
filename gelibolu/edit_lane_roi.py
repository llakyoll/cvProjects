"""Replace one calibrated turnstile lane polygon while preserving its metadata."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit one fixed turnstile lane ROI polygon.")
    parser.add_argument("--camera", choices=["turnstile-a", "turnstile-b"], required=True, help="Camera containing the lane.")
    parser.add_argument("--lane", required=True, help="Lane ID to replace, for example lane-3.")
    parser.add_argument("--config", type=Path, default=Path("config/rois.json"), help="ROI configuration JSON path.")
    return parser.parse_args()


def read_calibration_frame(camera: dict):
    capture = cv2.VideoCapture(camera["source"])
    if not capture.isOpened():
        raise RuntimeError(f"Could not open calibration source: {camera['source']}")
    capture.set(cv2.CAP_PROP_POS_MSEC, float(camera.get("calibration_time_seconds", 0)) * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read calibration frame: {camera['source']}")
    return frame


def select_replacement_polygon(frame, camera_name: str, lane: dict) -> list[list[int]]:
    window_name = f"Edit ROI - {camera_name} - {lane['id']}"
    selected: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = None

    def on_mouse(event, x, y, _flags, _params) -> None:
        nonlocal cursor
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            cursor = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        preview = frame.copy()
        # Existing ROIs remain visible for alignment; the target is orange.
        original = np.array(lane["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [original], True, (0, 100, 255), 2, cv2.LINE_AA)
        display_id = lane["id"].replace("lane-", "GISE-").upper()
        cv2.putText(preview, f"OLD {display_id}", tuple(original[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "Left click: vertex | ENTER: save new polygon | U: undo | R: reset | Q/ESC: cancel", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 140, 255), 2, cv2.LINE_AA)

        if selected:
            points = np.array(selected, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [points], False, (0, 200, 255), 2, cv2.LINE_AA)
            for point in selected:
                cv2.circle(preview, point, 4, (0, 200, 255), -1)
            if cursor is not None:
                cv2.line(preview, selected[-1], cursor, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(selected) >= 3:
            cv2.destroyWindow(window_name)
            return [[x, y] for x, y in selected]
        if key in (ord("u"), ord("U")) and selected:
            selected.pop()
        if key in (ord("r"), ord("R")):
            selected.clear()
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("ROI edit cancelled; no configuration was changed.")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    camera = config["cameras"][args.camera]
    lane = next((item for item in camera["lanes"] if item["id"] == args.lane), None)
    if lane is None:
        available = ", ".join(item["id"] for item in camera["lanes"])
        raise ValueError(f"Lane not found: {args.lane}. Available: {available}")

    replacement = select_replacement_polygon(read_calibration_frame(camera), args.camera, lane)
    lane["polygon"] = replacement
    args.config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {args.camera}/{args.lane} polygon with {len(replacement)} vertices.")
    print(f"Flow metadata preserved: {lane.get('flow', 'unassigned')}")


if __name__ == "__main__":
    main()
