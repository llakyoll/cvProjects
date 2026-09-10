"""Calibrate the two finite confirmation gates inside each turnstile lane ROI."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw two confirmation gates per calibrated turnstile lane.")
    parser.add_argument("--camera", choices=["turnstile-a", "turnstile-b"], required=True, help="Camera to calibrate.")
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


def select_lane_gates(frame, camera_name: str, lane: dict) -> dict[str, list[list[int]]]:
    """Collect two endpoints for gate A, then two endpoints for gate B."""
    window_name = f"Two-gate calibration - {camera_name} - {lane['id']}"
    selected: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = None
    lane_polygon = np.array(lane["polygon"], dtype=np.int32).reshape((-1, 1, 2))

    def on_mouse(event, x, y, _flags, _params) -> None:
        nonlocal cursor
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            cursor = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        preview = frame.copy()
        cv2.polylines(preview, [lane_polygon], True, (0, 200, 255), 3, cv2.LINE_AA)
        display_id = lane["id"].replace("lane-", "GISE-").upper()
        cv2.putText(preview, f"{camera_name} / {display_id} / {lane.get('flow', 'unassigned').upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "First 2 clicks: Gate A (approach side). Next 2: Gate B (opposite side).", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "ENTER: save lane | U: undo | R: reset | Q/ESC: cancel", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if len(selected) >= 2:
            cv2.line(preview, selected[0], selected[1], (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, "GATE A", selected[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        if len(selected) == 4:
            cv2.line(preview, selected[2], selected[3], (0, 80, 255), 3, cv2.LINE_AA)
            cv2.putText(preview, "GATE B", selected[2], cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(selected):
            color = (0, 255, 0) if index < 2 else (0, 80, 255)
            cv2.circle(preview, point, 5, color, -1)
        if len(selected) % 2 == 1 and cursor is not None:
            cv2.line(preview, selected[-1], cursor, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(selected) == 4:
            cv2.destroyWindow(window_name)
            return {
                "a": [[*selected[0]], [*selected[1]]],
                "b": [[*selected[2]], [*selected[3]]],
            }
        if key in (ord("u"), ord("U")) and selected:
            selected.pop()
        if key in (ord("r"), ord("R")):
            selected.clear()
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError(f"Gate calibration cancelled for {camera_name}/{lane['id']}.")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    camera = config["cameras"][args.camera]
    frame = read_calibration_frame(camera)
    for lane in camera["lanes"]:
        lane["gates"] = select_lane_gates(frame, args.camera, lane)
        print(f"Saved gates for {args.camera}/{lane['id']}: {lane['gates']}")

    args.config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved all two-gate calibrations for {args.camera} to: {args.config}")


if __name__ == "__main__":
    main()
