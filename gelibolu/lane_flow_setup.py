"""Assign entry or exit flow semantics to calibrated turnstile lane polygons."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label each calibrated turnstile lane as entry or exit.")
    parser.add_argument("--config", type=Path, default=Path("config/rois.json"), help="ROI configuration JSON path.")
    return parser.parse_args()


def read_calibration_frame(camera: dict):
    source = camera["source"]
    seek_seconds = float(camera.get("calibration_time_seconds", 0))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open calibration source: {source}")
    capture.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read calibration frame from: {source}")
    return frame


def choose_flow(frame, camera_name: str, lane: dict, lane_index: int, lane_total: int) -> str:
    """Show one lane polygon and return the operator's keyboard label."""
    current = lane.get("flow", "unassigned")
    polygon = np.array(lane["polygon"], dtype=np.int32).reshape((-1, 1, 2))
    window_name = "Label turnstile lane"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)

    while True:
        preview = frame.copy()
        overlay = preview.copy()
        cv2.fillPoly(overlay, [polygon], (0, 140, 255))
        cv2.addWeighted(overlay, 0.35, preview, 0.65, 0, preview)
        cv2.polylines(preview, [polygon], True, (0, 200, 255), 4, cv2.LINE_AA)
        display_id = lane["id"].replace("lane-", "GISE-").upper()
        cv2.putText(preview, f"{camera_name}  |  {display_id}  ({lane_index}/{lane_total})", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, f"Current: {current.upper()}", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "E: ENTRY    X: EXIT    B: BIDIRECTIONAL    SPACE: keep current    Q/ESC: abort", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("e"), ord("E")):
            return "entry"
        if key in (ord("x"), ord("X")):
            return "exit"
        if key in (ord("b"), ord("B")):
            return "bidirectional"
        if key == 32:
            return current
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Lane-flow labelling cancelled.")


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"ROI configuration not found: {args.config}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cameras = config.get("cameras", {})
    if not cameras:
        raise ValueError("No calibrated cameras found in the ROI configuration.")

    print("\nLabel each highlighted lane: E = entry, X = exit, B = bidirectional, Space = keep current.\n")
    for camera_name, camera in cameras.items():
        frame = read_calibration_frame(camera)
        lanes = camera.get("lanes", [])
        for index, lane in enumerate(lanes, start=1):
            lane["flow"] = choose_flow(frame, camera_name, lane, index, len(lanes))

    cv2.destroyAllWindows()

    unassigned = [
        f"{camera_name}/{lane['id']}"
        for camera_name, camera in cameras.items()
        for lane in camera.get("lanes", [])
        if lane.get("flow") not in {"entry", "exit", "bidirectional"}
    ]
    if unassigned:
        print("\nWarning: unassigned lanes remain: " + ", ".join(unassigned))
    else:
        print("\nAll lanes are labelled with a valid flow type.")

    args.config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved lane-flow labels to: {args.config}")


if __name__ == "__main__":
    main()
