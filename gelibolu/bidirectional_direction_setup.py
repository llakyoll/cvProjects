"""Visually choose the entry travel direction for each bidirectional lane."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set entry directions for bidirectional turnstile lanes.")
    parser.add_argument("--config", type=Path, default=Path("config/rois.json"), help="ROI configuration JSON path.")
    return parser.parse_args()


def read_frame(camera: dict):
    capture = cv2.VideoCapture(camera["source"])
    if not capture.isOpened():
        raise RuntimeError(f"Could not open calibration source: {camera['source']}")
    capture.set(cv2.CAP_PROP_POS_MSEC, float(camera.get("calibration_time_seconds", 0)) * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read calibration frame: {camera['source']}")
    return frame


def choose_direction(frame, camera_name: str, lane: dict) -> str:
    window_name = "Set bidirectional entry direction"
    polygon = np.array(lane["polygon"], dtype=np.int32).reshape((-1, 1, 2))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)
    while True:
        preview = frame.copy()
        overlay = preview.copy()
        cv2.fillPoly(overlay, [polygon], (0, 200, 255))
        cv2.addWeighted(overlay, 0.35, preview, 0.65, 0, preview)
        cv2.polylines(preview, [polygon], True, (0, 200, 255), 4, cv2.LINE_AA)
        display_id = lane["id"].replace("lane-", "GISE-").upper()
        cv2.putText(preview, f"{camera_name} / {display_id} - ENTRY travel direction", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "Arrow keys: entry direction  |  Q/ESC: abort", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, preview)
        key = cv2.waitKeyEx(0)
        key_map = {
            82: "up", 84: "down", 81: "left", 83: "right",  # Qt/legacy
            2490368: "up", 2621440: "down", 2424832: "left", 2555904: "right",  # OpenCV extended
            ord("w"): "up", ord("s"): "down", ord("a"): "left", ord("d"): "right",  # fallback
            ord("W"): "up", ord("S"): "down", ord("A"): "left", ord("D"): "right",
        }
        if key in key_map:
            return key_map[key]
        if (key & 0xFF) in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Bidirectional direction setup cancelled.")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bidirectional_lanes = [
        (camera_name, camera, lane)
        for camera_name, camera in config["cameras"].items()
        for lane in camera["lanes"]
        if lane.get("flow") == "bidirectional"
    ]
    if not bidirectional_lanes:
        print("No bidirectional lanes found.")
        return

    cached_frames = {}
    for camera_name, camera, lane in bidirectional_lanes:
        frame = cached_frames.setdefault(camera_name, read_frame(camera))
        lane["entry_direction"] = choose_direction(frame, camera_name, lane)
        print(f"{camera_name}/{lane['id']}: entry direction = {lane['entry_direction']}")

    cv2.destroyAllWindows()
    args.config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved bidirectional directions to: {args.config}")


if __name__ == "__main__":
    main()
