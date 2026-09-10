"""Visual QA tool for YOLO26 person detection in calibrated turnstile videos.

This phase intentionally performs detection only. It does not assign tracking
IDs and does not increment lane or passage counts.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.detector import PersonDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO26 person detection without tracking or counting.")
    parser.add_argument("--camera", choices=["turnstile-a", "turnstile-b"], required=True, help="Camera ROI configuration to display.")
    parser.add_argument("--source", default=None, help="Optional video override; defaults to the camera calibration source.")
    parser.add_argument("--config", type=Path, default=Path("config/rois.json"), help="ROI configuration JSON path.")
    parser.add_argument("--model", default="yolo26l.pt", help="YOLO26 weights path or name.")
    parser.add_argument("--conf", type=float, default=0.4, help="Minimum person-detection confidence.")
    parser.add_argument("--windowed", action="store_true", help="Do not maximize the preview window.")
    return parser.parse_args()


def load_camera_config(config_path: Path, camera_name: str) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"ROI configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        return config["cameras"][camera_name]
    except KeyError as error:
        raise KeyError(f"Camera not found in ROI configuration: {camera_name}") from error


def lane_color(flow: str) -> tuple[int, int, int]:
    return {"entry": (0, 200, 0), "exit": (0, 80, 255), "bidirectional": (0, 200, 255)}.get(flow, (200, 200, 200))


def draw_lanes(frame, lanes: list[dict]) -> None:
    for lane in lanes:
        polygon = np.array(lane["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        color = lane_color(lane.get("flow", "unassigned"))
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        # Hershey's built-in OpenCV font is ASCII-only, hence GISE on video.
        display_id = lane["id"].replace("lane-", "GISE-").upper()
        label = f"{display_id} · {lane.get('flow', 'unassigned').upper()}"
        x, y = lane["polygon"][0]
        cv2.putText(frame, label, (x, max(y - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_detection_banner(frame, people: int, confidence: float) -> None:
    cv2.rectangle(frame, (18, 18), (388, 106), (255, 255, 255), -1)
    cv2.rectangle(frame, (18, 18), (388, 106), (0, 140, 255), 2)
    cv2.putText(frame, "YOLO26 PERSON DETECTION", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detected now: {people}  |  conf >= {confidence:.2f}", (34, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(frame, "Tracking and counting disabled", (34, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if not 0 < args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1.")

    camera = load_camera_config(args.config, args.camera)
    source = args.source or camera["source"]
    detector = PersonDetector(args.model, args.conf)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    window_name = f"Person detection - {args.camera}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Detection-only mode: {args.camera}. Press Q to close.")
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break

        results = detector.detect(frame)
        annotated = results.plot()
        people = len(results.boxes) if results.boxes is not None else 0
        draw_lanes(annotated, camera["lanes"])
        draw_detection_banner(annotated, people, args.conf)
        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
