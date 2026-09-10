"""Visual QA tool for YOLO26 + BoT-SORT person tracking without counting."""

import argparse
from pathlib import Path

import cv2

from detect_people import draw_lanes, load_camera_config
from src.detector import PersonDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run person tracking without lane counting.")
    parser.add_argument("--camera", choices=["turnstile-a", "turnstile-b"], required=True, help="Camera ROI configuration to display.")
    parser.add_argument("--source", default=None, help="Optional video override; defaults to the camera calibration source.")
    parser.add_argument("--config", type=Path, default=Path("config/rois.json"), help="ROI configuration JSON path.")
    parser.add_argument("--model", default="yolo26l.pt", help="YOLO26 weights path or name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum person-detection confidence.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size; higher improves small-person tracking.")
    parser.add_argument("--tracker", default="config/turnstile_botsort.yaml", help="Ultralytics tracker configuration.")
    parser.add_argument("--windowed", action="store_true", help="Do not maximize the preview window.")
    return parser.parse_args()


def draw_tracking_banner(frame, detected: int, tracked: int, tracker: str) -> None:
    cv2.rectangle(frame, (18, 18), (430, 112), (255, 255, 255), -1)
    cv2.rectangle(frame, (18, 18), (430, 112), (0, 140, 255), 2)
    cv2.putText(frame, "YOLO26 PERSON TRACKING", (34, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detected: {detected}  |  Track IDs: {tracked}", (34, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{tracker.replace('.yaml', '').upper()} · counting disabled", (34, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if not 0 < args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1.")

    camera = load_camera_config(args.config, args.camera)
    source = args.source or camera["source"]
    detector = PersonDetector(args.model, args.conf, args.imgsz)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    window_name = f"Person tracking - {args.camera}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Tracking-only mode: {args.camera} with {args.tracker}. Press Q to close.")
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break

        results = detector.track(frame, args.tracker)
        boxes = results.boxes
        detected = len(boxes) if boxes is not None else 0
        tracked = len(boxes.id) if boxes is not None and boxes.id is not None else 0
        annotated = results.plot()
        draw_lanes(annotated, camera["lanes"])
        draw_tracking_banner(annotated, detected, tracked, args.tracker)
        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
