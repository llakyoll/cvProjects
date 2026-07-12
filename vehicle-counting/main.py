"""Entry point for the vehicle-counting project.

Reads a video source, runs YOLO detection + tracking on each frame, and
counts vehicles that pass through a two-line corridor, split by direction
(in/out).
"""

import argparse

import cv2

from src.counter import CorridorCounter
from src.detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count vehicles passing through a two-line corridor in a video stream.")
    parser.add_argument("--source", required=True, help="Video file path or RTSP/webcam stream URL.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path or name.")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    parser.add_argument("--line1", type=int, default=320, help="Pixel coordinate of the first corridor line.")
    parser.add_argument("--line2", type=int, default=400, help="Pixel coordinate of the second corridor line.")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal",
                         help="Corridor line orientation.")
    parser.add_argument("--output", default=None, help="Optional path to save the annotated output video.")
    return parser.parse_args()


def draw_corridor(frame, line_near: int, line_far: int, orientation: str) -> None:
    h, w = frame.shape[:2]
    for line_position, color in ((line_near, (255, 200, 0)), (line_far, (0, 200, 255))):
        if orientation == "horizontal":
            cv2.line(frame, (0, line_position), (w, line_position), color, 2)
        else:
            cv2.line(frame, (line_position, 0), (line_position, h), color, 2)


def draw_count_panel(frame, count: int) -> None:
    x, y, panel_w, panel_h = 20, 20, 220, 90

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 200, 255), 2)

    cv2.putText(frame, "VEHICLES IN", (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, str(count), (x + 15, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    detector = VehicleDetector(model_path=args.model, conf_threshold=args.conf)
    counter = CorridorCounter(line1=args.line1, line2=args.line2, orientation=args.orientation)

    cap = cv2.VideoCapture(args.source)
    writer = None

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = detector.track(frame)
        boxes = results.boxes
        if boxes.id is not None:
            for box, track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
                x1, y1, x2, y2 = box
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                counter.update(track_id, centroid)

        annotated = results.plot()
        draw_corridor(annotated, counter.line_near, counter.line_far, args.orientation)
        draw_count_panel(annotated, counter.count_in)

        if args.output:
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
            writer.write(annotated)
        else:
            cv2.imshow("vehicle-counting", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Total in: {counter.count_in}")


if __name__ == "__main__":
    main()
