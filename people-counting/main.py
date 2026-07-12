"""Entry point for the people-counting project.

Reads a video source, runs YOLO detection + tracking on each frame, and
counts people entering/exiting a rectangular zone.
"""

import argparse

import cv2

from src.detector import PersonDetector
from src.zone_counter import ZoneCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count people entering/exiting a zone in a video stream.")
    parser.add_argument("--source", required=True, help="Video file path or RTSP/webcam stream URL.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path or name.")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    parser.add_argument("--zone", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), default=[200, 150, 600, 450],
                         help="Rectangular entry/exit zone as x1 y1 x2 y2.")
    parser.add_argument("--output", default=None, help="Optional path to save the annotated output video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = PersonDetector(model_path=args.model, conf_threshold=args.conf)
    counter = ZoneCounter(zone=tuple(args.zone))

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
        zx1, zy1, zx2, zy2 = counter.zone
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 200, 255), 2)
        cv2.putText(annotated, f"Entries: {counter.entries}  Exits: {counter.exits}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if args.output:
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
            writer.write(annotated)
        else:
            cv2.imshow("people-counting", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Total entries: {counter.entries}, Total exits: {counter.exits}")


if __name__ == "__main__":
    main()
