"""Count restroom entrances in a video stream and print a threshold alarm."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import cv2

from src.counter import EntranceCounter
from src.detector import PersonDetector


LINE_CONFIG_PATH = Path(__file__).with_name("line_config.json")
DEFAULT_LINE_START = (623, 424)
DEFAULT_LINE_END = (280, 457)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count restroom entrances and raise a terminal threshold alarm."
    )
    parser.add_argument("--source", required=True, help="MP4/AVI/MOV/MKV video path, RTSP URL, or webcam index.")
    parser.add_argument("--model", default="yolo26l.pt", help="YOLO weights path or name.")
    parser.add_argument("--conf", type=float, default=0.4, help="Minimum detection confidence.")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker configuration.")
    parser.add_argument(
        "--entry-direction", choices=["both", "forward", "backward"], default="both",
        help="Crossing direction relative to the selected first-to-second point line.",
    )
    parser.add_argument("--threshold", type=int, default=2, help="Entry count that triggers the alarm.")
    parser.add_argument("--output", default=None, help="Optional path for annotated output video.")
    parser.add_argument("--windowed", action="store_true", help="Do not maximize the calibration and live windows.")
    parser.add_argument("--redraw-line", action="store_true", help="Draw and replace the saved entrance line.")
    return parser.parse_args()


def video_source(value: str):
    return int(value) if value.isdigit() else value


def open_video(source):
    """Open a source with the default backend, then retry FFmpeg when available."""
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if capture.isOpened():
        return capture

    capture.release()
    raise RuntimeError(
        f"Could not open video source: {source}. "
        "For AVI files, use a valid video stream such as H.264/MJPEG. "
        "H.265 streams in AVI containers are often unsupported; re-encode them to H.264 MP4."
    )


class FFmpegVideoWriter:
    """Write BGR frames as an H.264 MP4 through the system FFmpeg binary."""

    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]):
        width, height = frame_size
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}", "-framerate", str(fps),
                "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", output_path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg video writer is unavailable.")
        self.process.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        error_output = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        if self.process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not write the output video: {error_output.strip()}")


def create_video_writer(output_path: str, fps: float, frame_size: tuple[int, int]):
    """Prefer OpenCV output, then fall back to FFmpeg when its MP4 encoder is absent."""
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if writer.isOpened():
        return writer
    writer.release()
    print("OpenCV MP4 writer is unavailable; using FFmpeg H.264 output.")
    return FFmpegVideoWriter(output_path, fps, frame_size)


def configure_window(window_name: str, windowed: bool) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if not windowed:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw_counting_line(frame, counter: EntranceCounter) -> None:
    cv2.line(frame, counter.start, counter.end, (0, 200, 255), 3)


def foot_point(box: list[float]) -> tuple[float, float]:
    """Return the bottom-center point of a person detection box (foot position)."""
    x1, _y1, x2, y2 = box
    return (x1 + x2) / 2, y2


def load_counting_line() -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Load locally saved endpoints, falling back to the calibrated default."""
    if not LINE_CONFIG_PATH.is_file():
        return DEFAULT_LINE_START, DEFAULT_LINE_END
    try:
        config = json.loads(LINE_CONFIG_PATH.read_text(encoding="utf-8"))
        start, end = config["start"], config["end"]
        if len(start) != 2 or len(end) != 2:
            raise ValueError("Each endpoint must contain x and y coordinates.")
        return (int(start[0]), int(start[1])), (int(end[0]), int(end[1]))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Ignoring invalid saved line configuration: {error}")
        return DEFAULT_LINE_START, DEFAULT_LINE_END


def save_counting_line(start: tuple[int, int], end: tuple[int, int]) -> None:
    """Persist the selected line locally for the next application run."""
    LINE_CONFIG_PATH.write_text(
        json.dumps({"start": start, "end": end}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved entrance line coordinates: start={start}, end={end}")


def select_counting_line(frame, windowed: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select the two end points of one entrance line."""
    window_name = "Select entrance line"
    selected: list[tuple[int, int]] = []
    frozen_frame = frame.copy()

    def on_click(event, x, y, _flags, _params) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(selected) == 2:
            selected.clear()
        selected.append((x, y))

    configure_window(window_name, windowed)
    cv2.setMouseCallback(window_name, on_click)

    while True:
        preview = frozen_frame.copy()
        for point in selected:
            cv2.circle(preview, point, 6, (0, 200, 255), -1)
        if len(selected) == 2:
            cv2.line(preview, selected[0], selected[1], (0, 200, 255), 3)

        instruction = "Click 2 points | Enter: confirm | R: reset | Q: cancel"
        cv2.putText(preview, instruction, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if len(selected) < 2:
            cv2.putText(preview, f"Select point {len(selected) + 1}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(preview, "Line ready - press Enter", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(selected) == 2:
            cv2.destroyWindow(window_name)
            return selected[0], selected[1]
        if key in (ord("r"), ord("R")):
            selected.clear()
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Line setup cancelled.")


def draw_monitor_panel(
    frame, count: int, tracked_persons: int, threshold: int, alarmed: bool, tracker: str
) -> None:
    """Draw a high-contrast orange and white entrance-monitoring panel."""
    orange = (0, 140, 255)
    dark_orange = (0, 95, 190)
    charcoal = (45, 45, 45)
    soft_gray = (235, 235, 235)
    green = (40, 165, 70)
    red = (45, 45, 220)
    x, y, width, height = 24, 24, 360, 235

    # Subtle shadow, white body, and orange title band.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + 6, y + 8), (x + width + 6, y + height + 8), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 58), orange, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), dark_orange, 2)

    cv2.putText(frame, "ENTRANCE MONITOR", (x + 18, y + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "LIVE", (x + width - 70, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "PEOPLE ENTERED", (x + 20, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, charcoal, 1, cv2.LINE_AA)
    cv2.putText(frame, str(count), (x + 20, y + 152), cv2.FONT_HERSHEY_SIMPLEX, 2.0, orange, 4, cv2.LINE_AA)
    cv2.putText(frame, f"/ {threshold}", (x + 118, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, charcoal, 2, cv2.LINE_AA)

    cv2.line(frame, (x + 20, y + 170), (x + width - 20, y + 170), soft_gray, 1)
    cv2.putText(frame, f"TRACKED NOW  {tracked_persons}", (x + 20, y + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.55, charcoal, 1, cv2.LINE_AA)
    cv2.putText(frame, f"TRACKER  {tracker.replace('.yaml', '').upper()}", (x + 20, y + 222), cv2.FONT_HERSHEY_SIMPLEX, 0.48, charcoal, 1, cv2.LINE_AA)

    status_color = red if alarmed else green
    status_text = "ALARM ACTIVE" if alarmed else "MONITORING"
    status_width = 138 if alarmed else 128
    cv2.rectangle(frame, (x + width - status_width - 18, y + height - 39), (x + width - 18, y + height - 16), status_color, -1)
    cv2.putText(frame, status_text, (x + width - status_width - 10, y + height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if args.threshold < 1:
        raise ValueError("--threshold must be at least 1.")

    detector = PersonDetector(args.model, args.conf, args.tracker)
    capture = open_video(video_source(args.source))

    line = None if args.redraw_line else load_counting_line()
    if line is None:
        ok, first_frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError("Could not read the first frame for line setup.")
        start, end = select_counting_line(first_frame, args.windowed)
        save_counting_line(start, end)
    else:
        start, end = line
        print(f"Using fixed entrance line coordinates: start={start}, end={end}")
    counter = EntranceCounter(start, end, args.entry_direction)
    display_window = "restroom-usage-alert"
    configure_window(display_window, args.windowed)

    writer = None
    alarmed = False
    print(f"Monitoring restroom entrances; alarm threshold: {args.threshold}.")

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break

        results = detector.track(frame)
        tracked_persons = 0
        if results.boxes.id is not None:
            tracked_persons = len(results.boxes.id)
            for box, track_id in zip(results.boxes.xyxy.tolist(), results.boxes.id.int().tolist()):
                if counter.update(track_id, foot_point(box)):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] ENTRY COUNTED: {counter.entry_count}/{args.threshold} (track ID: {track_id})")

        if not alarmed and counter.entry_count >= args.threshold:
            alarmed = True
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] ALARM: restroom entry threshold reached ({counter.entry_count}/{args.threshold}).")

        annotated = results.plot()
        draw_counting_line(annotated, counter)
        draw_monitor_panel(
            annotated,
            counter.entry_count,
            tracked_persons,
            args.threshold,
            alarmed,
            args.tracker,
        )

        if args.output:
            if writer is None:
                height, width = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS) or 25
                writer = create_video_writer(args.output, fps, (width, height))
            writer.write(annotated)
        else:
            cv2.imshow(display_window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Final entrance count: {counter.entry_count}")


if __name__ == "__main__":
    main()
