"""Count track-confirmed turnstile passages per configured lane."""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2

from detect_people import draw_lanes, load_camera_config
from src.detector import PersonDetector
from src.lane_counter import LanePassageCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count tracked person passages by turnstile lane.")
    parser.add_argument("--camera", choices=["turnstile-a", "turnstile-b"], required=True, help="Camera ROI configuration to count.")
    parser.add_argument("--source", default=None, help="Optional video override; defaults to the configured source.")
    parser.add_argument("--config", type=Path, default=Path("config/rois.json"), help="ROI configuration JSON path.")
    parser.add_argument("--model", default="yolo26l.pt", help="YOLO26 weights path or name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum person-detection confidence.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size; higher improves small-person tracking.")
    parser.add_argument("--tracker", default="config/turnstile_botsort.yaml", help="Ultralytics tracker configuration.")
    parser.add_argument(
        "--crowd-threshold",
        type=int,
        default=10,
        help="Trigger a crowd alarm when a frame contains more detected people than this value (default: 10; 0 disables).",
    )
    parser.add_argument("--windowed", action="store_true", help="Do not maximize the preview window.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without creating an OpenCV preview window; combine with --save-video for background recording.",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        nargs="?",
        const=Path("outputs"),
        default=None,
        help="Record the annotated output. Optionally provide a .mp4/.avi path; without one, a timestamped MP4 is saved in outputs/.",
    )
    return parser.parse_args()


def box_center(box: list[float]) -> tuple[float, float]:
    """Return the detection-box centre used for gate crossing checks."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def display_gise_id(identifier: str) -> str:
    """Use a gişe label while supporting older lane-* configuration files."""
    return identifier.replace("lane-", "GISE-").upper()


class FFmpegVideoWriter:
    """Stream BGR OpenCV frames to FFmpeg for reliable MP4 or AVI output."""

    def __init__(self, output_path: Path, frame, fps: float) -> None:
        """Start an FFmpeg encoder matching the annotated frame dimensions.

        Args:
            output_path: Destination video path ending in `.mp4` or `.avi`.
            frame: First annotated BGR frame, used to obtain dimensions.
            fps: Output frame rate.

        Raises:
            RuntimeError: If FFmpeg is not installed or cannot start.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required for video recording but was not found in PATH.")

        height, width = frame.shape[:2]
        self.output_path = output_path
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self.process.stdin is None:
            raise RuntimeError(f"Could not start FFmpeg output process for: {output_path}")

    def write(self, frame) -> None:
        """Write one BGR frame to the running FFmpeg process."""
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"FFmpeg stopped while recording: {self.output_path}") from error

    def release(self) -> None:
        """Finish encoding and verify that FFmpeg completed successfully."""
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg could not finalize output video: {self.output_path}")


def create_video_writer(output_path: Path, frame, fps: float) -> FFmpegVideoWriter:
    """Create an FFmpeg output writer matched to the annotated frame dimensions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix.lower()
    if extension not in {".mp4", ".avi"}:
        raise ValueError("--save-video must end with .mp4 or .avi")

    writer = FFmpegVideoWriter(output_path, frame, fps)
    print(f"Recording annotated output to: {output_path} ({fps:.2f} FPS, FFmpeg/libx264)")
    return writer


def resolve_output_path(output_path: Path | None, camera_name: str) -> Path | None:
    """Resolve the optional recording flag to a concrete video path.

    Args:
        output_path: Explicit output path or the `outputs` sentinel from a bare
            `--save-video` flag.
        camera_name: Camera configuration name used in auto-generated files.

    Returns:
        A concrete output path, or None when recording is disabled.

    Raises:
        ValueError: If an explicit path has no supported video extension.
    """
    if output_path is None:
        return None
    if output_path == Path("outputs"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return output_path / f"{camera_name}-counted-{timestamp}.mp4"
    if output_path.suffix.lower() not in {".mp4", ".avi"}:
        raise ValueError("--save-video path must end with .mp4 or .avi")
    return output_path


def draw_crowd_alarm(frame, detected_people: int, threshold: int) -> None:
    """Draw a clear alarm banner onto frames exceeding the crowd threshold."""
    width = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 58), (0, 0, 210), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(
        frame,
        f"CROWD ALARM: {detected_people} PEOPLE (LIMIT {threshold})",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_gates(frame, lanes: list[dict]) -> None:
    for lane in lanes:
        gates = lane.get("gates")
        if not gates:
            continue
        for gate_name, color in (("a", (0, 255, 0)), ("b", (0, 80, 255))):
            start, end = gates[gate_name]
            cv2.line(frame, tuple(start), tuple(end), color, 3, cv2.LINE_AA)
            cv2.putText(frame, f"{display_gise_id(lane['id'])} {gate_name.upper()}", tuple(start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


def validate_lane_config(lanes: list[dict]) -> None:
    missing = [lane["id"] for lane in lanes if lane.get("flow") == "bidirectional" and lane.get("entry_direction") not in {"up", "down", "left", "right"}]
    if missing:
        raise ValueError(
            "Bidirectional lanes need an entry direction before counting: "
            + ", ".join(missing)
            + ". Run: python bidirectional_direction_setup.py"
        )
    missing_gates = [lane["id"] for lane in lanes if set(lane.get("gates", {})) != {"a", "b"}]
    if missing_gates:
        raise ValueError(
            "All lanes need two confirmation gates before counting: "
            + ", ".join(missing_gates)
            + ". Run: python calibrate_lane_gates.py --camera <turnstile-a|turnstile-b>"
        )


def draw_count_panel(frame, counter: LanePassageCounter, lanes: list[dict]) -> None:
    """Draw one horizontal card per turnstile lane; no overall people total."""
    height, width = frame.shape[:2]
    orange, charcoal = (0, 140, 255), (45, 45, 45)
    green, red = (0, 160, 0), (0, 80, 255)
    margin, gap, panel_height = 18, 10, 142
    y = height - panel_height - margin
    card_width = max(140, (width - 2 * margin - gap * (len(lanes) - 1)) // len(lanes))

    # Translucent background makes the long panel readable over the video.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - 8), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    for index, lane in enumerate(lanes):
        x = margin + index * (card_width + gap)
        stats = counter.stats[lane["id"]]
        flow = lane.get("flow", "unassigned")
        accent = green if flow == "entry" else red if flow == "exit" else orange
        cv2.rectangle(frame, (x, y), (x + card_width, y + panel_height), (255, 255, 255), -1)
        cv2.rectangle(frame, (x, y), (x + card_width, y + 35), accent, -1)
        cv2.rectangle(frame, (x, y), (x + card_width, y + panel_height), accent, 2)
        cv2.putText(frame, display_gise_id(lane["id"]), (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

        if flow == "bidirectional":
            cv2.putText(frame, f"IN   {stats['entry']}", (x + 12, y + 69), cv2.FONT_HERSHEY_SIMPLEX, 0.62, green, 2, cv2.LINE_AA)
            cv2.putText(frame, f"OUT  {stats['exit']}", (x + 12, y + 101), cv2.FONT_HERSHEY_SIMPLEX, 0.62, red, 2, cv2.LINE_AA)
        else:
            label = "IN" if flow == "entry" else "OUT"
            count = stats["entry"] if flow == "entry" else stats["exit"]
            cv2.putText(frame, label, (x + 12, y + 67), cv2.FONT_HERSHEY_SIMPLEX, 0.52, charcoal, 1, cv2.LINE_AA)
            cv2.putText(frame, str(count), (x + 12, y + 116), cv2.FONT_HERSHEY_SIMPLEX, 1.55, accent, 3, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if args.crowd_threshold < 0:
        raise ValueError("--crowd-threshold cannot be negative.")
    output_path = resolve_output_path(args.save_video, args.camera)
    if args.no_display and output_path is None:
        raise ValueError("--no-display requires --save-video so the background run produces an output video.")
    camera = load_camera_config(args.config, args.camera)
    validate_lane_config(camera["lanes"])
    source = args.source or camera["source"]
    detector = PersonDetector(args.model, args.conf, args.imgsz)
    counter = LanePassageCounter(camera["lanes"])
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    window_name = f"Gise counting - {args.camera}"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if not args.windowed:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    output_writer = None
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = source_fps if source_fps and source_fps > 0 else 25.0
    latest_event = None
    crowd_alarm_active = False
    print(f"Counting mode: {args.camera} with {args.tracker}. Press Q to close.")
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        results = detector.track(frame, args.tracker)
        boxes = results.boxes
        detected_people = len(boxes) if boxes is not None else 0
        crowd_detected = args.crowd_threshold > 0 and detected_people > args.crowd_threshold
        if crowd_detected and not crowd_alarm_active:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] CROWD ALARM: {detected_people} people detected (limit: {args.crowd_threshold})")
        elif crowd_alarm_active and not crowd_detected:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] CROWD ALARM CLEARED: {detected_people} people detected")
        crowd_alarm_active = crowd_detected
        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
                for event in counter.update(track_id, box_center(box)):
                    latest_event = f"{display_gise_id(event['lane_id'])}  {event['flow'].upper()}  ID {event['track_id']}"
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {latest_event}  gates={event['sequence']}")

        annotated = results.plot()
        draw_lanes(annotated, camera["lanes"])
        draw_gates(annotated, camera["lanes"])
        draw_count_panel(annotated, counter, camera["lanes"])
        if crowd_alarm_active:
            draw_crowd_alarm(annotated, detected_people, args.crowd_threshold)
        if output_path is not None:
            if output_writer is None:
                output_writer = create_video_writer(output_path, annotated, output_fps)
            output_writer.write(annotated)
        if not args.no_display:
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    capture.release()
    if output_writer is not None:
        output_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print("Final per-gişe passage counts:")
    for lane_id, stats in counter.stats.items():
        print(f"  {display_gise_id(lane_id)}: entry={stats['entry']}, exit={stats['exit']}")


if __name__ == "__main__":
    main()
