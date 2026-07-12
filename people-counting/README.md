# People Counting

Real-time people detection and counting through a two-line corridor.

## Demo

![demo](assets/demo.gif)

## Features

- Detects people with YOLOv8, restricted to the person class.
- Tracks each person across frames to avoid double-counting.
- Counts a person only once they have crossed both corridor lines in
  sequence, filtering out people who merely approach and turn away.
- Counts entries and exits separately.
- Supports horizontal or vertical corridor lines.
- Runs on any video file, RTSP stream, or webcam index.
- Optionally exports an annotated output video.

## How it works

```mermaid
flowchart LR
    A[Video Source] --> B[YOLO Detection]
    B --> C[ByteTrack Tracking]
    C --> D[Two-Line Corridor Counter]
    D --> E[Annotated Output / Counts]
```

Each frame is run through YOLO detection restricted to the person class,
then Ultralytics' built-in ByteTrack tracker assigns a stable id to each
person. The counter tracks each person's progress through two configured
lines: it only registers a count once a track has crossed the near line
and then the far line (an entry), or the far line and then the near line
(an exit). Crossing only one of the two lines does not count.

## Installation

1. Create a virtual environment: `python -m venv .venv && .venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`

## Usage

```bash
python main.py --source rtsp://192.168.1.64/stream2 --line1 320 --line2 400 --orientation horizontal
```

| Argument | Description | Default |
|---|---|---|
| `--source` | Video file path or RTSP/webcam stream URL | *required* |
| `--model` | YOLO weights path or name | `yolov8n.pt` |
| `--conf` | Detection confidence threshold | `0.4` |
| `--line1` | Pixel coordinate of the first corridor line | `320` |
| `--line2` | Pixel coordinate of the second corridor line | `400` |
| `--orientation` | `horizontal` or `vertical` | `horizontal` |
| `--output` | Path to save the annotated output video | `None` (shows a live window) |

## Project structure

```
people-counting/
├── main.py             # CLI entry point, video loop
├── src/
│   ├── detector.py      # YOLO detection + tracking wrapper
│   └── counter.py       # Two-line corridor counting logic
├── requirements.txt
└── assets/
```

## Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics_YOLO-111F68?style=flat)

## License + author

**Ahmed Akyol** — Computer Vision Engineer · [GitHub](https://github.com/llakyoll) · [LinkedIn](https://www.linkedin.com/in/ahmed-akyol-84766622b/)
