# License Plate OCR

GPU-first license-plate detection and recognition for images, videos, RTSP streams, and webcams.

## Demo

<!-- TODO: demo.gif -->

No real demo asset is committed yet.

## Features

- Detect license plates with a pinned YOLOv11 ONNX detector.
- Download and hash-verify detector weights at runtime.
- Recognize cropped plates with FastPlateOCR in batches.
- Process images, videos, RTSP streams, and numeric webcam sources.
- Render plate boxes with detector and OCR confidence labels.
- Show safe plate crops, OCR predictions, and confidence values in a fixed top panel.
- Restrict reported plates to an optional polygon ROI, with interactive drawing when omitted.
- Save annotated image or video output when requested.

## How it works

```mermaid
flowchart LR
    A[Image, video, RTSP, or webcam] --> B[OpenCV capture]
    B --> C[YOLOv11 ONNX plate detection]
    C --> D[Crop and pad detected plates]
    D --> E[FastPlateOCR recognition]
    E --> F[Display or optional annotated output]
```

The detector runs first, then each valid plate crop is passed to FastPlateOCR as a batch. The scene is annotated below a fixed 160-pixel panel containing safe, resized plate crops, plate text, and OCR/detector confidence values for the current frame; when the panel cannot fit every card, it reports the remaining count. When no valid result exists, the panel displays `No plates detected`. The detector artifact is downloaded from the pinned Hugging Face revision when it is missing, and the SHA-256 digest is checked before inference. Each frame is processed independently; this MVP does not track identities across frames.

## Installation

1. Change into the project directory:

   ```bash
   cd license-plate-ocr
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the pinned dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   The pinned requirements are `huggingface-hub==1.30.0`, `pytest==9.1.1`, `ultralytics==8.4.86`, `fast-plate-ocr==1.1.0`, and `onnxruntime-gpu==1.26.0`; this set is intended for Python 3.13. The OCR package is installed without its OpenCV extra because OpenCV and NumPy are supplied by the `cvstack` environment.

   This project environment expects NumPy and OpenCV to be supplied by the
   `cvstack` CMake environment, so they are intentionally not pinned here.
   Verify that `import numpy` and `import cv2` work before running the CLI.

4. Provide an NVIDIA GPU with a compatible driver and CUDA-capable runtime. The default detector device is `cuda:0`, and `onnxruntime-gpu` is installed explicitly for GPU OCR execution. FastPlateOCR 1.1.0 accepts `cuda`, `cpu`, or `auto`; the adapter maps an indexed value such as `cuda:0` to `cuda` for OCR while preserving `cuda:0` for the detector. Confirm that the installed ONNX Runtime and Ultralytics backends can access the intended GPU before processing a source.

The first run needs network access to download the detector into `models/license-plate-finetune-v1l.onnx`. The model file is ignored by Git and is never committed. The application verifies the file against the expected digest both when reusing an existing file and after downloading it; a mismatch stops execution.

The detector artifact is pinned as follows:

| Field | Value |
|---|---|
| Hugging Face repository | [`morsetechlab/yolov11-license-plate-detection`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) |
| Revision | `0f8dc03` |
| Filename | `license-plate-finetune-v1l.onnx` |
| SHA-256 | `5efdfbe4909bfa6c895bed48676b7de695bf71788932e095e7bc74b8b52b75d8` |
| Default local path | `models/license-plate-finetune-v1l.onnx` |

## Usage

Run the CLI from `license-plate-ocr/`.

Process an image and save the annotated result:

```bash
python main.py --source plate.jpg --output plate-annotated.jpg
```

Open a webcam by passing a non-negative numeric source. The CLI converts `0` to webcam index `0`:

```bash
python main.py --source 0
```

Process a video and save an annotated MP4:

```bash
python main.py --source input.mp4 --output annotated.mp4
```

Process an RTSP stream:

```bash
python main.py --source rtsp://user:password@camera.example/stream
```

Limit processing to a rectangular or free-form polygon by supplying integer points separated by semicolons:

```bash
python main.py --source input.mp4 --polygon "100,200;900,200;900,600;100,600" --output annotated.mp4
```

If `--polygon` is omitted, an ROI window opens on the first image/video/camera frame. Left-click to add points, right-click to undo the last point, press `Enter` to confirm at least three points, or `Esc`/`q` to cancel. The preview is capped at 1280 pixels wide and clicks are mapped back to the original frame. The detector still evaluates the full frame; only detections whose bounding-box center is inside or on the polygon boundary are reported and drawn.

If `--output` is omitted, the application displays the annotated frames. Press `q` to stop a video or stream window. Image sources are identified by their supported image suffix; all other string sources are opened through OpenCV video capture. The `--conf` value must be finite and within `[0, 1]`, while `--imgsz` must be a positive integer.

| Argument | Description | Default |
|---|---|---|
| `--source` | Required image path, video path, RTSP URL, or non-negative numeric webcam index. | None |
| `--detector-model` | Local path for the pinned detector ONNX file. | `models/license-plate-finetune-v1l.onnx` |
| `--ocr-model` | FastPlateOCR model name. | `cct-xs-v2-global-model` |
| `--conf` | Minimum YOLO detection confidence, from `0` through `1`. | `0.35` |
| `--imgsz` | Positive YOLO inference image size. | `640` |
| `--device` | Detector device; indexed CUDA values are normalized to FastPlateOCR's accepted `cuda` value for OCR. | `cuda:0` |
| `--output` | Optional annotated image or video output path. | None |
| `--polygon` | Optional ROI points in `x1,y1;x2,y2;...` format; omitted for interactive selection. | None |

This MVP intentionally excludes tracking, counting, and temporal smoothing. A plate can therefore be reported independently on multiple frames, and readings are not stabilized across time.

## Project structure

```text
license-plate-ocr/
├── README.md                 # Setup, usage, model provenance, and limitations
├── requirements.txt          # Pinned Python dependencies
├── main.py                   # CLI and image/video/stream orchestration
├── assets/
│   └── .gitkeep              # Reserved for a future real demo asset
├── models/                   # Runtime detector cache; ONNX weights are ignored
├── src/
│   ├── model_manager.py      # Download and SHA-256 verification
│   ├── ocr.py                # FastPlateOCR adapter
│   ├── pipeline.py           # Detection, crop, and OCR composition
│   ├── plate_detector.py     # YOLO detector adapter
│   ├── types.py              # Shared detection types
│   └── visualization.py      # OpenCV annotations
└── tests/                    # Dependency-light unit and CLI tests
```

## Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics%20YOLO-111F68?style=flat)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?style=flat)
![FastPlateOCR](https://img.shields.io/badge/FastPlateOCR-2E7D32?style=flat)

## License + author footer

The detector model is published under the **AGPL-3.0** license according to its [Hugging Face model card](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/tree/0f8dc03). Review the license obligations before redistributing or deploying an application that uses the weights.

The model card warns that train and test data may overlap, which can make reported metrics optimistic, and that performance is limited on high-resolution imagery. Treat this repository as an MVP and validate the detector and OCR pipeline on representative data before production use.

**Ahmed Akyol** — Computer Vision Engineer
[GitHub](https://github.com/llakyoll) · [LinkedIn](https://www.linkedin.com/in/ahmed-akyol-84766622b/)
