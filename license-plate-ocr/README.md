# License Plate OCR

GPU-first license-plate detection and recognition for images, videos, RTSP streams, and webcams.

## Demo

<!-- TODO: demo.gif -->

No real demo asset is committed yet.

## Features

- Detect license plates with a pinned YOLOv11 PyTorch detector.
- Download and hash-verify detector weights at runtime.
- Display smoothed processing FPS alongside the source FPS.
- Recognize cropped plates with FastPlateOCR in batches.
- Resolve stable plate identities with ByteTrack plus motion and OCR fallback association.
- Show five persistent plate crops with confidence-weighted OCR consensus.
- Restrict reported plates to an optional polygon ROI, with interactive drawing when omitted.

## How it works

```mermaid
flowchart LR
    A[Image, video, RTSP, or webcam] --> B[OpenCV capture]
    B --> C[YOLOv11 PyTorch plate detection]
    C --> D[ByteTrack association]
    D --> E[Crop and FastPlateOCR recognition]
    E --> F[Motion and OCR identity fallback]
    F --> G[Consensus panel and annotated output]
```

The detector runs first, then Ultralytics ByteTrack attempts to associate each plate before valid crops are passed to FastPlateOCR as a batch. An application-level associator preserves backend IDs when available and resolves missing IDs from predicted center movement, normalized box size, and OCR similarity. Each resolved identity accumulates confidence-weighted OCR votes, and the strongest reading and a large crop appear in a fixed 270-pixel panel containing the five most recent records in one horizontal row. The panel also reports exponentially smoothed end-to-end processing FPS beside the source FPS. Association state expires after 30 missed frames, while recent panel records remain visible.

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

4. Provide an NVIDIA GPU with a compatible driver and CUDA-capable PyTorch runtime. The default detector device is `cuda:0`, and `onnxruntime-gpu` is installed explicitly for GPU OCR execution. FastPlateOCR 1.1.0 accepts `cuda`, `cpu`, or `auto`; the adapter maps an indexed value such as `cuda:0` to `cuda` for OCR while preserving `cuda:0` for the detector. Confirm that the installed ONNX Runtime and Ultralytics PyTorch backends can access the intended GPU before processing a source.

The first run needs network access to download the detector into `models/license-plate-finetune-v1l.pt`. The model file is ignored by Git and is never committed. The application verifies the file against the expected digest both when reusing an existing file and after downloading it; a mismatch stops execution.

The detector artifact is pinned as follows:

| Field | Value |
|---|---|
| Hugging Face repository | [`morsetechlab/yolov11-license-plate-detection`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) |
| Revision | `0f8dc03` |
| Filename | `license-plate-finetune-v1l.pt` |
| SHA-256 | `f3d25e066e4ff41c64c2bcbf4fd35fa85abaad5a37769b61c804de8f3291ff2c` |
| Default local path | `models/license-plate-finetune-v1l.pt` |

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
| `--detector-model` | Local path for the pinned detector PyTorch weight. | `models/license-plate-finetune-v1l.pt` |
| `--ocr-model` | FastPlateOCR model name. | `cct-xs-v2-global-model` |
| `--conf` | Minimum YOLO detection confidence, from `0` through `1`. | `0.35` |
| `--imgsz` | Positive YOLO inference image size. | `640` |
| `--device` | Detector device; indexed CUDA values are normalized to FastPlateOCR's accepted `cuda` value for OCR. | `cuda:0` |
| `--output` | Optional annotated image or video output path. | None |
| `--polygon` | Optional ROI points in `x1,y1;x2,y2;...` format; omitted for interactive selection. | None |
| `--debug` | Print backend tracker IDs, resolved application IDs, OCR counts, and panel record counts. | Disabled |

ByteTrack remains the primary identity source. When it returns no ID, the fallback associator prevents successful detections and OCR readings from being discarded before they reach the panel.

## Project structure

```text
license-plate-ocr/
├── README.md                 # Setup, usage, model provenance, and limitations
├── requirements.txt          # Pinned Python dependencies
├── main.py                   # CLI and image/video/stream orchestration
├── assets/
│   └── .gitkeep              # Reserved for a future real demo asset
├── models/                   # Runtime detector cache; detector weights are ignored
├── src/
│   ├── model_manager.py      # Download and SHA-256 verification
│   ├── ocr.py                # FastPlateOCR adapter
│   ├── pipeline.py           # Detection, crop, and OCR composition
│   ├── plate_associator.py   # Stable application-level plate identities
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
