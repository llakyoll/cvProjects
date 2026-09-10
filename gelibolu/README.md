# Turnstile Lane Usage

Calibration and analytics workflow for comparing usage across turnstile lanes.
The intended reporting layer will provide per-lane passage totals, hourly and
daily usage, the busiest lane, and peak-use periods.

## Phase 1 — fixed ROI calibration

This first phase does **not** run person detection or counting. It opens one
representative video for each turnstile camera and lets the operator draw one
polygon per physical lane. Those coordinates are saved as fixed settings for
the later counting phase.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Select ROIs for two turnstiles

Choose one representative clip per camera, ideally with a clear, unobstructed
view of every lane. The tool opens camera A first, then camera B.

```bash
python roi_selector.py \
  --source-a "/path/to/turnstile-a.mp4" \
  --source-b "/path/to/turnstile-b.mp4"
```

For each video:

1. Left-click each corner of one turnstile lane polygon.
2. Press **Enter** to finish that lane.
3. Repeat for every lane visible in the frame, then press **Space** to finish that camera.
4. Press **U** to undo the latest point (or latest completed lane), **R** to clear all lanes, or **Esc** to cancel.

The tool prints every polygon to the terminal and writes the final fixed
coordinates to `config/rois.json`. It will not overwrite an existing file
without `--replace`:

```bash
python roi_selector.py --source-a /path/a.mp4 --source-b /path/b.mp4 --replace
```

### Update one lane ROI

To correct a single lane without recreating every ROI or changing its flow
label, use the lane editor. The old polygon is orange and the new polygon is
cyan while drawing:

```bash
python edit_lane_roi.py --camera turnstile-a --lane lane-3
```

Click the new polygon vertices, press **Enter** to save, **U** to undo a point,
or **R** to restart that polygon. The lane's `entry`, `exit`, or `bidirectional`
metadata remains unchanged.

## Label lane flow

After ROI calibration, identify whether each physical turnstile lane represents
an entry or exit. This information is deliberately operator-provided rather
than inferred from the image:

```bash
python lane_flow_setup.py
```

For every highlighted `camera / lane`, press **E** for `entry`, **X** for
`exit`, or **B** for a two-way (`bidirectional`) lane. Press **Space** to
retain an existing label. These fixed `flow` labels are saved alongside the
lane polygon and will keep entry/exit totals separate in the counting phase.

## Planned next phase

With the ROIs and lane-flow labels fixed, the counting pipeline will track
people and attribute each valid passage to its lane. Bidirectional lanes will
separate entries and exits by detected travel direction. Aggregation will then
calculate hourly and daily entry/exit counts, busiest lane, and peak periods.

## Phase 2 — person detection only

YOLO26L is now integrated for person-only detection. This visual QA command
draws person boxes and the fixed lane polygons, but deliberately does **not**
track people or update counts yet:

```bash
python detect_people.py --camera turnstile-a
python detect_people.py --camera turnstile-b
```

Use `--source /path/to/another-clip.mp4` to test a different clip from the
same camera, and `--conf 0.35` to adjust the minimum detection confidence.

## Phase 3 — person tracking only

BoT-SORT assigns temporary track IDs across consecutive frames while keeping
lane counting disabled. This lets you visually validate ID stability, occlusion
handling, and alignment with the calibrated lane polygons before count logic is
introduced:

```bash
python track_people.py --camera turnstile-a
python track_people.py --camera turnstile-b
```

The preview displays `Track IDs`; this must be greater than zero while people
are visible. The default `config/turnstile_botsort.yaml` uses higher-resolution
YOLO26 inference (`imgsz=1280`), appearance matching, and a 75-frame lost-track
buffer to reduce ID switches around adjacent turnstiles. Use `--imgsz 960` when
speed is more important than tracking accuracy, or `--tracker bytetrack.yaml`
to compare with ByteTrack.

## Phase 4 — two-gate lane passage counting

Each lane needs two finite lines inside its polygon: gate A on the approach
side and gate B on the opposite confirmation side. A track must cross both
gates in sequence to count. This rejects people who merely enter the ROI,
approach a turnstile, or turn back before completing the passage.

Calibrate the gates once for each camera:

```bash
python calibrate_lane_gates.py --camera turnstile-a
python calibrate_lane_gates.py --camera turnstile-b
```

For a single-direction lane, draw gate A on the expected approaching side and
gate B on the opposite side. For a bidirectional lane, either order is valid:
A→B and B→A are classified by the configured entry direction.

Before counting, configure the two-way lanes' on-screen entry movement:

```bash
python bidirectional_direction_setup.py
```

For each highlighted two-way lane, press the arrow key matching the direction
that represents an **entry** in the image. `W`/`A`/`S`/`D` also work as an
arrow-key fallback. Then start the lane counter:

```bash
python count_people.py --camera turnstile-a
python count_people.py --camera turnstile-b
```

To record the annotated preview (person boxes, gişe ROIs, A/B gates, and the
count panel), pass an output path:

```bash
python count_people.py --camera turnstile-b --save-video outputs/turnstile-b-counted.mp4
```

To process and record a video without opening an OpenCV window, add
`--no-display` (a video path is required in this mode):

```bash
python count_people.py --camera turnstile-b \
  --save-video outputs/turnstile-b-counted.mp4 \
  --no-display
```

The counter raises a terminal and on-screen **CROWD ALARM** when a frame has
more than 10 detected people. Change the limit (or set it to `0` to disable
the alarm) with `--crowd-threshold`:

```bash
python count_people.py --camera turnstile-b --crowd-threshold 10
```

The horizontal panel shows only per-lane counts; it does not show an overall
people total. Bidirectional lanes show their entry and exit values separately.
Hourly/daily aggregation is the next reporting phase.
