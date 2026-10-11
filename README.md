# Football Vision

Computer Vision system that turns a football video into **tracked players on a
top-down pitch map** plus tactical analytics (heatmaps, team shape, distance and
speed). Classical CV (edges, Hough lines, homography) is combined with pretrained
deep-learning models (YOLO detection, ByteTrack tracking).

This repository is a **code skeleton** generated from the project plan
(`docs/project_plan_vi.md`, Vietnamese) and the system diagram (`docs/pipeline.drawio`,
open it at [draw.io](https://app.diagrams.net)). The infrastructure, I/O, orchestration,
analytics and visualization are implemented and tested; the core CV algorithms are
**stubs with precise docstrings, `TODO(A|B|C|D)` markers and contract tests**, so the
team can split the work and implement them independently.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dl,tracking,demo]"   # or: pip install -r requirements.txt && pip install -e .
make check                                  # ruff + pydoclint + pytest
```

Phase 0 "walking skeleton" (video -> YOLO -> video with boxes):

```bash
python scripts/run_pipeline.py --config configs/walking_skeleton.yaml \
    --set video.path=data/raw/clip.mp4
```

Artifacts and videos land in `outputs/walking_skeleton/`. Ultralytics downloads the YOLO
weights on first use. Stages are cached; re-run one with `--force detect`, or everything
with `--force`.

## Pipeline stages

| Stage (`--stages`) | Phase | Package | Main artifacts | Owner |
|---|---|---|---|---|
| `preprocess` | P1 | `preprocessing` | `meta.json`, `pitch_mask/` | A |
| `pitch_lines` | P2 | `pitch` | `pitch_lines.csv` | A |
| `homography` | P3 | `homography` | `homography.npy`, `homography_quality.csv` | B |
| `detect` | P4 | `detection` | `detections.csv` | C |
| `track` | P5 | `tracking` | `tracks.csv` | C |
| `team` | P6 | `team` | `team_labels.csv` (+ team/role in `tracks.csv`) | C |
| `project` | P7 | `analytics` | `pitch_xy.csv` | D |
| `analytics` | P7 | `analytics` | `analytics/` (heatmaps, team shape, kinematics) | D |
| `render` | P8 | `viz` | `outputs/overlay.mp4` or `outputs/side_by_side.mp4` | D |

Evaluation (P9) lives in `football_vision.eval` and `scripts/eval_*.py`.
Owners A-D follow section 6 of the plan: A = data and pitch geometry, B = homography and
camera, C = detection, tracking and teams, D = analytics, visualization and integration.

## Implementation status

| Area | Status |
|---|---|
| Config, artifacts (data contract), video I/O, logging, boxes, pipeline runner and stages | Implemented, tested |
| `preprocessing.frames`, `pitch.template`, `homography.geometry`, `analytics.*` | Implemented, tested |
| `detection` (YOLO wrapper, filters), `viz.overlay/topdown/render`, `eval` helpers, ablation runner | Implemented, tested |
| OpenCV baselines: `homography.keyframe.homography_from_landmarks`, `propagate_homography` | Implemented, tested |
| `preprocessing.shots`, `preprocessing.pitch_mask`, `pitch.edges/hough/keypoints` | **Stub (A)** |
| `homography.dlt`, `homography.ransac` (estimator), `keyframe` (auto), `propagation` (alignment), `smoothing`, `quality` | **Stub (B)**; DLT/RANSAC have contract tests |
| `detection.tiling`, `tracking` (Kalman, assignment, SORT, ByteTrack, stitching), `team.*`, `eval.detection`, `eval.tracking.evaluate_tracking` | **Stub (C)**; Kalman/assignment have contract tests |
| `viz.figures`, `viz.render.make_filmstrip` | **Stub (D)** |

Find all open work with `grep -rn "TODO(" src`. Each stub raises `NotImplementedError`
and documents its algorithm, arguments and result. Contract tests live in
`tests/test_contracts.py` and are marked `xfail`: remove the marker once you implement the
function so the test guards your code.

## Repository layout

```text
football-vision/
├── configs/            # YAML: default (reference), walking_skeleton, fixed_camera, broadcast
├── data/               # raw video + annotations (git-ignored); see data/README.md
├── demo/               # Streamlit app: streamlit run demo/streamlit_app.py -- --run-dir outputs/<run>
├── docs/               # project plan (Vietnamese) and the draw.io pipeline diagram
├── notebooks/          # exploration only; no core logic
├── reports/            # figures/, tables/, experiments.md
├── scripts/            # run_pipeline, annotate_landmarks, eval_*, run_ablation, make_figures
├── src/football_vision/
│   ├── config.py  artifacts.py  video.py  boxes.py  datatypes.py  constants.py  errors.py  utils.py
│   ├── preprocessing/  pitch/  homography/  detection/  tracking/  team/   # P1-P6
│   ├── analytics/  viz/  eval/                                              # P7-P9
│   └── pipeline/       # Stage classes and PipelineRunner
└── tests/              # unit, integration and contract tests
```

The plan lists the packages directly under `src/`; they are nested in the
`football_vision` package here so that `import football_vision.homography` works after
`pip install -e .`.

## Conventions and data contract

* **Image coordinates**: pixels, origin top-left, `x` right, `y` down (frames are resized to
  `video.size`, default 1280x720).
* **Pitch coordinates**: meters, origin top-left of the top-down map, `X` along the length
  (0-105 m), `Y` along the width (0-68 m).
* `frame_id` is 0-based; time is `frame_id / fps`.
* A homography **always maps image -> pitch**; use the inverse to draw the pitch on a frame.
* Project the **foot point** (bottom-center of the box), never the box center: the
  homography is only valid on the ground plane.

| Artifact | Columns / content |
|---|---|
| `meta.json` | source, fps, width, height, n_frames, shots |
| `pitch_lines.csv` | frame_id, x1, y1, x2, y2, orient, score |
| `homography.npy` + `homography_quality.csv` | (T, 3, 3) matrices; frame_id, valid, n_inliers, reproj_err_m, overlay_score |
| `detections.csv` | frame_id, x1, y1, x2, y2, conf, cls |
| `tracks.csv` | frame_id, track_id, x1, y1, x2, y2, conf, team, role |
| `team_labels.csv` | track_id, team, role (extension of the plan, written by stage `team`) |
| `pitch_xy.csv` | frame_id, track_id, X, Y, team, role, valid |
| `ball.csv` | frame_id, x, y, conf, interpolated (optional Level 4) |

Schemas are enforced by `ArtifactStore`; they are defined in `football_vision/artifacts.py`.

## Configuration

Configuration is a tree of dataclasses (`football_vision/config.py`). Layer YAML files and
override single values from the command line:

```bash
python scripts/run_pipeline.py -c configs/default.yaml -c configs/fixed_camera.yaml \
    --set video.path=data/raw/clip.mp4 --set detection.imgsz=960 --stages preprocess detect
```

Every run stores its resolved `config.yaml`, `timings.json` and `run.log` next to the
artifacts, so results are reproducible.

## Typical workflow

1. Put a clip in `data/raw/` and run the walking skeleton.
2. Click landmarks: `python scripts/annotate_landmarks.py --video <clip> --output data/annotations/landmarks_clip.json`.
3. Run `configs/fixed_camera.yaml` (Level 2) and add `team` to `stages` once P6 is implemented (Level 3).
4. Evaluate: `scripts/eval_homography.py`, `scripts/eval_detection.py`, `scripts/eval_tracking.py`.
5. Compare variants with `scripts/run_ablation.py --ablation A5` (A1-A8 are defined in `eval/ablation.py`).
6. Create report figures with `scripts/make_figures.py` and log experiments in `reports/experiments.md`.

## Development

* `make check` runs `ruff check`, `pydoclint` and `pytest`; `make format` formats the code.
* Code is **English**, typed, and documented with **Google-style docstrings**; see
  [CONTRIBUTING.md](CONTRIBUTING.md) for the rules and the Git workflow.
* CI (`.github/workflows/ci.yml`) runs the same checks on Python 3.10-3.12.

## Data and licensing

Do not commit raw video (copyright, size). `data/README.md` explains the layout and the
annotation formats. Cite every dataset and model in the report. The code is released under
the MIT license (`LICENSE`); replace it if your course requires another one.
