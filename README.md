# Football Vision

Computer Vision system that turns broadcast or tactical football video into
**tracked players on a 2D top-down pitch map** plus tactical analytics (heatmaps,
team shape, distance, speed, and ball possession). Classical CV (pitch masks, Hough
lines, DLT/RANSAC homography, SIFT propagation) is combined with deep-learning
detection (Ultralytics YOLO) and multi-object tracking (SORT / ByteTrack).

All pipeline stages (`P1–P9`) from all four contributors (`A–D`) are **implemented,
tested, and verified** end-to-end.

---

## 1. Installation

This project is configured via modern Python packaging (`pyproject.toml` + `uv.lock`, compatible with Python 3.10–3.12).

### Option A: Using `uv` (Recommended — 10-100x faster)

If you don't have `uv` installed yet, install it in one command:
* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Then simply clone and sync:
```bash
# Full environment: core + YOLO + tracking + Streamlit demo + dev/test tools
uv sync --all-extras

# Or core + dev/test tools only
uv sync
```
*(A `.python-version` pinning `3.12` is included in the repository; `uv` will automatically download and use isolated Python 3.12 even if your machine has an incompatible Python version).*

---

### Option B: Traditional `pip` / `venv` (No `uv` required)

If you prefer standard Python without installing `uv`:
```bash
# 1. Create and activate a virtual environment (Python 3.10 to 3.12)
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 2. Install package with all dependencies directly from pyproject.toml
pip install -e ".[all,dev]"
```

---

> **Windows note:** `pyproject.toml` pins `opencv-python>=4.8,<5.0` because OpenCV
> `5.x` ships unsigned binaries that Windows Smart App Control may block. If
> generated `.exe` shims in `.venv\Scripts\` or `venv\Scripts\` are blocked by
> Application Control, invoke tools via `python -m <tool>` (e.g.
> `.\venv\Scripts\python.exe -m pytest`, `python -m pydoclint.main ...`).

Verify code style, Google docstrings, tests, and coverage:

```bash
uv run ruff check .
uv run python -m pydoclint.main src tests scripts demo
uv run python -m pytest
```

*(On Linux / macOS you can also run `uv run pydoclint ...` and `uv run pytest` directly; on Windows, using `python -m` ensures Windows Smart App Control does not block the unsigned `.exe` shims).*

---

## 2. Instant Interactive Demo (Streamlit)

A lightweight precomputed run (`demo/sample_run/`, generated from `v_HdiyOtliFiw_c004`
frames `675–1025`, ~3 MB total including compressed H.264 `side_by_side.mp4`) is
checked into the repository so the demo works immediately after cloning — even before
downloading full raw videos into `data/raw/`:

```bash
uv run python -m streamlit run demo/streamlit_app.py
```

To inspect your own pipeline run under `outputs/<run_name>/`:

```bash
uv run python -m streamlit run demo/streamlit_app.py -- --run-dir outputs/base_c004_f675
```

If raw source video is present in `data/raw/`, Streamlit renders interactive toggled
layers (`Boxes and ids`, `Pitch lines`, `Top-down map`) per frame; when running from
the bundled sample alone, it shows the rendered MP4 video, matching side-by-side /
top-down frame viewer, and player kinematics table cleanly.

---

## 3. Running the Full Pipeline

See [data/README.md](data/README.md) to download and convert SportsMOT clips into
`data/raw/` and `data/annotations/`.

### Option A: Quick "walking skeleton" (video -> YOLO -> boxes)

```bash
uv run python scripts/run_pipeline.py \
    --config configs/walking_skeleton.yaml \
    --set video.path=data/raw/v_HdiyOtliFiw_c004.mp4
```

### Option B: Fixed or tactical camera (`configs/fixed_camera.yaml`)

1. Annotate pitch landmarks on frame 0 (skip if you already have a landmarks JSON):

   ```bash
   uv run python scripts/annotate_landmarks.py \
       --video data/raw/clip.mp4 --frame 0 \
       --output data/annotations/landmarks_clip.json
   ```

2. Run the pipeline:

   ```bash
   uv run python scripts/run_pipeline.py \
       --config configs/fixed_camera.yaml \
       --set video.path=data/raw/clip.mp4 \
       --set homography.landmarks_path=data/annotations/landmarks_clip.json \
       --stages preprocess pitch_lines homography detect track team project analytics render
   ```

### Option C: Moving broadcast camera (recommended for SportsMOT `c004` `f675–f1025`)

For panning/tilting broadcast clips, use `configs/broadcast.yaml` (`mode=propagate`),
a 7-point keyframe anchor, `stitch=true`, and `yolov8s.pt@960`:

```bash
uv run python scripts/run_pipeline.py \
    --config configs/broadcast.yaml \
    --set run_name=base_c004_f675 \
    --set video.path=data/raw/v_HdiyOtliFiw_c004.mp4 \
    --set video.start_frame=675 \
    --set video.end_frame=1025 \
    --set homography.mode=propagate \
    --set homography.landmarks_path=data/annotations/v_HdiyOtliFiw_c004_landmarks_f675.json \
    --set homography.reanchor_overlay_threshold=0.0 \
    --set homography.reanchor_every_n=0 \
    --set homography.smoothing_window=9 \
    --set detection.weights=yolov8s.pt \
    --set detection.imgsz=960 \
    --set tracking.tracker=bytetrack \
    --set tracking.space=image \
    --set tracking.match_thresh=0.90 \
    --set tracking.stitch=true \
    --set tracking.track_buffer=45 \
    --stages preprocess pitch_lines homography detect track team project analytics render
```

> Outputs are written to `outputs/<run_name>/` (git-ignored at `/outputs/`).
> Completed stages are automatically cached; re-run specific stages with
> `--force detect track` or all stages with `--force`.

---

## 4. Evaluation, Ablations (`A1–A8`) & Figures

Ablation experiments are defined in `src/football_vision/eval/ablation.py` and run on
top of a completed base run, writing CSV tables to `reports/tables/`:

```bash
# Example: A5 (SORT vs ByteTrack)
uv run python scripts/run_ablation.py --ablation A5 \
    --config configs/fixed_camera.yaml \
    --set run_name=base_c004 \
    --set video.path=data/raw/v_HdiyOtliFiw_c004.mp4 \
    --mot-gt data/annotations/v_HdiyOtliFiw_c004_mot.txt \
    --output-dir reports/tables
```

Generate all report figures (`homography_error.png`, `heatmap_*.png`,
`speed_distribution.png`, `team_shape_*.png`, `pipeline_steps_*.png`, `filmstrip.png`)
into `reports/figures/`:

```bash
uv run python scripts/make_figures.py \
    --run-dir outputs/base_c004_f675 \
    --compare-run-dir outputs/base_c004_f675_A7_savgol_0 \
    --frame-id 850 \
    --output-dir reports/figures
```

Standalone evaluation scripts: `scripts/eval_homography.py`,
`scripts/eval_detection.py`, `scripts/eval_tracking.py`. Logged experiment results live
in [reports/experiments.md](reports/experiments.md).

---

## 5. Testing & Branch Coverage

`uv run pytest` automatically runs the entire suite with **branch coverage** enabled
and writes an HTML coverage report to `htmlcov/index.html` (`addopts` in
`pyproject.toml`):

```bash
uv run pytest                                   # full suite + terminal & HTML branch coverage
uv run pytest --no-cov                          # fast test run without coverage instrumentation
uv run ruff format --check . && uv run ruff check . && uv run pydoclint src tests scripts demo
```

See [tests/README.md](tests/README.md) for detailed test commands, filtering, and
suite layout.

---

## 6. Pipeline Stages & Implementation Status

| Stage (`--stages`) | Phase | Package | Main artifacts | Owner / Status |
|---|---|---|---|---|
| `preprocess` | P1 | `preprocessing` | `meta.json`, `pitch_mask/` | (A) Implemented, tested |
| `pitch_lines` | P2 | `pitch` | `pitch_lines.csv` | (A) Implemented, tested |
| `homography` | P3 | `homography` | `homography.npy`, `homography_quality.csv` | (B) Implemented, tested |
| `detect` | P4 | `detection` | `detections.csv` | (C) Implemented, tested |
| `track` | P5 | `tracking` | `tracks.csv` | (C) Implemented, tested |
| `team` | P6 | `team` | `team_labels.csv` (+ team/role in `tracks.csv`) | (C) Implemented, tested |
| `project` | P7 | `analytics` | `pitch_xy.csv` | (D) Implemented, tested |
| `analytics` | P7 | `analytics` | `analytics/` (heatmaps, team shape, kinematics) | (D) Implemented, tested |
| `render` | P8 | `viz` | `outputs/overlay.mp4` or `outputs/side_by_side.mp4` | (D) Implemented, tested |

---

## 7. Repository Layout & Data Contract

```text
football-vision/
├── pyproject.toml      # Project metadata, extras, uv dependency-groups, pytest/ruff/pydoclint/coverage config
├── uv.lock             # Locked dependency graph for reproducible `uv sync`
├── Makefile            # Shortcut targets (`install`, `check`, `test`, `coverage`, `run`)
├── configs/            # YAML: default (reference), walking_skeleton, fixed_camera, broadcast
├── data/               # Raw video + annotations (git-ignored); see data/README.md
├── demo/               # streamlit_app.py + bundled sample_run/ (~3 MB) for instant demo
├── docs/               # Project plan (Vietnamese) and draw.io pipeline diagram
├── reports/            # figures/, tables/ablation_A*.csv, experiments.md
├── scripts/            # run_pipeline, annotate_landmarks, eval_*, run_ablation, make_figures
├── src/football_vision/
│   ├── config.py  artifacts.py  video.py  boxes.py  datatypes.py  constants.py  errors.py  utils.py
│   ├── preprocessing/  pitch/  homography/  detection/  tracking/  team/   # P1-P6
│   ├── analytics/  viz/  eval/                                              # P7-P9
│   └── pipeline/       # Stage classes and PipelineRunner
└── tests/              # Unit, integration, contract tests + tests/README.md
```

* **Image coordinates**: pixels, origin top-left, `x` right, `y` down (default `1280x720`).
* **Pitch coordinates**: meters, origin top-left of the top-down map, `X` in `0–105 m`, `Y` in `0–68 m`.
* `frame_id` is 0-based; time is `frame_id / fps`.
* A homography **always maps image -> pitch**; project **foot points** (bottom-center of box), never box centers.

Schemas are enforced by `ArtifactStore` (`src/football_vision/artifacts.py`). Rules and
docstring style live in [CONTRIBUTING.md](CONTRIBUTING.md).
