# Testing & Coverage Guide

All tests are fast, deterministic, and self-contained — they run in ~10–15 seconds using synthetic videos, synthetic pitch geometry, and mocked YOLO predictions (no `data/raw/*.mp4` videos or model weights required).

---

## Quick commands

| Goal | Command (`uv`) | Command (venv python) |
|---|---|---|
| Full suite + branch & HTML coverage | `uv run pytest` | `python -m pytest` |
| Fast run (skip coverage) | `uv run pytest --no-cov` | `python -m pytest --no-cov` |
| Run one test file | `uv run pytest tests/test_homography.py --no-cov` | `python -m pytest tests/test_homography.py --no-cov` |
| Filter tests by keyword | `uv run pytest -k "tracker or stitch" --no-cov` | `python -m pytest -k "tracker or stitch" --no-cov` |
| Fail fast + stop on first error | `uv run pytest -x --no-cov` | `python -m pytest -x --no-cov` |
| Lint + docstrings + coverage suite | `make check` | `python -m ruff check . && python -m pydoclint.main src tests scripts demo && python -m pytest` |

> **Windows Smart App Control note:** If generated `.exe` wrappers inside `venv\Scripts\` are blocked by Application Control, replace `python` with `.\venv\Scripts\python.exe` and invoke `pydoclint` via `python -m pydoclint.main ...`.

---

## Branch Coverage & HTML Report

Coverage settings are configured in `pyproject.toml` under `[tool.pytest.ini_options]`, `[tool.coverage.run]`, and `[tool.coverage.report]`:

```toml
addopts = "-ra --cov=football_vision --cov-branch --cov-report=term-missing --cov-report=html"
```

Running `pytest` without extra flags automatically:
1. Measures both **statement** and **branch** coverage across `src/football_vision/`.
2. Prints a per-module summary table with exact missed lines/branches (`Missing`) to the terminal.
3. Writes an interactive HTML report to `htmlcov/index.html` (git-ignored).

Open the HTML coverage report to inspect uncovered lines/branches visually:

```powershell
# Windows (PowerShell)
start htmlcov\index.html
```

```bash
# macOS / Linux
open htmlcov/index.html   # or: xdg-open htmlcov/index.html
```

---

## Test Suite Layout

| File | Coverage focus |
|---|---|
| `conftest.py`, `helpers.py` | Synthetic video fixture (`synthetic_video`, `base_cfg`) and shared camera helpers |
| `test_contracts.py` | Public API and numerical contracts for core CV algorithms |
| `test_config.py`, `test_artifacts.py`, `test_boxes.py`, `test_video_io.py`, `test_utils_and_frames.py` | Infrastructure, config parsing/overrides, `ArtifactStore` schemas, bbox & frame/video I/O |
| `test_pitch_mask_shots.py`, `test_pitch_lines.py`, `test_template.py` | P1 preprocessing (pitch mask, shot cuts) and P2 pitch lines/intersections/template |
| `test_geometry.py`, `test_homography.py`, `test_homography_baseline.py` | P3 DLT, RANSAC, keyframe estimation, SIFT/ORB/AKAZE propagation, validity, smoothing |
| `test_yolo_wrapper.py`, `test_detection_postprocess.py`, `test_tiling.py`, `test_sportsmot.py` | P4 YOLO wrapper, pitch/size filters, SAHI-style tiled inference, SportsMOT adapter |
| `test_tracking_helpers.py`, `test_team.py`, `test_possession.py` | P5 Kalman/assignment/SORT/ByteTrack/stitching and P6 kit clustering/roles/passes |
| `test_projection.py`, `test_kinematics.py`, `test_heatmap_shape.py`, `test_viz_figures.py` | P7 analytics/projection/smoothing, P8 figures/filmstrip, `make_figures.py` |
| `test_eval.py`, `test_pipeline.py`, `test_coverage_edges.py` | P9 evaluation metrics, full end-to-end `PipelineRunner` stages, ablation runner, edge guards |
| `test_docstrings.py` | Verifies every dataclass attribute and public symbol follows Google docstring conventions |
