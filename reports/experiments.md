# Experiment log

One row per experiment. Every number in the report must be traceable to a row here, a
configuration file and a commit.

| Date | Author | Change | Config | Metric (dev) | Notes |
|---|---|---|---|---|---|
| 2026-11-19 | Team | Base run `base_c004` on `v_HdiyOtliFiw_c004` (1025 frames, CPU) | `configs/fixed_camera.yaml` + `yolov8n.pt@640` (`b8256aa`) | Pipeline full artifacts generated | Upstream base run for A1–A8 (`reports/tables/`) |
| 2026-11-19 | Team (A) | **A1**: Pitch line extraction (`canny` vs `tophat`) | `configs/broadcast.yaml` (`b8256aa`) | `canny`: line_acc=0.360, stable=0.963, 16.53s · `tophat`: line_acc=0.259, stable=0.895, 23.09s | `reports/tables/ablation_A1.csv` |
| 2026-11-19 | Team (B) | **A2**: Homography mode (`fixed` vs `propagate_no_reanchor` vs `propagate_reanchor`) | `configs/fixed_camera.yaml` (`b8256aa` + stages.py fix) | `fixed`: valid=1.00, med=16.68m, p95=37.05m · `prop_no_reanchor`: valid=0.00 · `prop_reanchor`: valid=0.857, med=28.50m, jitter=6690.6px | `reports/tables/ablation_A2.csv`; `fixed` avoids drift without re-anchor fallback |
| 2026-11-19 | Team (B) | **A3**: Homography temporal smoothing (`window_0`, `window_9`, `window_21`) | `configs/fixed_camera.yaml` (`b8256aa`) | All windows: valid=1.00, jitter=0.0px, med=16.68m | `reports/tables/ablation_A3.csv`; identical under `mode=fixed` (smoothing only affects `propagate`) |
| 2026-11-19 | Team (C) | **A4**: Detector size/imgsz (`yolov8{n,s,m}` × `{640,960,1280}`, `stride=5`) | `configs/fixed_camera.yaml` (`b8256aa`) | `n_640`: 34.2fps, mAP50=0.792 · `s_640`: 21.1fps, mAP50=0.822 · `m_1280`: 2.6fps, mAP50=0.837, mAP50-95=0.665 | `reports/tables/ablation_A4.csv`; CPU inference throughput vs mAP trade-off |
| 2026-11-19 | Team (C) | **A5**: Tracker (`sort` vs `bytetrack`) | `configs/fixed_camera.yaml` (`b8256aa`) | `sort`: IDF1=0.251, MOTA=0.658, HOTA=0.293, IDSW=522 · `bytetrack`: IDF1=0.268, MOTA=0.674, HOTA=0.311, IDSW=428 | `reports/tables/ablation_A5.csv`; ByteTrack reduces ID switches by 18.0% |
| 2026-11-19 | Team (C) | **A6**: Tracking space (`image` vs `pitch`) | `configs/fixed_camera.yaml` (`b8256aa`) | `image`: IDF1=0.268, MOTA=0.674, IDSW=428 · `pitch`: IDF1=0.179, MOTA=0.546, IDSW=1045 | `reports/tables/ablation_A6.csv`; fixed-H projection distortion on panning clip hurts pitch-space tracking |
| 2026-11-19 | Team (D) | **A7**: Trajectory smoothing (`savgol_0`, `savgol_15`, `savgol_25`) | `configs/fixed_camera.yaml` (`b8256aa`) | `savgol_0`: stat_dist=40.28 m/min · `savgol_15`: 33.84 m/min · `savgol_25`: 33.13 m/min | `reports/tables/ablation_A7.csv`; SavGol smoothing reduces stationary jitter distance by ~18% |
| 2026-11-19 | Team (C) | **A8**: Team color space (`lab` vs `hsv`) | `configs/fixed_camera.yaml` (`b8256aa`) | `lab`: 3.99s, team_acc=NaN · `hsv`: 4.21s, team_acc=NaN | `reports/tables/ablation_A8.csv`; `team` column unannotated in `v_HdiyOtliFiw_c004_teams.csv` |
