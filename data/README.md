# Data

Nothing in this folder is committed except `.gitkeep` files and this README.

```text
data/
├── raw/           # source clips (never commit; record source and license below)
└── annotations/   # ground truth: landmarks, detection boxes, MOT tracks
```

## Clips

Record for every clip: source, license/terms, fps, resolution, duration, and which role it
plays (dev, test, "hard" case). Prefer a fixed or tactical camera for the main clip and a
broadcast clip for the moving-camera experiments. Check the terms of public datasets
(SoccerNet, SportsMOT, ...) before use; some require registration or an NDA.

| Clip | Source | License / terms | fps | Size | Duration | Role |
|---|---|---|---|---|---|---|
| | | | | | | |

## Annotation formats

* **Keyframe landmarks** (`scripts/annotate_landmarks.py`): `{"corner_tl": [x, y], ...}`; names
  come from `PitchTemplate.landmarks()`. Used to estimate the homography.
* **Held-out landmarks** (`scripts/eval_homography.py`): `{"120": {"center": [x, y], ...}, ...}`
  with 30-50 frames. Do **not** reuse landmarks that estimated the homography.
* **Detection ground truth**: CSV with `frame_id, x1, y1, x2, y2, cls`, 100-200 frames.
* **Tracking ground truth**: MOT text file (`frame, id, left, top, width, height, ...`), 10-20 s,
  for example exported from CVAT.

Split dev and test by **clip**, not by frame, to avoid leakage.
