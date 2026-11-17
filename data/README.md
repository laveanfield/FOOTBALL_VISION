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

## Reproducing the SportsMOT clips

The tracking and detection ground truth comes from **SportsMOT** (Cui et al., ICCV
2023, <https://huggingface.co/datasets/MCG-NJU/SportsMOT>), licensed **CC BY-NC 4.0**
(non-commercial use; cite the paper). The videos and labels are never committed; each
member rebuilds them with the steps below (or copies them from the team Drive).

1. On the dataset page, open *Files and versions* and accept the license form.
2. Create a *Read* token at <https://huggingface.co/settings/tokens>, then:

   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login
   ```

3. Download only the train split (6.7 GB; only train and val ship ground truth):

   ```bash
   mkdir -p ../datasets/sportsmot && cd ../datasets/sportsmot
   hf download MCG-NJU/SportsMOT dataset/train.tar splits_txt/football.txt \
       --repo-type dataset --local-dir .
   ```

4. Extract the four football games of the train split, then delete the archive:

   ```bash
   tar -xf dataset/train.tar --wildcards --no-anchored \
       'v_1yHWGw8DH4A_c*' 'v_gQNyhv8y0QY_c*' 'v_HdiyOtliFiw_c*' 'v_iIxMOsCGH58_c013*'
   ```

5. From the repository root, convert the sequences:

   ```bash
   python scripts/prepare_sportsmot.py --root ../datasets/sportsmot \
       --football ../datasets/sportsmot/splits_txt/football.txt
   ```

   This writes `data/raw/<seq>.mp4`, `data/annotations/<seq>_mot.txt`,
   `<seq>_detections.csv` (every 5th frame) and `<seq>_teams.csv`, prints the rows
   for the table above and suggests frames for the held-out landmarks.

6. Click landmarks (needs a display; on WSL this requires WSLg):

   ```bash
   # keyframe used to estimate the homography
   python scripts/annotate_landmarks.py --video data/raw/<seq>.mp4 --frame 0 \
       --output data/annotations/<seq>_landmarks.json
   # held-out frames (use the frame ids printed in step 5), one file per frame
   python scripts/annotate_landmarks.py --video data/raw/<seq>.mp4 --frame 120 \
       --output data/annotations/heldout/<seq>_f120.json
   python scripts/merge_landmarks.py \
       --inputs data/annotations/heldout/<seq>_f*.json \
       --output data/annotations/<seq>_heldout.json
   ```

   Click intersections of pitch lines, at least four well-spread points per frame.
   Never reuse the keyframe for evaluation.

7. Optional (ablation A8): fill the `team` column of `<seq>_teams.csv` with 0 or 1,
   looking at each track at its `first_frame_id`.

Upload the hand-made files (`*_landmarks.json`, `*_heldout.json`, filled
`*_teams.csv`) to the team Drive right away: they cannot be regenerated.

**Limitations to state in the report.** SportsMOT labels players only (no referees,
coaches or ball), so detection precision on these clips is a lower bound. The clips
are broadcast footage with a moving camera, so they use the `keyframe` homography
mode; the fixed-camera (Level 2) clip comes from another source.

## Annotation formats

* **Keyframe landmarks** (`scripts/annotate_landmarks.py`): `{"corner_tl": [x, y], ...}`; names
  come from `PitchTemplate.landmarks()`. Used to estimate the homography.
* **Held-out landmarks** (`scripts/eval_homography.py`): `{"120": {"center": [x, y], ...}, ...}`
  with 30-50 frames. Do **not** reuse landmarks that estimated the homography.
* **Detection ground truth**: CSV with `frame_id, x1, y1, x2, y2, cls`, 100-200 frames.
* **Tracking ground truth**: MOT text file (`frame, id, left, top, width, height, ...`), 10-20 s,
  for example exported from CVAT.

Split dev and test by **clip**, not by frame, to avoid leakage.
