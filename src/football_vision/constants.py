"""Project-wide constants: pitch geometry, class ids, labels and conventions.

Conventions used throughout the package:

* Image coordinates are in pixels with the origin at the top-left corner, ``x``
  pointing right and ``y`` pointing down.
* Pitch coordinates are in meters with the origin at the top-left corner of the
  top-down map. ``X`` runs along the pitch length (0 to 105 m) and ``Y`` runs
  along the pitch width (0 to 68 m).
* ``frame_id`` is 0-based and the timestamp of a frame is ``frame_id / fps``.
* A homography always maps image coordinates to pitch coordinates. Its inverse
  maps pitch coordinates back to the image.
"""

from __future__ import annotations

# Pitch geometry (meters), following the IFAB Laws of the Game for a 105 x 68 m pitch.
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0
GOAL_WIDTH_M: float = 7.32
PENALTY_AREA_DEPTH_M: float = 16.5
PENALTY_AREA_WIDTH_M: float = GOAL_WIDTH_M + 2 * PENALTY_AREA_DEPTH_M  # 40.32
GOAL_AREA_DEPTH_M: float = 5.5
GOAL_AREA_WIDTH_M: float = GOAL_WIDTH_M + 2 * GOAL_AREA_DEPTH_M  # 18.32
PENALTY_SPOT_DISTANCE_M: float = 11.0
CENTER_CIRCLE_RADIUS_M: float = 9.15

# Class ids of the COCO label set used by pretrained YOLO models.
COCO_PERSON_ID: int = 0
COCO_SPORTS_BALL_ID: int = 32

# Team and role labels stored in ``tracks.csv`` and ``pitch_xy.csv``.
TEAM_UNKNOWN: int = -1
ROLE_PLAYER: str = "player"
ROLE_GOALKEEPER: str = "goalkeeper"
ROLE_REFEREE: str = "referee"
ROLE_UNKNOWN: str = "unknown"
ROLES: tuple[str, ...] = (ROLE_PLAYER, ROLE_GOALKEEPER, ROLE_REFEREE, ROLE_UNKNOWN)

# Highest speed (m/s) considered physically plausible for a football player.
MAX_PLAUSIBLE_SPEED_MPS: float = 12.0
