"""Interactive tool to click pitch landmarks on one video frame.

The clicked pixels, together with the metric template, define the keyframe
homography of the semi-automatic mode (and held-out landmarks for evaluation).

Typical usage example:

    python scripts/annotate_landmarks.py --video data/raw/clip.mp4 --frame 0
        --output data/annotations/landmarks_clip.json

Keys: click = place the landmark named in the prompt, s = skip it, u = undo,
q = finish. Click at least four (preferably eight) well-spread landmarks.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2

from football_vision.artifacts import save_landmarks_json
from football_vision.datatypes import Frame
from football_vision.video import VideoReader

DEFAULT_LANDMARKS = (
    "corner_tl",
    "corner_tr",
    "corner_br",
    "corner_bl",
    "halfway_top",
    "halfway_bottom",
    "center",
    "left_pa_outer_top",
    "left_pa_outer_bottom",
    "right_pa_outer_top",
    "right_pa_outer_bottom",
)


def read_frame(video: Path, frame_id: int, size: tuple[int, int]) -> Frame:
    """Reads a single frame.

    Args:
        video: Video file.
        frame_id: 0-based frame id.
        size: ``(width, height)`` that the frame is resized to; use the same size as
            the pipeline so that the clicked pixels match.

    Returns:
        The BGR frame.

    Raises:
        ValueError: If the video has no such frame.
    """
    with VideoReader(video, size=size, start=frame_id, stop=frame_id + 1) as reader:
        for _, frame in reader:
            return frame
    raise ValueError(f"{video} has no frame {frame_id}")


def annotate(frame: Frame, names: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Runs the interactive click loop.

    Args:
        frame: BGR frame to annotate.
        names: Landmark names to click, in order.

    Returns:
        Mapping from landmark name to the clicked ``(x, y)`` pixel.
    """
    clicks: dict[str, tuple[float, float]] = {}
    state = {"index": 0}
    window = "annotate landmarks"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and state["index"] < len(names):
            clicks[names[state["index"]]] = (float(x), float(y))
            state["index"] += 1

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while state["index"] < len(names):
        canvas = frame.copy()
        for name, (x, y) in clicks.items():
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                name,
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
        prompt = f"click: {names[state['index']]}   (s skip, u undo, q finish)"
        cv2.putText(
            canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
        )
        cv2.putText(
            canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            state["index"] += 1
        elif key == ord("u") and state["index"] > 0:
            state["index"] -= 1
            clicks.pop(names[state["index"]], None)
    cv2.destroyAllWindows()
    return clicks


def main(argv: Sequence[str] | None = None) -> int:
    """Parses arguments, runs the annotation loop and saves the JSON file.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code: 0 on success, 1 if fewer than four landmarks
        were clicked.
    """
    parser = argparse.ArgumentParser(
        description="Click pitch landmarks on a video frame."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--frame", type=int, default=0, help="0-based frame id (default: 0)."
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Processing width (default: 1280)."
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Processing height (default: 720)."
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=list(DEFAULT_LANDMARKS),
        help="Landmark names to click.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON file to write."
    )
    args = parser.parse_args(argv)
    frame = read_frame(args.video, args.frame, (args.width, args.height))
    clicks = annotate(frame, args.names)
    if len(clicks) < 4:
        print(
            f"Only {len(clicks)} landmark(s) clicked; at least 4 are needed. "
            "Nothing saved."
        )
        return 1
    print(
        f"Saved {len(clicks)} landmarks to {save_landmarks_json(clicks, args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
