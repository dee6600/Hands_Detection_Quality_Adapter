#!/usr/bin/env python3
"""
visualize_clip.py
==================
A simple, self-contained script to load and visualize one "temporal
segmentation" clip bundle (video_left.mp4, video_right.mp4, frame_ts.json,
hand_boxes.json, vio_pose.json, meta.json).

It produces two things:

  1. An annotated MP4 (left-eye video with hand boxes drawn on top,
     color-coded left/right, with a confidence score shown).
  2. A PNG figure with the VIO camera trajectory (top-down path + a
     speed-over-time plot).

HOW TO USE
----------
1. Put all 6 files for one clip in a single folder (any filenames are fine,
   as long as they *end with* the usual suffixes, e.g. "..._meta.json",
   "..._hand_boxes.json", "..._video_left.mp4", etc. -- that's exactly how
   Claude.ai names your uploads, so you can usually just point this at your
   uploads folder).
2. Edit the two lines below (CLIP_DIR / OUTPUT_DIR) or pass them as
   command-line arguments:

       python3 visualize_clip.py /path/to/clip_folder /path/to/output_folder

3. Run it. Progress is printed to the terminal.

Requires: opencv-python, matplotlib, numpy  (all pip-installable)
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1. EDIT THESE if you're not passing them as command-line arguments
# ----------------------------------------------------------------------
CLIP_DIR = "/mnt/user-data/uploads"
OUTPUT_DIR = "/mnt/user-data/outputs"

# Optional: cap how many frames of video to annotate (None = whole clip).
# The full 90s / 2700-frame clip takes a couple of minutes to encode;
# set this to e.g. 300 for a quick preview first.
MAX_FRAMES = None


# ----------------------------------------------------------------------
# 2. Helpers to find + load each file by its suffix (prefix can be
#    anything, e.g. the random upload id Claude.ai adds)
# ----------------------------------------------------------------------
def find_file(clip_dir: Path, suffix: str) -> Path:
    matches = sorted(clip_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No file ending in '{suffix}' found in {clip_dir}")
    return matches[0]


def load_clip(clip_dir: str):
    clip_dir = Path(clip_dir)
    paths = {
        "meta": find_file(clip_dir, "meta.json"),
        "frame_ts": find_file(clip_dir, "frame_ts.json"),
        "hand_boxes": find_file(clip_dir, "hand_boxes.json"),
        "vio_pose": find_file(clip_dir, "vio_pose.json"),
        "video_left": find_file(clip_dir, "video_left.mp4"),
        "video_right": find_file(clip_dir, "video_right.mp4"),
    }

    print("Found files:")
    for k, v in paths.items():
        print(f"  {k:12s} -> {v.name}")

    with open(paths["meta"]) as f:
        meta = json.load(f)
    with open(paths["frame_ts"]) as f:
        frame_ts = json.load(f)
    with open(paths["hand_boxes"]) as f:
        hand_boxes = json.load(f)
    with open(paths["vio_pose"]) as f:
        vio_pose = json.load(f)

    return meta, frame_ts, hand_boxes, vio_pose, paths["video_left"], paths["video_right"]


# ----------------------------------------------------------------------
# 3. Overlay hand boxes on the video
# ----------------------------------------------------------------------
def overlay_hand_boxes(video_path: Path, hand_boxes: dict, out_path: Path, max_frames=None):
    """
    Draws each detected hand box on every frame:
      - green box  = left hand  (class 0)
      - orange box = right hand (class 1)
      - label shows handedness + confidence
    Frames with 3+ boxes (noisy / bystander frames) get a small warning
    tag in the corner so you can spot them at a glance.
    """
    # Index detections by frame number for fast lookup
    dets_by_frame = {f["frame"]: f["detections"] for f in hand_boxes["frames"]}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = total_frames if max_frames is None else min(total_frames, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    COLOR = {0: (0, 200, 0), 1: (0, 140, 255)}  # BGR: left=green, right=orange
    LABEL = {0: "left", 1: "right"}

    print(f"\nAnnotating {n_frames} frames -> {out_path.name}")
    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        detections = dets_by_frame.get(frame_idx, [])
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            cls = det["class"]
            conf = det["confidence"]
            color = COLOR.get(cls, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{LABEL.get(cls, '?')} {conf:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if len(detections) >= 3:
            cv2.putText(frame, "NOISY FRAME (3+ boxes)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  ...{frame_idx}/{n_frames} frames done")

    cap.release()
    writer.release()
    print(f"Done: wrote {frame_idx} annotated frames.")


# ----------------------------------------------------------------------
# 4. Plot the VIO camera trajectory
# ----------------------------------------------------------------------
def plot_trajectory(vio_pose: dict, meta: dict, out_path: Path):
    t = np.array(vio_pose["t"])
    x = np.array(vio_pose["x"])
    y = np.array(vio_pose["y"])
    z = np.array(vio_pose["z"])
    speed = np.array(vio_pose["speed"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Camera motion — {meta.get('label', '')} "
        f"({meta.get('job', '')}, {meta.get('duration_s', '?')}s)",
        fontsize=13,
    )

    # Top-down path (x vs y), colored by time so you can see direction of travel
    ax = axes[0]
    sc = ax.scatter(x, y, c=t, cmap="viridis", s=4)
    ax.plot(x[0], y[0], "go", markersize=10, label="start")
    ax.plot(x[-1], y[-1], "rs", markersize=10, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Top-down path (total {vio_pose.get('path_length_m', '?')} m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="time (s)")

    # Height over time
    ax = axes[1]
    ax.plot(t, z, color="tab:purple")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("z / height (m)")
    ax.set_title("Camera height over time")

    # Speed over time
    ax = axes[2]
    ax.plot(t, speed, color="tab:red")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Camera speed over time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved trajectory plot -> {out_path.name}")


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    clip_dir = sys.argv[1] if len(sys.argv) > 1 else CLIP_DIR
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, frame_ts, hand_boxes, vio_pose, video_left, video_right = load_clip(clip_dir)

    print(f"\nClip: {meta.get('cid')}  |  Task: {meta.get('label')}  |  "
          f"{meta.get('duration_s')}s @ {meta.get('fps')} fps")

    annotated_video_path = output_dir / f"{meta.get('cid', 'clip')}_hands_overlay.mp4"
    overlay_hand_boxes(video_left, hand_boxes, annotated_video_path, max_frames=MAX_FRAMES)

    trajectory_plot_path = output_dir / f"{meta.get('cid', 'clip')}_trajectory.png"
    plot_trajectory(vio_pose, meta, trajectory_plot_path)

    print("\nAll done! Outputs:")
    print(f"  - {annotated_video_path}")
    print(f"  - {trajectory_plot_path}")


if __name__ == "__main__":
    main()