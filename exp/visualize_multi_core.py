#!/usr/bin/env python3
"""
visualize.py
============
A simple, self-contained script to load and visualize "temporal
segmentation" clip bundles (video_left.mp4, video_right.mp4, frame_ts.json,
hand_boxes.json, vio_pose.json, meta.json).

For each clip it produces:
  1. An annotated MP4 (left-eye video with hand boxes drawn on top,
     color-coded left/right, with a confidence score shown).
  2. A PNG figure with the VIO camera trajectory (top-down path,
     height-over-time, and speed-over-time).

HOW TO USE
----------
1. Put all 6 files for one clip in a single folder (any filenames are fine,
   as long as they *end with* the usual suffixes, e.g. "..._meta.json",
   "..._hand_boxes.json", "..._video_left.mp4", etc.).
2. Run the script with either a single clip folder or a parent folder that
   contains many clip folders.

       python3 visualize.py /path/to/clip_folder /path/to/output_folder

   or for batch processing:

       python3 visualize.py /path/to/parent_folder /path/to/output_folder

   Each clip gets its own output subfolder under the output root.

3. Optional flags:

       --max-frames N     cap annotated frames per clip (quick preview)
       --workers N        how many clips to process at once in parallel
                           (default: all CPU cores minus one). Pass 1 to
                           force plain sequential processing.

BATCH PROCESSING IS PARALLEL
-----------------------------
Each clip is fully independent (its own video decode/encode + its own
plot), so when there's more than one clip to do, this script processes
several clips *at the same time* in separate OS processes -- one clip per
CPU core, roughly. This is real multi-core parallelism (not just
threads fighting over Python's GIL), so on an 8/10-core Apple Silicon
Mac you should see close to an 8-10x speedup on a big batch, limited by
however many clips you're actually running.

While parallel batch processing is running, you'll see a live
multi-line dashboard with one progress bar per clip currently in
flight. When only one clip is being processed (a single clip folder,
or --workers 1), you get the simpler single progress bar instead.

Requires: opencv-python, matplotlib, numpy  (all pip-installable)
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
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
MAX_FRAMES = None


# ----------------------------------------------------------------------
# 2. Progress bar rendering (single-clip + multi-clip dashboard)
# ----------------------------------------------------------------------
def format_bar(current: int, total: int, start_time: float, bar_len: int = 24) -> str:
    """Returns one formatted progress-bar line (no leading \\r, no trailing newline)."""
    frac = (current / total) if total else 1.0
    frac = min(max(frac, 0.0), 1.0)
    filled = int(bar_len * frac)
    bar = "#" * filled + "." * (bar_len - filled)

    elapsed = max(time.time() - start_time, 1e-6)
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0

    return (f"[{bar}] {current}/{total} ({frac*100:5.1f}%)  "
            f"{rate:5.1f} fps  ETA {remaining:4.0f}s")


def print_progress(current: int, total: int, start_time: float, bar_len: int = 30):
    """Single-line, self-overwriting progress bar for the sequential / one-clip case."""
    line = "\r  " + format_bar(current, total, start_time, bar_len)
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


class Dashboard:
    """
    Renders one progress line per in-flight clip, redrawn in place using
    ANSI cursor movement. Backed by a multiprocessing.Manager dict so
    worker processes can report progress without their stdout interleaving.
    """

    def __init__(self, clip_ids):
        self.clip_ids = list(clip_ids)
        self._n_lines_drawn = 0

    def render(self, state: dict):
        # Move cursor back up to the top of the previous render, then overwrite.
        if self._n_lines_drawn:
            sys.stdout.write(f"\033[{self._n_lines_drawn}A")

        lines = []
        for cid in self.clip_ids:
            info = state.get(cid)
            if info is None:
                lines.append(f"  {cid:<24s} waiting...")
                continue
            stage = info.get("stage", "")
            if stage == "done":
                lines.append(f"  {cid:<24s} done.")
            elif stage == "error":
                lines.append(f"  {cid:<24s} ERROR: {info.get('error', '')}"[:100])
            elif stage == "trajectory":
                lines.append(f"  {cid:<24s} plotting trajectory...")
            else:  # "annotating" or unset
                current = info.get("current", 0)
                total = info.get("total", 1)
                start = info.get("start", time.time())
                lines.append(f"  {cid:<24s} {format_bar(current, total, start)}")

        for line in lines:
            sys.stdout.write("\033[2K" + line + "\n")  # clear line, then print
        sys.stdout.flush()
        self._n_lines_drawn = len(lines)


# ----------------------------------------------------------------------
# 3. Helpers to find + load each file by its suffix (prefix can be
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
# 4. Overlay hand boxes on the video
# ----------------------------------------------------------------------
def overlay_hand_boxes(video_path: Path, hand_boxes: dict, out_path: Path,
                        max_frames=None, on_progress=None):
    """
    Draws each detected hand box on every frame:
      - green box  = left hand  (class 0)
      - orange box = right hand (class 1)
      - label shows handedness + confidence
    Frames with 3+ boxes (noisy / bystander frames) get a small warning
    tag in the corner so you can spot them at a glance.

    on_progress(current, total, start_time), if given, is called after every
    frame instead of printing directly -- used for the multi-clip dashboard.
    """
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

    if on_progress is None:
        print(f"\nAnnotating {n_frames} frames -> {out_path.name}")

    frame_idx = 0
    start_time = time.time()
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

        if on_progress is not None:
            # Throttle updates a little so we're not hammering the shared dict.
            if frame_idx == n_frames or frame_idx % 5 == 0:
                on_progress(frame_idx, n_frames, start_time)
        else:
            print_progress(frame_idx, n_frames, start_time)

    cap.release()
    writer.release()

    if on_progress is None:
        print(f"Done: wrote {frame_idx} annotated frames.")
    else:
        on_progress(n_frames, n_frames, start_time)


# ----------------------------------------------------------------------
# 5. Plot the VIO camera trajectory
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

    ax = axes[1]
    ax.plot(t, z, color="tab:purple")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("z / height (m)")
    ax.set_title("Camera height over time")

    ax = axes[2]
    ax.plot(t, speed, color="tab:red")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Camera speed over time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 6. Per-clip processing (this is the unit of work each worker runs)
# ----------------------------------------------------------------------
def process_clip(clip_dir: Path, output_root: Path, max_frames=None, on_progress=None):
    meta, frame_ts, hand_boxes, vio_pose, video_left, video_right = load_clip(clip_dir)
    clip_id = meta.get("cid") or clip_dir.name
    output_dir = output_root / clip_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if on_progress is None:
        print(f"\nClip: {clip_id}  |  Task: {meta.get('label')}  |  "
              f"{meta.get('duration_s')}s @ {meta.get('fps')} fps")

    annotated_video_path = output_dir / f"{clip_id}_hands_overlay.mp4"
    overlay_hand_boxes(video_left, hand_boxes, annotated_video_path,
                        max_frames=max_frames, on_progress=on_progress)

    if on_progress is not None:
        on_progress(-1, -1, 0.0, stage="trajectory")

    trajectory_plot_path = output_dir / f"{clip_id}_trajectory.png"
    plot_trajectory(vio_pose, meta, trajectory_plot_path)

    if on_progress is None:
        print(f"\nOutputs:")
        print(f"  - {annotated_video_path}")
        print(f"  - {trajectory_plot_path}")

    return clip_id, str(annotated_video_path), str(trajectory_plot_path)


def _worker_entry(clip_dir_str, output_root_str, max_frames, clip_id, progress_dict):
    """Runs in a separate process. Reports progress into the shared dict instead of stdout."""
    def report(current, total, start_time, stage="annotating"):
        if stage == "trajectory":
            progress_dict[clip_id] = {"stage": "trajectory"}
        else:
            progress_dict[clip_id] = {
                "stage": "annotating", "current": current, "total": total, "start": start_time,
            }

    try:
        process_clip(Path(clip_dir_str), Path(output_root_str), max_frames, on_progress=report)
        progress_dict[clip_id] = {"stage": "done"}
        return clip_id, True, None
    except Exception as exc:  # noqa: BLE001
        err = f"{exc}"
        progress_dict[clip_id] = {"stage": "error", "error": err}
        return clip_id, False, traceback.format_exc()


# ----------------------------------------------------------------------
# 7. Batch driver: parallel across clips using a process pool
# ----------------------------------------------------------------------
def run_batch(clip_dirs, output_root: Path, max_frames, workers: int):
    clip_ids = []
    for cd in clip_dirs:
        try:
            with open(find_file(cd, "meta.json")) as f:
                clip_ids.append(json.load(f).get("cid") or cd.name)
        except Exception:
            clip_ids.append(cd.name)

    print(f"Found {len(clip_dirs)} clip folders. Processing with {workers} parallel worker(s)...\n")

    manager = mp.Manager()
    progress_dict = manager.dict()
    dashboard = Dashboard(clip_ids)
    dashboard.render(progress_dict)

    results = []
    ctx = mp.get_context("spawn")  # safe default on macOS
    with ctx.Pool(processes=workers) as pool:
        async_results = [
            pool.apply_async(
                _worker_entry,
                args=(str(cd), str(output_root), max_frames, cid, progress_dict),
            )
            for cd, cid in zip(clip_dirs, clip_ids)
        ]
        while True:
            dashboard.render(progress_dict)
            if all(r.ready() for r in async_results):
                break
            time.sleep(0.3)
        dashboard.render(progress_dict)

        for r in async_results:
            results.append(r.get())

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"\nBatch complete: {n_ok} succeeded, {n_fail} failed.")
    for clip_id, ok, err in results:
        if not ok:
            print(f"\n--- {clip_id} failed ---\n{err}")


# ----------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------
def is_clip_folder(path: Path) -> bool:
    """A folder counts as a single clip if it directly contains a meta.json-suffixed file."""
    return any(path.glob("*meta.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize one clip, or batch process (in parallel) a folder of clips."
    )
    parser.add_argument("input", nargs="?", default=CLIP_DIR,
                         help="Path to a clip folder or a parent folder containing many clip folders")
    parser.add_argument("output", nargs="?", default=OUTPUT_DIR,
                         help="Path to an output folder where each clip gets its own subfolder")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                         help="Optional cap on how many frames to annotate per clip")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of clips to process in parallel (default: CPU cores - 1). "
                              "Use 1 to force sequential processing with a single progress bar.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        print(f"Input path not found: {input_path}")
        sys.exit(1)

    if is_clip_folder(input_path):
        clip_dirs = [input_path]
    else:
        clip_dirs = [p for p in sorted(input_path.iterdir()) if p.is_dir() and is_clip_folder(p)]
        if not clip_dirs:
            print(f"No clip subfolders found in {input_path} "
                  f"(looked for subfolders containing a *meta.json file).")
            sys.exit(1)

    default_workers = max(1, (os.cpu_count() or 2) - 1)
    workers = args.workers if args.workers is not None else default_workers
    workers = max(1, min(workers, len(clip_dirs)))

    if len(clip_dirs) == 1 or workers == 1:
        # Simple sequential path with the classic single progress bar.
        for clip_dir in clip_dirs:
            try:
                process_clip(clip_dir, output_root, max_frames=args.max_frames)
            except Exception:
                print(f"\nERROR processing {clip_dir}:")
                traceback.print_exc()
    else:
        run_batch(clip_dirs, output_root, args.max_frames, workers)


if __name__ == "__main__":
    main()