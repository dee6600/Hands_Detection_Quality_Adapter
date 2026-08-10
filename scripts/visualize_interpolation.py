#!/usr/bin/env python3
"""Milestone 5: visualize stage 4 (interpolation + exit) on real clips.

Runs the full real pipeline (`pipeline.run_pipeline`) on one or more real
clips and draws every final detection color-coded by its tag:

  green  solid = kept (reported/merged)
  cyan   solid = interpolated (a gap stage 4 filled back in)
  red    dashed = rejected (any stage 1/3 reason)

A track whose final state is `exiting` (a confirmed border exit -- see
interpolation.py) gets an "EXIT" label drawn at its last real frame, so
exit detection can be checked by eye too.

BATCH BY DEFAULT: with no clip selection given, picks 5 random clips from
the data directory and writes one annotated video per clip. Use --num-clips
to change how many, or --clips to name specific ones.

Usage (run from the repo root):

    # 5 random clips (default) -> ./output/interpolation/
    python scripts/visualize_interpolation.py

    # 10 random clips, capped to 300 frames each for a quick look
    python scripts/visualize_interpolation.py --num-clips 10 --max-frames 300

    # specific clips
    python scripts/visualize_interpolation.py --clips 0c54a47b_t010 407258cd_t036

    # reproducible random sample
    python scripts/visualize_interpolation.py --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from adapter.ingest import load_clip
from adapter.pipeline import run_pipeline
from adapter.types import Tag, TrackState

_COLOR = {
    "kept": (0, 200, 0),  # green
    "interpolated": (255, 220, 0),  # cyan-ish (BGR)
    "rejected": (0, 0, 255),  # red
}


def _category(detection) -> str:
    if detection.tag == Tag.REJECTED:
        return "rejected"
    if detection.tag == Tag.INTERPOLATED:
        return "interpolated"
    return "kept"


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len=8):
    x1, y1 = pt1
    x2, y2 = pt2
    for a, b in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        _draw_dashed_line(img, a, b, color, thickness, dash_len)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
    import math

    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    n_dashes = max(1, int(dist / (dash_len * 2)))
    for i in range(n_dashes):
        start_frac = (2 * i) / (2 * n_dashes)
        end_frac = (2 * i + 1) / (2 * n_dashes)
        sx, sy = x1 + (x2 - x1) * start_frac, y1 + (y2 - y1) * start_frac
        ex, ey = x1 + (x2 - x1) * end_frac, y1 + (y2 - y1) * end_frac
        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), color, thickness)


def process_clip(clip_dir: str, out_video: str, max_frames: int | None = None) -> dict:
    """Runs the real pipeline on one clip and writes an annotated video.
    Returns a stats dict (counts per category, tracks, exits) for the
    caller to print a summary from.
    """
    clip = load_clip(clip_dir)
    tracks = run_pipeline(clip.detections, clip.pose)

    by_frame: dict[int, list] = defaultdict(list)
    exit_markers: dict[int, int] = {}  # frame -> track_id, drawn once at the exit point
    for track in tracks:
        for det in track.detections:
            by_frame[det.frame].append(det)
        if track.state == TrackState.EXITING and track.detections:
            exit_markers[track.detections[-1].frame] = track.track_id

    left_path = os.path.join(clip_dir, "video_left.mp4")
    cap = cv2.VideoCapture(left_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {left_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = len(clip.detections)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    counts = {k: 0 for k in _COLOR}
    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        for det in by_frame.get(frame_idx, []):
            category = _category(det)
            counts[category] += 1
            color = _COLOR[category]
            x1, y1, x2, y2 = [int(v) for v in det.xyxy]
            if category == "rejected":
                _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, category, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        track_id = exit_markers.get(frame_idx)
        if track_id is not None:
            cv2.putText(frame, f"EXIT (track {track_id})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    n_exiting = sum(1 for t in tracks if t.state == TrackState.EXITING)
    total = sum(counts.values())
    return {
        "frames_written": frame_idx,
        "counts": counts,
        "total": total,
        "n_tracks": len(tracks),
        "n_exiting": n_exiting,
    }


def _print_stats(stats: dict) -> None:
    total = stats["total"]
    for category, n in stats["counts"].items():
        print(f"    {category:14s} {n:6d}  ({n / max(1, total):.1%})")
    print(f"    {'tracks exiting':14s} {stats['n_exiting']:6d}  / {stats['n_tracks']} tracks total")


def _discover_clip_ids(data_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
        and os.path.exists(os.path.join(data_dir, name, "video_left.mp4"))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="directory containing clip bundle folders")
    ap.add_argument("--output-dir", default="scripts/output/interpolation",
                     help="directory to write one annotated mp4 per clip into")
    ap.add_argument("--num-clips", type=int, default=5,
                     help="how many random clips to process (ignored if --clips is given)")
    ap.add_argument("--clips", nargs="+", default=None, metavar="CLIP_ID",
                     help="explicit clip ids to process instead of a random sample")
    ap.add_argument("--seed", type=int, default=None, help="random seed, for a reproducible sample")
    ap.add_argument("--max-frames", type=int, default=None, help="cap rendered frames per clip")
    args = ap.parse_args()

    available = _discover_clip_ids(args.data_dir)
    if not available:
        print(f"no clip bundles found under {args.data_dir}")
        sys.exit(1)

    if args.clips:
        missing = [c for c in args.clips if c not in available]
        if missing:
            print(f"unknown clip id(s): {missing}\navailable: {available}")
            sys.exit(1)
        selected = args.clips
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(available, min(args.num_clips, len(available)))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"processing {len(selected)}/{len(available)} clips -> {os.path.abspath(args.output_dir)}")
    print(f"selected: {selected}\n")

    results = []
    start = time.time()
    for i, cid in enumerate(selected, 1):
        clip_dir = os.path.join(args.data_dir, cid)
        out_video = os.path.join(args.output_dir, f"{cid}_interp.mp4")
        print(f"[{i}/{len(selected)}] {cid} ...")
        try:
            stats = process_clip(clip_dir, out_video, max_frames=args.max_frames)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append((cid, None))
            continue
        print(f"  wrote {stats['frames_written']} frames -> {out_video}")
        _print_stats(stats)
        results.append((cid, stats))

    elapsed = time.time() - start
    n_ok = sum(1 for _, s in results if s is not None)
    print(f"\nlegend: green=kept  cyan=interpolated  red dashed=rejected  orange label=confirmed exit")
    print(f"done: {n_ok}/{len(selected)} clips processed in {elapsed:.1f}s -> {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
