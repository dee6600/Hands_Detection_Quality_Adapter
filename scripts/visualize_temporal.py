#!/usr/bin/env python3
"""Milestone 4: visualize stage 3 (temporal rejection) on a real clip.

Runs the real pipeline order (stage 1 -> tracker -> stage 3, using the
clip's own VIO pose as the camera-motion signal) and draws every
stage-1 survivor color-coded by what stage 3 did to it:

  green       solid  = kept (reported/merged, survived all three rules)
  red         dashed = rejected: implausible displacement (rule 4)
  purple      dashed = rejected: unsupported / flicker track (rule 5)
  yellow      dashed = rejected: static background (rule 6)

`apply_stage3` only ever sets a boolean `rejected` tag, so this script
figures out *which* rule caused each rejection by diffing tags before/after
each rule runs in turn (tags are monotonic -- reported/merged -> rejected,
never back) -- this bookkeeping is diagnostic-only and deliberately kept out
of `adapter/temporal.py` itself.

Usage (run from the repo root):

    python scripts/visualize_temporal.py data/0c54a47b_t010 /tmp/temporal_t010.mp4
    python scripts/visualize_temporal.py data/0c54a47b_t010 /tmp/temporal_t010.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.ingest import load_clip
from adapter.temporal import reject_implausible_displacement, reject_static, reject_unsupported
from adapter.types import Config, Tag

_COLOR = {
    "kept": (0, 200, 0),  # green
    "displacement": (0, 0, 255),  # red
    "unsupported": (255, 0, 200),  # purple-ish (BGR)
    "static": (0, 220, 220),  # yellow
}


def _classify_stage3(tracks, pose, config) -> dict[int, str]:
    """Returns {id(detection): reason} for every detection stage 3 rejects."""

    def snapshot():
        return {id(d): d.tag for t in tracks for d in t.detections}

    reasons: dict[int, str] = {}

    before = snapshot()
    reject_implausible_displacement(tracks, config)
    after = snapshot()
    for det_id, tag in after.items():
        if tag == Tag.REJECTED and before[det_id] != Tag.REJECTED:
            reasons[det_id] = "displacement"

    before = after
    reject_unsupported(tracks, config)
    after = snapshot()
    for det_id, tag in after.items():
        if tag == Tag.REJECTED and before[det_id] != Tag.REJECTED:
            reasons.setdefault(det_id, "unsupported")

    before = after
    reject_static(tracks, pose, config)
    after = snapshot()
    for det_id, tag in after.items():
        if tag == Tag.REJECTED and before[det_id] != Tag.REJECTED:
            reasons.setdefault(det_id, "static")

    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", help="clip bundle directory (contains hand_boxes.json, video_left.mp4, ...)")
    ap.add_argument("out_video", help="path to write the annotated mp4 to")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    clip = load_clip(args.clip_dir)
    config = Config()

    stage1_frames = [apply_stage1(list(dets), config) for dets in clip.detections]
    tracks = track_detections(stage1_frames, config)
    reasons = _classify_stage3(tracks, clip.pose, config)

    by_frame: dict[int, list] = defaultdict(list)
    for track in tracks:
        for det in track.detections:
            by_frame[det.frame].append(det)

    left_path = os.path.join(args.clip_dir, "video_left.mp4")
    cap = cv2.VideoCapture(left_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {left_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = len(clip.detections)
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)

    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    counts = {k: 0 for k in _COLOR}
    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        for det in by_frame.get(frame_idx, []):
            category = reasons.get(id(det), "kept")
            counts[category] += 1
            color = _COLOR[category]
            x1, y1, x2, y2 = [int(v) for v in det.xyxy]
            if category == "kept":
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, category, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"wrote {frame_idx} annotated frames -> {args.out_video}")
    print("legend: green=kept  red=displacement  purple=unsupported(flicker)  yellow=static")
    total = sum(counts.values())
    for category, n in counts.items():
        print(f"  {category:14s} {n:6d}  ({n / max(1, total):.1%})")


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


if __name__ == "__main__":
    main()
