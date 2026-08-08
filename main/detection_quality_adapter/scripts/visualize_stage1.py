#!/usr/bin/env python3
"""Milestone 2: visualize stage 1 (geometric rejection) on a real clip.

Draws every raw detection on the left-eye video, color-coded by what stage 1
did to it, so the three rules can be checked by eye instead of just by
assertion:

  green  solid  = kept untouched (reported)
  blue   solid  = kept, absorbed a duplicate (merged)
  orange dashed = dropped as a duplicate of another box (reject_duplicates)
  red    dashed = dropped for implausible size or shape

Usage (run from `main/detection_quality_adapter/`):

    python scripts/visualize_stage1.py ../../data/0c54a47b_t010 /tmp/out.mp4
    python scripts/visualize_stage1.py ../../data/0c54a47b_t010 /tmp/out.mp4 --max-frames 300
    
    python main/detection_quality_adapter/scripts/visualize_stage1.py data/557419bb_t030 main/detection_quality_adapter/scripts/tmp/out.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from adapter.geometric import apply_stage1
from adapter.ingest import load_clip
from adapter.types import Config, Tag

_COLOR = {
    "kept": (0, 200, 0),  # green
    "merged": (255, 140, 0),  # blue-ish (BGR)
    "dropped_duplicate": (0, 165, 255),  # orange
    "rejected": (0, 0, 255),  # red
}


def _classify(original_dets, survivors) -> list[tuple]:
    """Pairs each original detection with a visualization category."""
    survivor_ids = {id(d) for d in survivors}
    out = []
    for d in original_dets:
        if id(d) in survivor_ids:
            category = "merged" if d.tag == Tag.MERGED else "kept"
        else:
            # rejected size/shape sets tag=REJECTED; a dropped duplicate's
            # tag is left untouched (still REPORTED) since it's the *winner*
            # that gets tagged MERGED, not the loser.
            category = "rejected" if d.tag == Tag.REJECTED else "dropped_duplicate"
        out.append((d, category))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", help="clip bundle directory (contains hand_boxes.json, video_left.mp4, ...)")
    ap.add_argument("out_video", help="path to write the annotated mp4 to")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    clip = load_clip(args.clip_dir)
    config = Config()

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

        # apply_stage1 mutates tags in place; work on copies so re-running
        # this script (or a future frame) isn't affected by prior mutation.
        original_dets = [copy.copy(d) for d in clip.detections[frame_idx]]
        survivors = apply_stage1(original_dets, config)

        for det, category in _classify(original_dets, survivors):
            counts[category] += 1
            color = _COLOR[category]
            x1, y1, x2, y2 = [int(v) for v in det.xyxy]
            dashed = category in ("dropped_duplicate", "rejected")
            if dashed:
                _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{category} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"wrote {frame_idx} annotated frames -> {args.out_video}")
    print("legend: green=kept  blue=merged  orange=dropped_duplicate  red=rejected(size/shape)")
    total = sum(counts.values())
    for category, n in counts.items():
        print(f"  {category:18s} {n:6d}  ({n / max(1, total):.1%})")


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len=8):
    x1, y1 = pt1
    x2, y2 = pt2
    for (a, b) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
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
