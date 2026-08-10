#!/usr/bin/env python3
"""Milestone 3: visualize the tracker (association) on a real clip.

Runs stage 1 (geometric rejection) then the tracker on a real clip's raw
detections, and draws each surviving detection color-coded by its assigned
track ID -- so identity continuity (or a swap/fragmentation bug) can be
checked by eye instead of just by assertion. Each track also gets a fading
trail of its recent centers, and its predicted-vs-actual gap is visible
directly: a short gap should show the SAME color box resume, a track that
should have ended (patience exceeded) should show a NEW color.

Usage (run from the repo root):

    python scripts/visualize_tracks.py data/0c54a47b_t010 /tmp/tracks_t010.mp4
    python scripts/visualize_tracks.py data/0c54a47b_t010 /tmp/tracks_t010.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.ingest import load_clip
from adapter.types import Config

_TRAIL_LEN = 20


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually-distinct BGR color per track id."""
    hue = (track_id * 0.61803398875) % 1.0  # golden-ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


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

    # index: frame_idx -> list of (track_id, Detection) for fast per-frame lookup
    by_frame: dict[int, list[tuple[int, "object"]]] = defaultdict(list)
    for track in tracks:
        for det in track.detections:
            by_frame[det.frame].append((track.track_id, det))

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

    trails: dict[int, list[tuple[float, float]]] = defaultdict(list)
    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id, det in by_frame.get(frame_idx, []):
            color = _color_for_track(track_id)
            x1, y1, x2, y2 = [int(v) for v in det.xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"id {track_id}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            trails[track_id].append(det.center)
            trails[track_id] = trails[track_id][-_TRAIL_LEN:]

        for track_id, trail in trails.items():
            color = _color_for_track(track_id)
            for pt in trail:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, color, -1)

        n_active_ids = len(by_frame.get(frame_idx, []))
        cv2.putText(frame, f"frame {frame_idx}  tracks this frame: {n_active_ids}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"wrote {frame_idx} annotated frames -> {args.out_video}")
    print(f"{len(tracks)} tracks total")
    lifespans = sorted(
        ((t.track_id, t.detections[0].frame, t.detections[-1].frame, len(t.detections)) for t in tracks),
        key=lambda x: x[1],
    )
    print(f"{'track_id':>8} {'first':>6} {'last':>6} {'n_dets':>7}")
    for track_id, first, last, n in lifespans:
        print(f"{track_id:>8} {first:>6} {last:>6} {n:>7}")


if __name__ == "__main__":
    main()
