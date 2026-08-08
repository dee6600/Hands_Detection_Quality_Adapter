#!/usr/bin/env python3
"""Milestone 6: visualize the hand-specialized pipeline on real clips.

Runs `pipeline.run_hand_pipeline` (hand_config, stages 1-5, plus stage 6
stereo depth if --stereo-depth is passed) and draws every final detection
color-coded by what happened to it:

  green   solid  = kept (reported/merged)
  cyan    solid  = interpolated
  red     dashed = rejected: stage 1 (duplicate/size/shape)
  orange  dashed = rejected: stage 3 (displacement/flicker/static)
  purple  dashed = rejected: stage 5 (selection -- outranked by a better
                   track for the 2-hand cap)
  yellow  dashed = rejected: stage 6 (stereo depth -- beyond arm's reach)

This finer breakdown (vs. `visualize_interpolation.py`'s flat "rejected")
exists specifically to check Milestone 6's two NEW rejection reasons by eye:
selection enforcing the per-frame cap by track quality, and stereo depth
catching bystanders' hands. An "EXIT" label marks a confirmed border exit,
same as the generic pipeline's visualization.

BATCH BY DEFAULT: with no clip selection given, picks 5 random clips and
writes one annotated video per clip.

Usage (run from `main/detection_quality_adapter/`):

    # 5 random clips (default) -> ./output/hand_pipeline/
    python scripts/visualize_hand_pipeline.py

    # more clips, capped frames, and turn on stereo depth (stage 6)
    python scripts/visualize_hand_pipeline.py --num-clips 10 --max-frames 300 --stereo-depth

    # specific clips, reproducible sample
    python scripts/visualize_hand_pipeline.py --clips 0c54a47b_t010 407258cd_t036
    python scripts/visualize_hand_pipeline.py --seed 42

    # every clip in the dataset, sorted (not a random sample) -- this is the
    # slow one: ~20 min for all 39 clips without --stereo-depth, several
    # hours with it (a real video seek + template match per surviving
    # detection, across the whole dataset)
    python scripts/visualize_hand_pipeline.py --all
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

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.hand_config import hand_config
from adapter.ingest import load_clip
from adapter.interpolation import apply_stage4
from adapter.selection import apply_selection
from adapter.temporal import apply_stage3
from adapter.types import Tag, TrackState

_COLOR = {
    "kept": (0, 200, 0),  # green
    "interpolated": (255, 220, 0),  # cyan-ish (BGR)
    "rejected_geometric": (0, 0, 255),  # red
    "rejected_temporal": (0, 140, 255),  # orange
    "rejected_selection": (255, 0, 200),  # purple-ish
    "rejected_stereo_depth": (0, 220, 220),  # yellow
}
_DASHED = {"rejected_geometric", "rejected_temporal", "rejected_selection", "rejected_stereo_depth"}


def _snapshot(tracks) -> dict[int, Tag]:
    return {id(d): d.tag for t in tracks for d in t.detections}


def run_hand_pipeline_with_reasons(detections, pose, config, video_left_path=None, video_right_path=None):
    """Same stages `pipeline.run_hand_pipeline` runs, but snapshotting tags
    at each stage boundary to recover *which* stage is responsible for each
    detection's final tag -- diagnostic-only bookkeeping, kept out of the
    core modules (same pattern as `visualize_temporal.py`'s `_classify_stage3`).
    """
    frame_count = len(detections)
    stage1_frames = [apply_stage1(list(frame_dets), config) for frame_dets in detections]
    tracks = track_detections(stage1_frames, config)
    after_stage1 = _snapshot(tracks)

    apply_stage3(tracks, pose, config)
    after_stage3 = _snapshot(tracks)

    apply_stage4(tracks, config, frame_count=frame_count)
    after_stage4 = _snapshot(tracks)

    apply_selection(tracks, config)
    after_stage5 = _snapshot(tracks)

    after_stage6 = after_stage5
    if video_left_path is not None and video_right_path is not None:
        from adapter.stereo_depth import apply_stereo_depth_stage

        apply_stereo_depth_stage(tracks, video_left_path, video_right_path, config)
        after_stage6 = _snapshot(tracks)

    reasons: dict[int, str] = {}
    for track in tracks:
        for det in track.detections:
            key = id(det)
            final = det.tag
            if final == Tag.INTERPOLATED:
                reasons[key] = "interpolated"
            elif final == Tag.REJECTED:
                if after_stage6.get(key) == Tag.REJECTED and after_stage5.get(key) != Tag.REJECTED:
                    reasons[key] = "rejected_stereo_depth"
                elif after_stage5.get(key) == Tag.REJECTED and after_stage4.get(key) != Tag.REJECTED:
                    reasons[key] = "rejected_selection"
                elif after_stage3.get(key) == Tag.REJECTED and after_stage1.get(key, Tag.REPORTED) != Tag.REJECTED:
                    reasons[key] = "rejected_temporal"
                else:
                    reasons[key] = "rejected_geometric"
            else:
                reasons[key] = "kept"

    return tracks, reasons


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


def process_clip(clip_dir: str, out_video: str, max_frames: int | None = None, use_stereo_depth: bool = False) -> dict:
    clip = load_clip(clip_dir)
    config = hand_config()

    # Unlike visualize_interpolation.py, --max-frames caps what's actually
    # FED to the pipeline here, not just the rendered video length: stage 6
    # (stereo depth) does a video seek + template match per surviving
    # detection, so running it over a full ~2700-frame clip when the caller
    # only wanted a quick preview would take many minutes instead of seconds.
    detections = clip.detections[:max_frames] if max_frames is not None else clip.detections
    pose = clip.pose[:max_frames] if max_frames is not None else clip.pose

    video_left_path = os.path.join(clip_dir, "video_left.mp4")
    video_right_path = os.path.join(clip_dir, "video_right.mp4")
    tracks, reasons = run_hand_pipeline_with_reasons(
        detections, pose, config,
        video_left_path=video_left_path if use_stereo_depth else None,
        video_right_path=video_right_path if use_stereo_depth else None,
    )

    by_frame: dict[int, list] = defaultdict(list)
    exit_markers: dict[int, int] = {}
    for track in tracks:
        for det in track.detections:
            by_frame[det.frame].append(det)
        if track.state == TrackState.EXITING and track.detections:
            exit_markers[track.detections[-1].frame] = track.track_id

    cap = cv2.VideoCapture(video_left_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_left_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = len(detections)

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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
            if category in _DASHED:
                _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, category, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

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
        print(f"    {category:22s} {n:6d}  ({n / max(1, total):.1%})")
    print(f"    {'tracks exiting':22s} {stats['n_exiting']:6d}  / {stats['n_tracks']} tracks total")


def _discover_clip_ids(data_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
        and os.path.exists(os.path.join(data_dir, name, "video_left.mp4"))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../../data", help="directory containing clip bundle folders")
    ap.add_argument("--output-dir", default="scripts/output/hand_pipeline",
                     help="directory to write one annotated mp4 per clip into")
    ap.add_argument("--num-clips", type=int, default=5,
                     help="how many random clips to process (ignored if --clips is given)")
    ap.add_argument("--clips", nargs="+", default=None, metavar="CLIP_ID",
                     help="explicit clip ids to process instead of a random sample")
    ap.add_argument("--all", action="store_true", help="process every clip found under --data-dir, sorted, not a random sample")
    ap.add_argument("--seed", type=int, default=None, help="random seed, for a reproducible sample")
    ap.add_argument("--max-frames", type=int, default=None, help="cap rendered frames per clip")
    ap.add_argument("--stereo-depth", action="store_true",
                     help="also run stage 6 (stereo depth) -- slower, needs per-detection video seeks")
    args = ap.parse_args()

    available = _discover_clip_ids(args.data_dir)
    if not available:
        print(f"no clip bundles found under {args.data_dir}")
        sys.exit(1)

    if args.all:
        selected = available
    elif args.clips:
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
    print(f"stereo depth (stage 6): {'ON' if args.stereo_depth else 'off'}")
    print(f"selected: {selected}\n")

    results = []
    start = time.time()
    for i, cid in enumerate(selected, 1):
        clip_dir = os.path.join(args.data_dir, cid)
        out_video = os.path.join(args.output_dir, f"{cid}_hand.mp4")
        print(f"[{i}/{len(selected)}] {cid} ...")
        try:
            stats = process_clip(clip_dir, out_video, max_frames=args.max_frames, use_stereo_depth=args.stereo_depth)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append((cid, None))
            continue
        print(f"  wrote {stats['frames_written']} frames -> {out_video}")
        _print_stats(stats)
        results.append((cid, stats))

    elapsed = time.time() - start
    n_ok = sum(1 for _, s in results if s is not None)
    print("\nlegend: green=kept  cyan=interpolated  red=stage1(geometric)  "
          "orange=stage3(temporal)  purple=stage5(selection)  yellow=stage6(stereo depth)")
    print(f"done: {n_ok}/{len(selected)} clips processed in {elapsed:.1f}s -> {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
