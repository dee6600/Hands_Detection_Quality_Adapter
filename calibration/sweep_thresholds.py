#!/usr/bin/env python3
"""Milestone 7: derive what CAN be derived from this dataset without labels.

Spec S7 wants thresholds derived from a labelled reference set. This
dataset doesn't have one (see `metrics.py`'s module docstring for the full
explanation) -- `hand_boxes.json` is the noisy input, not ground truth.
What's honestly computable without labels, and implemented here:

  - **Dropout-length distribution** (spec S7 names this directly: "sets the
    largest break that may safely be interpolated"). Measured using a
    deliberately loosened `max_dropout_frames` so natural gap sizes aren't
    pre-clipped by the current threshold -- same "measure the unconstrained
    distribution first" approach the speed threshold got in an earlier
    checkpoint (see planning.md's Working Strategy).
  - **Rejection-reason frequency** -- a real, unsupervised proxy for spec
    S7's "observed frequency of each false-positive class... prioritize
    calibration effort." NOT a confirmed false-positive rate (that needs
    labels to know which rejections were actually correct), but genuinely
    where each rule is firing on real data.
  - **Raw box size/shape distributions** -- informs `plausible_size`/
    `plausible_shape`, purely descriptive of what the detector emits.
  - **Interpolated-detection proportion** -- spec S8's labels-free standing
    check, run for real across all 39 clips.

**Not attempted**: candidate-pool-width tuning ("rate at which a true hand
is outranked by a spurious box") genuinely needs labels -- there's no way to
know which candidate was "true" without them.

Run directly for a full report against the real dataset:

    python calibration/sweep_thresholds.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.hand_config import hand_config
from adapter.ingest import load_clip
from adapter.interpolation import apply_stage4
from adapter.pipeline import run_hand_pipeline
from adapter.selection import apply_selection
from adapter.temporal import apply_stage3
from adapter.types import Config, Tag
from calibration.metrics import interpolated_proportion

_PERCENTILES = (1, 5, 25, 50, 75, 90, 95, 99, 100)


def discover_clip_ids(data_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
        and os.path.exists(os.path.join(data_dir, name, "hand_boxes.json"))
    )


def _percentiles(values) -> dict[int, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in _PERCENTILES}


def _snapshot(tracks) -> dict[int, Tag]:
    return {id(d): d.tag for t in tracks for d in t.detections}


def raw_box_geometry(data_dir: str, clip_ids: list[str] | None = None) -> dict:
    """Real box side-length (px) and aspect-ratio distributions across every
    raw (pre-stage-1) detection in the given clips."""
    clip_ids = clip_ids or discover_clip_ids(data_dir)
    sides, ratios = [], []
    for cid in clip_ids:
        clip = load_clip(os.path.join(data_dir, cid))
        for dets in clip.detections:
            for d in dets:
                sides.append(max(d.width, d.height))
                if d.height > 0:
                    ratios.append(d.width / d.height)
    return {"n": len(sides), "side_px": _percentiles(sides), "aspect_ratio": _percentiles(ratios)}


def dropout_length_distribution(data_dir: str, clip_ids: list[str] | None = None, config: Config | None = None) -> dict:
    """Real internal-gap lengths (frames actually missing) within tracks,
    built with `max_dropout_frames` loosened to 10,000 so the current
    threshold doesn't pre-clip what we're trying to measure.
    """
    clip_ids = clip_ids or discover_clip_ids(data_dir)
    base_config = config or hand_config()
    loose_config = replace(base_config, max_dropout_frames=10_000)

    gaps = []
    for cid in clip_ids:
        clip = load_clip(os.path.join(data_dir, cid))
        stage1_frames = [apply_stage1(list(d), loose_config) for d in clip.detections]
        tracks = track_detections(stage1_frames, loose_config)
        for track in tracks:
            for prev, curr in zip(track.detections, track.detections[1:]):
                dt = curr.frame - prev.frame
                if dt > 1:
                    gaps.append(dt - 1)

    return {"n": len(gaps), "gap_length_frames": _percentiles(gaps)}


def rejection_reason_frequency(data_dir: str, clip_ids: list[str] | None = None, config: Config | None = None) -> dict:
    """Frequency of each rejection reason across stages 1/3/5 (stage 6 /
    stereo depth needs video, skipped here for speed -- see
    `scripts/visualize_hand_pipeline.py --stereo-depth` for that one).

    Deliberately accounts for detections on BOTH sides of `track_detections`,
    not just its output:

    - `reject_duplicates` drops a duplicate's loser from its returned list
      without ever setting its tag (only the surviving winner gets tagged
      `merged`), and `reject_implausible_size`/`_shape` do the same after
      tagging `rejected`. Either way, a stage-1 casualty never reaches
      `track_detections` and so never appears in any `Track` -- walking only
      `tracks` would silently drop it from the count entirely. A first
      attempt at this function did exactly that and undercounted the real
      dataset by 4.1% (147,590 vs. the true 153,876) with no error raised.
    - Going the other direction, stage 4 (`interpolation.py`) can CREATE
      brand-new `Detection` objects for frames with no raw detection at
      all -- these never existed as raw input, so a fix that walked only
      the original per-frame detection lists (to catch the stage-1 case
      above) would then miss every genuinely-fabricated `interpolated`
      detection instead, silently undercounting from the other direction.
      Caught by a total-count assertion in `test_sweep_thresholds.py`
      before this shipped, not by inspection.

    The correct set is: everything in the original per-frame lists NOT
    present in any final track (stage-1 casualties), plus everything IN a
    final track (covers original survivors and stage-4 fabrications alike).
    """
    clip_ids = clip_ids or discover_clip_ids(data_dir)
    config = config or hand_config()
    counts: Counter[str] = Counter()

    for cid in clip_ids:
        clip = load_clip(os.path.join(data_dir, cid))
        frame_count = len(clip.detections)

        stage1_frames = [apply_stage1(list(d), config) for d in clip.detections]
        tracks = track_detections(stage1_frames, config)
        in_a_track = {id(d) for t in tracks for d in t.detections}

        for frame_dets in clip.detections:
            for det in frame_dets:
                if id(det) in in_a_track:
                    continue
                # never made it past stage 1 -- tag alone tells us why: a
                # dedup loser is left untouched (still `reported`), a
                # size/shape reject is explicitly tagged `rejected`.
                if det.tag == Tag.REJECTED:
                    counts["rejected_size_shape"] += 1
                else:
                    counts["dropped_duplicate"] += 1

        apply_stage3(tracks, clip.pose, config)
        after_stage3 = _snapshot(tracks)
        apply_stage4(tracks, config, frame_count=frame_count)
        after_stage4 = _snapshot(tracks)
        apply_selection(tracks, config)
        after_stage5 = _snapshot(tracks)

        for track in tracks:
            for det in track.detections:
                key = id(det)
                final = det.tag
                if final == Tag.INTERPOLATED:
                    counts["interpolated"] += 1
                elif final != Tag.REJECTED:
                    counts["kept"] += 1
                elif after_stage5.get(key) == Tag.REJECTED and after_stage4.get(key) != Tag.REJECTED:
                    counts["rejected_selection"] += 1
                elif after_stage3.get(key) == Tag.REJECTED:
                    counts["rejected_temporal"] += 1
                else:
                    counts["rejected_other"] += 1

    total = sum(counts.values())
    return {"total": total, "counts": dict(counts)}


def interpolated_proportion_report(data_dir: str, clip_ids: list[str] | None = None, config: Config | None = None) -> dict:
    """Spec S8's labels-free standing check, run for real per clip."""
    clip_ids = clip_ids or discover_clip_ids(data_dir)
    config = config or hand_config()
    per_clip = {}
    for cid in clip_ids:
        clip = load_clip(os.path.join(data_dir, cid))
        tracks = run_hand_pipeline(clip.detections, clip.pose, config)
        per_clip[cid] = interpolated_proportion(tracks)

    values = list(per_clip.values())
    worst = max(per_clip, key=per_clip.get) if per_clip else None
    return {
        "per_clip": per_clip,
        "mean": float(np.mean(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
        "clip_with_max": worst,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    clip_ids = discover_clip_ids(args.data_dir)
    print(f"NOTE: no labelled reference set exists for this dataset -- everything below is\n"
          f"unsupervised (real data, no ground truth). See this module's docstring.\n")
    print(f"{len(clip_ids)} clips found under {args.data_dir}\n")

    print("=== Raw box geometry (pre-stage-1) ===")
    geom = raw_box_geometry(args.data_dir, clip_ids)
    print(f"n={geom['n']}")
    print(f"  side_px      {geom['side_px']}")
    print(f"  aspect_ratio {geom['aspect_ratio']}")

    print("\n=== Dropout-length distribution (unconstrained tracking) ===")
    dropout = dropout_length_distribution(args.data_dir, clip_ids)
    print(f"n={dropout['n']} gaps")
    print(f"  gap_length_frames {dropout['gap_length_frames']}")

    print("\n=== Rejection-reason frequency (hand pipeline, stages 1/3/5) ===")
    freq = rejection_reason_frequency(args.data_dir, clip_ids)
    total = freq["total"]
    for reason, n in sorted(freq["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {reason:22s} {n:7d}  ({n / max(1, total):.1%})")

    print("\n=== Interpolated-detection proportion (standing check, no labels needed) ===")
    interp = interpolated_proportion_report(args.data_dir, clip_ids)
    print(f"  mean across clips: {interp['mean']:.1%}")
    print(f"  max: {interp['max']:.1%} ({interp['clip_with_max']})")


if __name__ == "__main__":
    main()
