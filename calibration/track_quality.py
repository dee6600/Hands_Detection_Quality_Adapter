#!/usr/bin/env python3
"""Track-structure comparison: association-only vs. the full hand pipeline.

Compares two configurations of the same clip's tracks, per clip, aggregated
across all clips, and broken down by `meta.json`'s `job` field:

  - **association only** (stages 1-2): geometric rejection (`apply_stage1`)
    then the tracker (`track_detections`) -- no temporal rejection,
    interpolation, or selection.
  - **full hand pipeline** (stages 1-5): `run_hand_pipeline` with stage 6
    (stereo depth) left off, per this project's standing default.

Metrics reported per clip and aggregated: track count; mean/median track
length (frame span, gaps included) and detections-per-track (only frames
with a real or interpolated box, gaps excluded); a fragmentation proxy
(tracks that start close in space and time to another track's recent end --
see `fragmentation_candidates` below); gaps per track and total gap frames;
and `interpolated_proportion` (reused from `calibration/metrics.py`, always
0.0 for the association-only arm since stage 4 never runs there).

*** IMPORTANT FRAMING -- READ BEFORE INTERPRETING ANY NUMBER BELOW ***
None of these metrics are an accuracy claim, and there is no ground truth in
this dataset to make one (see `calibration/metrics.py`'s module docstring).
Longer tracks and fewer apparent fragments are NOT automatically "better":
a pipeline that wrongly bridges two different real objects into one track
produces exactly the same signature -- longer, fewer, cleaner-looking
tracks -- as a pipeline that correctly recovers one object's genuine gaps.
The two look identical from track-length/fragmentation numbers alone. This
is why every comparison here is reported alongside `interpolated_proportion`:
a pipeline arm with longer/fewer tracks AND a high interpolated proportion is
a stronger signal of over-bridging than a change in either number in
isolation. Read these as structural diagnostics of what the pipeline does to
track shape, not as evidence of correctness.

Critical gotcha (see `adapter/types.py`'s `Detection.tag`): every pipeline
stage mutates `Detection.tag` in place. This module calls `load_clip()`
fresh for each configuration compared for a given clip -- the association-
only run and the full-pipeline run never share `Detection` objects.

*** A finding this comparison surfaced, not a bug in it ***
Run against the real 39-clip dataset, `n_tracks`, `track_length_frames`, and
`fragmentation_candidates`/`fragmentation_rate` come back byte-for-byte
IDENTICAL between the two arms (787 tracks, 194.5-frame mean length, 254
fragmentation candidates, both arms). This is not a measurement error -- it
follows directly from the pipeline's own "nothing is ever deleted" design
(see the top-level README): stages 3-5 (temporal rejection, interpolation,
selection) only ever tag or append to an EXISTING track's `detections` list
(`temporal.py`/`selection.py` tag in place; `interpolation.py` only appends
new detections strictly BETWEEN a track's existing first and last frame,
then re-sorts -- see its `track.detections.append`/`.sort` calls). No stage
after association (stage 2) ever merges two tracks, splits one, or changes
a track's first/last frame. So track identity, count, and frame span are
fixed entirely by `track_detections` (stage 2) -- comparing "association
only" against "full pipeline" cannot by construction show the full pipeline
over- or under-fragmenting relative to association alone, because the two
arms' track BOUNDARIES are the same tracks. What DOES differ is track
CONTENT: `detections_per_track` rises (interpolation adds real detections),
`gaps_per_track`/`total_gap_frames` drop by roughly half (interpolation
fills many, not all, internal gaps -- gaps past `max_dropout_frames` are
left alone), and `interpolated_proportion` goes from a hard 0.0 (stage 4
never runs) to a real nonzero value. Read the length/count/fragmentation
metrics as characterizing the TRACKER (stage 2) alone, and the
detections-per-track/gaps/interpolated-proportion metrics as characterizing
what stages 3-5 add on top of it.

Run directly for a full report against the real dataset (writes
`calibration/track_quality_results.json`):

    conda activate koshalabs
    python calibration/track_quality.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.hand_config import hand_config
from adapter.ingest import ClipData, load_clip
from adapter.pipeline import run_hand_pipeline
from adapter.types import Config, Track

from calibration.metrics import interpolated_proportion
from calibration.sweep_thresholds import discover_clip_ids

_PERCENTILES = (1, 5, 25, 50, 75, 90, 95, 99, 100)

# Fragmentation-proxy defaults. Neither is derived from labels (none exist);
# both are deliberately generous heuristics so the proxy over-flags rather
# than under-flags candidate splits -- see `fragmentation_candidates`.
DEFAULT_TIME_WINDOW_FRAMES = 30  # 1s at this dataset's fixed 30fps
DEFAULT_DIAGONAL_MULTIPLIER = 2.0


def track_length_frames(track: Track) -> int:
    """Frame span from first to last detection, inclusive -- counts gap
    frames, unlike `detections_per_track`.
    """
    dets = track.detections
    if not dets:
        return 0
    return dets[-1].frame - dets[0].frame + 1


def gaps_for_track(track: Track) -> list[int]:
    """Length (in missing frames) of each internal gap in a track's
    detections, e.g. consecutive detections 4 frames apart leave a 3-frame
    gap. Same definition `sweep_thresholds.dropout_length_distribution`
    uses for its gap-length distribution.
    """
    gaps = []
    for prev, curr in zip(track.detections, track.detections[1:]):
        dt = curr.frame - prev.frame
        if dt > 1:
            gaps.append(dt - 1)
    return gaps


def fragmentation_candidates(
    tracks: list[Track],
    time_window_frames: int = DEFAULT_TIME_WINDOW_FRAMES,
    diagonal_multiplier: float = DEFAULT_DIAGONAL_MULTIPLIER,
) -> list[tuple[int, int, int, float]]:
    """Unsupervised proxy for "this track is probably a severed continuation
    of that one, not a genuinely new object." Flags a track A whose first
    detection appears shortly after (within `time_window_frames`) and close
    to (within `diagonal_multiplier` times the larger box's diagonal) some
    other track B's last detection.

    This is a proxy, not a confirmed split -- two different real objects
    that happen to pass through the same place in quick succession trigger
    it too, and there is no ground truth to tell the two apart. The spatial
    threshold scales with box size (rather than a fixed pixel distance)
    because this dataset's boxes vary hugely in scale with how close a hand
    is to the camera (see `hand_config.py`'s `plausible_size` derivation).

    Returns one `(track_b_id, track_a_id, dt_frames, distance_px)` tuple per
    candidate pair, sorted by track A's start frame. A track can appear as
    the "B" (probable predecessor) side of at most one candidate per "A"
    (nearest-in-time predecessor wins), avoiding double-counting one true
    split as multiple candidates.
    """
    ordered = sorted((t for t in tracks if t.detections), key=lambda t: t.detections[0].frame)
    candidates = []
    for i, track_a in enumerate(ordered):
        first_a = track_a.detections[0]
        best = None
        for track_b in ordered[:i]:
            last_b = track_b.detections[-1]
            dt = first_a.frame - last_b.frame
            if dt <= 0 or dt > time_window_frames:
                continue
            diagonal = max(
                math.hypot(first_a.width, first_a.height),
                math.hypot(last_b.width, last_b.height),
            )
            distance = math.hypot(
                first_a.center[0] - last_b.center[0], first_a.center[1] - last_b.center[1]
            )
            if distance > diagonal * diagonal_multiplier:
                continue
            if best is None or dt < best[1]:
                best = (track_b.track_id, dt, distance)
        if best is not None:
            track_b_id, dt, distance = best
            candidates.append((track_b_id, track_a.track_id, dt, distance))
    return candidates


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {str(p): float(np.percentile(arr, p)) for p in _PERCENTILES}


def _mean_median(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}


def summarize_tracks(
    tracks: list[Track],
    time_window_frames: int = DEFAULT_TIME_WINDOW_FRAMES,
    diagonal_multiplier: float = DEFAULT_DIAGONAL_MULTIPLIER,
) -> dict:
    """All track-structure metrics for one list of tracks (one clip, one
    pipeline configuration). See this module's docstring for the framing
    caveat before comparing two `summarize_tracks` outputs.
    """
    tracks = [t for t in tracks if t.detections]
    lengths = [track_length_frames(t) for t in tracks]
    det_counts = [len(t.detections) for t in tracks]
    all_gaps = [g for t in tracks for g in gaps_for_track(t)]
    gaps_per_track = [len(gaps_for_track(t)) for t in tracks]
    frag = fragmentation_candidates(tracks, time_window_frames, diagonal_multiplier)

    return {
        "n_tracks": len(tracks),
        "track_length_frames": _mean_median(lengths),
        "detections_per_track": _mean_median(det_counts),
        "gaps_per_track": _mean_median(gaps_per_track),
        "total_gap_frames": int(sum(all_gaps)),
        "n_gaps": len(all_gaps),
        "fragmentation_candidates": len(frag),
        "fragmentation_rate": len(frag) / len(tracks) if tracks else 0.0,
        "interpolated_proportion": interpolated_proportion(tracks),
    }


def association_only_tracks(clip: ClipData, config: Config) -> list[Track]:
    """Stages 1-2 only: geometric rejection then the tracker. No temporal
    rejection, interpolation, or selection.
    """
    stage1_frames = [apply_stage1(list(dets), config) for dets in clip.detections]
    return track_detections(stage1_frames, config)


def full_pipeline_tracks(clip: ClipData, config: Config) -> list[Track]:
    """Stages 1-5: the full hand pipeline, stage 6 (stereo depth) left off
    per this project's standing default (no video paths passed).
    """
    return run_hand_pipeline(clip.detections, clip.pose, config)


def compare_clip(
    clip_dir: str,
    config: Config | None = None,
    time_window_frames: int = DEFAULT_TIME_WINDOW_FRAMES,
    diagonal_multiplier: float = DEFAULT_DIAGONAL_MULTIPLIER,
) -> dict:
    """Association-only vs. full-pipeline track structure for one clip.
    Loads the clip fresh once per arm -- see this module's docstring on why
    `Detection.tag` mutation makes that mandatory, not just tidy.
    """
    config = config or hand_config()

    clip_a = load_clip(clip_dir)
    assoc_tracks = association_only_tracks(clip_a, config)
    assoc_summary = summarize_tracks(assoc_tracks, time_window_frames, diagonal_multiplier)

    clip_b = load_clip(clip_dir)
    full_tracks = full_pipeline_tracks(clip_b, config)
    full_summary = summarize_tracks(full_tracks, time_window_frames, diagonal_multiplier)

    return {
        "clip_id": os.path.basename(os.path.normpath(clip_dir)),
        "job": clip_a.meta.get("job"),
        "association_only": assoc_summary,
        "full_pipeline": full_summary,
    }


_POOLED_FIELDS = (
    "n_tracks",
    "total_gap_frames",
    "n_gaps",
    "fragmentation_candidates",
)
_RATE_FIELDS = ("fragmentation_rate", "interpolated_proportion")
_DIST_FIELDS = ("track_length_frames", "detections_per_track", "gaps_per_track")


def _aggregate_arm(clip_summaries: list[dict]) -> dict:
    """Pool raw per-track values across every clip in `clip_summaries`
    (rather than averaging each clip's mean) for the length/count
    distributions, sum the additive counts, and mean the per-clip rates.
    """
    agg: dict = {}
    for field in _POOLED_FIELDS:
        agg[field] = int(sum(s[field] for s in clip_summaries))
    for field in _RATE_FIELDS:
        values = [s[field] for s in clip_summaries]
        agg[field] = _mean_median(values)
    for field in _DIST_FIELDS:
        # Pool clip-level means weighted by that clip's track count, since
        # the raw per-track values aren't retained past `summarize_tracks`.
        means = [s[field]["mean"] for s in clip_summaries if s["n_tracks"] > 0]
        weights = [s["n_tracks"] for s in clip_summaries if s["n_tracks"] > 0]
        agg[field] = {
            "mean_of_clip_means": float(np.average(means, weights=weights)) if means else 0.0,
        }
    agg["n_clips"] = len(clip_summaries)
    return agg


def aggregate(results: list[dict]) -> dict:
    """Overall aggregation across every clip's comparison."""
    return {
        "association_only": _aggregate_arm([r["association_only"] for r in results]),
        "full_pipeline": _aggregate_arm([r["full_pipeline"] for r in results]),
    }


def aggregate_by_job(results: list[dict]) -> dict[str, dict]:
    """Same aggregation, grouped by `meta.json`'s `job` field. Every one of
    the 39 clips is a different job (see planning.md's Milestone 7 notes) --
    reporting only the pooled aggregate would hide job-specific structure.
    """
    by_job: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_job[r["job"] or "unknown"].append(r)
    return {job: aggregate(rs) for job, rs in sorted(by_job.items())}


def run_full_report(
    data_dir: str,
    clip_ids: list[str] | None = None,
    config: Config | None = None,
) -> dict:
    clip_ids = clip_ids or discover_clip_ids(data_dir)
    config = config or hand_config()
    results = [compare_clip(os.path.join(data_dir, cid), config) for cid in clip_ids]
    return {
        "per_clip": results,
        "overall": aggregate(results),
        "by_job": aggregate_by_job(results),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "track_quality_results.json"))
    args = ap.parse_args()

    print(
        "NOTE: track count/length/fragmentation are NOT accuracy claims -- there is no\n"
        "ground truth here, and wrongly bridging two objects produces the same 'longer,\n"
        "fewer tracks' signature as correctly recovering real gaps. Read every number\n"
        "next to interpolated_proportion. See this module's docstring.\n"
    )

    clip_ids = discover_clip_ids(args.data_dir)
    print(f"{len(clip_ids)} clips found under {args.data_dir}\n")

    report = run_full_report(args.data_dir, clip_ids)

    print("=== Overall (pooled across all clips) ===")
    for arm in ("association_only", "full_pipeline"):
        a = report["overall"][arm]
        print(f"--- {arm} ---")
        print(f"  n_tracks (summed)          {a['n_tracks']}")
        print(f"  track_length_frames        {a['track_length_frames']}")
        print(f"  detections_per_track       {a['detections_per_track']}")
        print(f"  fragmentation_rate         {a['fragmentation_rate']}")
        print(f"  gaps_per_track             {a['gaps_per_track']}")
        print(f"  total_gap_frames (summed)  {a['total_gap_frames']}")
        print(f"  interpolated_proportion    {a['interpolated_proportion']}")

    print("\n=== By job ===")
    for job, agg in report["by_job"].items():
        n_clips = agg["association_only"]["n_clips"]
        print(f"\n--- {job} ({n_clips} clip(s)) ---")
        for arm in ("association_only", "full_pipeline"):
            a = agg[arm]
            print(
                f"  {arm:16s} n_tracks={a['n_tracks']:4d}  "
                f"frag_rate(mean)={a['fragmentation_rate']['mean']:.3f}  "
                f"interp_prop(mean)={a['interpolated_proportion']['mean']:.3f}"
            )

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    main()
