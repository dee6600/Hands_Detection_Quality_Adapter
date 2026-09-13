#!/usr/bin/env python3
"""Label-free evaluation: how well Stage 4 (interpolation) recovers dropouts.

*** WHAT THIS MEASURES -- READ BEFORE CITING A NUMBER FROM THIS MODULE ***
This measures how well the pipeline reconstructs its OWN post-processed real
detections after they've been artificially deleted -- it is NOT a ground-truth
recovery rate. There is no labelled reference set for this dataset (see
`calibration/metrics.py`'s module docstring): the "real" box a masked frame is
scored against is itself just `hand_boxes.json`'s raw detector output, filtered
by stages 1-3. This is a **self-consistency / reconstruction-of-detector-output**
check, not an accuracy check against reality. A detector that is systematically
wrong in the same way at a masked frame and at its own interpolation of that
frame would still score perfectly here -- this can only show whether the
pipeline can put back what it itself would have reported, not whether that
report was ever correct.

*** METHOD ***
1. Run the generic stages-1-4 pipeline (`adapter.pipeline.run_pipeline`, NOT
   `run_hand_pipeline`) on the untouched clip using `hand_config()` thresholds.
   Stage 5 (selection) is deliberately excluded: it re-ranks/caps already-
   tracked-and-interpolated output, so including it would confound a
   stage-4-specific measurement with a stage-5 effect. This is the
   "reference" run.
2. Within each reference track, find maximal runs of consecutive frames
   carrying an untouched, non-duplicate raw detection (`tag == REPORTED`,
   deliberately NOT `MERGED`: a `MERGED` frame has a suppressed duplicate raw
   box still sitting in that frame's raw input, so masking only the surviving
   winner wouldn't actually empty the frame -- the previously-suppressed
   duplicate could resurface and survive stage 1 in its place, silently
   defeating the mask).
3. For each gap length in `GAP_LENGTHS` and each run long enough to keep
   `MIN_CONTEXT_FRAMES` real frames of context on both sides, mark one
   centered interior block of that length for removal (one placement per run
   per gap length -- see `_select_placements`'s docstring for why this stays
   deliberately simple rather than densely sampling every run).
4. For each gap length, `load_clip()` completely FRESH (see the critical
   gotcha below), delete exactly those raw boxes from the fresh clip's
   per-frame detection lists, and re-run the full stages-1-4 pipeline on the
   masked clip.
5. Re-locate each placement's track in the masked run via its still-untouched
   context frame (exact `xyxy` match -- a context frame was never masked, so
   stage 1 processes it identically in both runs) and check, per masked
   frame: did an `interpolated` detection appear there (recovery), and if so
   how close is it -- IoU and center distance in px -- to the real box that
   was removed? Also check the immediate unmasked context frames for
   unintended `interpolated` detections (false fill -- masking one object's
   box at a frame occasionally perturbs a *different* object's stage-1 dedup
   decision at that same frame; see `_mask_detections`'s docstring).

Stratified by gap length and by `meta.json`'s `job` field, across all 39 clips.

*** CRITICAL GOTCHA ***
`Detection.tag` is mutated in place by every pipeline stage. `load_clip()` is
called once for the reference run and again, completely separately, for every
masking configuration (one fresh load per gap length per clip) -- Detection
objects are never reused across two pipeline runs. Getting this wrong has
silently corrupted results in this codebase before (see `EVAL_LOG.md`).

*** A REAL FINDING THIS SURFACES, NOT A BUG IN THIS SCRIPT ***
`max_dropout_frames` (15 in `hand_config()`) is read by two different stages
for two different purposes that compose badly at the boundary: the tracker's
own patience window (`association.py`: a track expires once
`frame_idx - last_frame > max_dropout_frames`) and stage 4's own gap-length
cutoff (`interpolation.py`: skip if `dt > max_dropout_frames`). A masked gap
of exactly 15 frames means `dt = 16` between the surviving real detections on
either side of it -- `16 > 15`, so the TRACKER ITSELF expires and starts a
brand-new track before stage 4 is ever consulted about that gap. Expect
gap_length=15's recovery rate to sit near 0% by construction, regardless of
how good stage 4's own fill logic is -- this is a real off-by-one-flavored
interaction between two stages sharing one config field, worth flagging for
the paper rather than something this evaluation script should paper over.
`adapter/` is out of scope for this module to change.

Run directly for a full report against the real dataset (writes
`calibration/dropout_recovery_results.json`):

    conda activate koshalabs
    python calibration/dropout_recovery.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapter.hand_config import hand_config
from adapter.ingest import ClipData, load_clip
from adapter.pipeline import run_pipeline
from adapter.types import Config, Detection, Tag, Track

from calibration.sweep_thresholds import discover_clip_ids

Box = tuple[float, float, float, float]

GAP_LENGTHS: tuple[int, ...] = (1, 2, 3, 5, 10, 15)
# Real frames of untouched (non-masked) detection required on each side of a
# masked block: enough for `interpolation.py`'s "incoming velocity" to be
# computed from two real anchors before the gap, not just one, and enough
# that the false-fill check below is looking at a meaningful, stable window.
MIN_CONTEXT_FRAMES = 3
_PERCENTILES = (5, 25, 50, 75, 95)


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_distance(a: Box, b: Box) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return float(np.hypot(ax - bx, ay - by))


def _percentiles(values: list[float]) -> dict[int, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in _PERCENTILES}


@dataclass
class MaskPlacement:
    """One artificial dropout: a contiguous block of `gap_length` real,
    `REPORTED` frames removed from one track's raw input, with real,
    untouched frames on both sides to anchor re-identification and check for
    false fill.
    """

    gap_length: int
    frames: list[int]
    pre_frame: int
    pre_xyxy: Box
    removed_boxes: dict[int, Box]
    context_frames: list[int]


def _real_runs(track: Track) -> list[list[Detection]]:
    """Maximal stretches of consecutive-frame, untouched raw detections
    (`tag == REPORTED` only -- see the module docstring for why `MERGED` is
    excluded) within one track. Anything else (`REJECTED`, `INTERPOLATED`, or
    a frame index jump) breaks a run.
    """
    dets = sorted(track.detections, key=lambda d: d.frame)
    runs: list[list[Detection]] = []
    current: list[Detection] = []
    for d in dets:
        contiguous = bool(current) and d.frame == current[-1].frame + 1
        if d.tag != Tag.REPORTED or not contiguous:
            if len(current) > 1:
                runs.append(current)
            current = [d] if d.tag == Tag.REPORTED else []
            continue
        current.append(d)
    if len(current) > 1:
        runs.append(current)
    return runs


def _select_placements(tracks: list[Track], gap_length: int) -> list[MaskPlacement]:
    """One centered placement per qualifying run, not a dense sliding-window
    sample of every possible position -- keeps each placement's masked block
    and context window unambiguous and independent of every other placement
    in the same clip, at the cost of under-using very long runs (a long run
    could host several non-overlapping placements; only one is taken here).
    A denser sampler is a reasonable future extension, not needed for a first
    label-free pass.
    """
    needed = gap_length + 2 * MIN_CONTEXT_FRAMES
    placements = []
    for track in tracks:
        for run in _real_runs(track):
            if len(run) < needed:
                continue
            start = (len(run) - gap_length) // 2
            block = run[start : start + gap_length]
            pre = run[start - 1]
            context_before = run[start - MIN_CONTEXT_FRAMES : start]
            context_after = run[start + gap_length : start + gap_length + MIN_CONTEXT_FRAMES]
            placements.append(
                MaskPlacement(
                    gap_length=gap_length,
                    frames=[d.frame for d in block],
                    pre_frame=pre.frame,
                    pre_xyxy=pre.xyxy,
                    removed_boxes={d.frame: d.xyxy for d in block},
                    context_frames=[d.frame for d in context_before + context_after],
                )
            )
    return placements


def _mask_detections(clip: ClipData, placements: list[MaskPlacement]) -> None:
    """Delete exactly the raw boxes named by `placements` from `clip`'s
    per-frame detection lists, matched by exact `xyxy` (safe: these are
    freshly loaded, untouched values straight from `hand_boxes.json`, so a
    reference detection's box and its raw-input counterpart are bit-for-bit
    identical). Mutates `clip.detections` in place.

    Caveat: if the removed box happened to be the winning side of a stage-1
    duplicate merge against a DIFFERENT object's box at that same frame,
    deleting it can change that other object's dedup outcome too (its
    previously-suppressed duplicate could now survive, or it could simply
    stop being tagged `merged`). Not corrected for here -- rare in practice
    (this dataset's real duplicate pairs are a small fraction of all boxes),
    and it's a genuine side effect of deleting a real detection, not a bug in
    the masking itself.
    """
    to_remove: dict[int, list[Box]] = {}
    for p in placements:
        for f, xyxy in p.removed_boxes.items():
            to_remove.setdefault(f, []).append(xyxy)

    for f, boxes in to_remove.items():
        remaining = list(clip.detections[f])
        for xyxy in boxes:
            for i, d in enumerate(remaining):
                if d.xyxy == xyxy:
                    del remaining[i]
                    break
        clip.detections[f] = remaining


def _empty_accumulator() -> dict:
    return {
        "n_placements": 0,
        "n_placements_track_lost": 0,
        "n_frames_masked": 0,
        "n_recovered": 0,
        "n_context_checked": 0,
        "n_false_fill": 0,
        "ious": [],
        "center_dists": [],
    }


def _merge_accumulator(a: dict, b: dict) -> dict:
    for key in (
        "n_placements",
        "n_placements_track_lost",
        "n_frames_masked",
        "n_recovered",
        "n_context_checked",
        "n_false_fill",
    ):
        a[key] += b[key]
    a["ious"].extend(b["ious"])
    a["center_dists"].extend(b["center_dists"])
    return a


def _summarize_accumulator(acc: dict) -> dict:
    return {
        "n_placements": acc["n_placements"],
        "n_placements_track_lost": acc["n_placements_track_lost"],
        "n_frames_masked": acc["n_frames_masked"],
        "n_recovered": acc["n_recovered"],
        "recovery_rate": (acc["n_recovered"] / acc["n_frames_masked"]) if acc["n_frames_masked"] else None,
        "iou": _percentiles(acc["ious"]),
        "center_distance_px": _percentiles(acc["center_dists"]),
        "n_context_checked": acc["n_context_checked"],
        "n_false_fill": acc["n_false_fill"],
        "false_fill_rate": (acc["n_false_fill"] / acc["n_context_checked"]) if acc["n_context_checked"] else None,
    }


def _score_placements(placements: list[MaskPlacement], tracks_masked: list[Track]) -> dict:
    """Re-identify each placement's track in the masked run and score
    recovery/localization/false-fill against it. A placement whose track
    can't be re-identified at all (the pre-gap anchor frame vanished from
    every track -- shouldn't happen since context frames are never masked,
    but guarded against defensively) counts every one of its masked frames as
    unrecovered rather than being silently dropped from the denominator.
    """
    index: dict[tuple[int, Box], Track] = {(d.frame, d.xyxy): t for t in tracks_masked for d in t.detections}

    acc = _empty_accumulator()
    acc["n_placements"] = len(placements)
    for p in placements:
        track = index.get((p.pre_frame, p.pre_xyxy))
        acc["n_frames_masked"] += len(p.frames)
        if track is None:
            acc["n_placements_track_lost"] += 1
            continue

        by_frame = {d.frame: d for d in track.detections}
        for f in p.frames:
            det = by_frame.get(f)
            if det is not None and det.tag == Tag.INTERPOLATED:
                acc["n_recovered"] += 1
                real_box = p.removed_boxes[f]
                acc["ious"].append(_iou(det.xyxy, real_box))
                acc["center_dists"].append(_center_distance(det.xyxy, real_box))

        for f in p.context_frames:
            acc["n_context_checked"] += 1
            det = by_frame.get(f)
            if det is not None and det.tag == Tag.INTERPOLATED:
                acc["n_false_fill"] += 1

    return acc


def evaluate_clip(clip_dir: str, gap_lengths: tuple[int, ...] = GAP_LENGTHS, config: Config | None = None) -> dict:
    """Run the reference pass once, then one completely fresh masked pass per
    gap length. Returns `{"clip_id", "job", "by_gap": {gap_length: accumulator}}`
    with RAW accumulators (lists, not percentile summaries) so callers can
    pool them across clips before computing percentiles.
    """
    config = config or hand_config()
    clip_id = os.path.basename(os.path.normpath(clip_dir))

    clip_ref = load_clip(clip_dir)
    tracks_ref = run_pipeline(clip_ref.detections, clip_ref.pose, config)

    by_gap = {}
    for gap_length in gap_lengths:
        placements = _select_placements(tracks_ref, gap_length)
        if not placements:
            by_gap[gap_length] = _empty_accumulator()
            continue

        clip_masked = load_clip(clip_dir)  # fresh load -- never reuse tags/objects across runs
        _mask_detections(clip_masked, placements)
        tracks_masked = run_pipeline(clip_masked.detections, clip_masked.pose, config)

        by_gap[gap_length] = _score_placements(placements, tracks_masked)

    return {"clip_id": clip_id, "job": clip_ref.meta.get("job"), "by_gap": by_gap}


def aggregate(results: list[dict]) -> dict:
    """Pool raw accumulators across clips, one summary per gap length."""
    gap_lengths = sorted({g for r in results for g in r["by_gap"]})
    pooled = {g: _empty_accumulator() for g in gap_lengths}
    for r in results:
        for g, acc in r["by_gap"].items():
            _merge_accumulator(pooled[g], acc)
    return {"n_clips": len(results), "by_gap": {g: _summarize_accumulator(acc) for g, acc in pooled.items()}}


def aggregate_by_job(results: list[dict]) -> dict:
    by_job: dict[str, list[dict]] = {}
    for r in results:
        by_job.setdefault(r["job"], []).append(r)
    return {job: aggregate(rs) for job, rs in by_job.items()}


def run_full_report(
    data_dir: str,
    clip_ids: list[str] | None = None,
    gap_lengths: tuple[int, ...] = GAP_LENGTHS,
    config: Config | None = None,
) -> dict:
    clip_ids = clip_ids or discover_clip_ids(data_dir)
    config = config or hand_config()

    raw_results = [evaluate_clip(os.path.join(data_dir, cid), gap_lengths, config) for cid in clip_ids]

    per_clip = [
        {"clip_id": r["clip_id"], "job": r["job"], "by_gap": {g: _summarize_accumulator(acc) for g, acc in r["by_gap"].items()}}
        for r in raw_results
    ]
    return {
        "per_clip": per_clip,
        "overall": aggregate(raw_results),
        "by_job": aggregate_by_job(raw_results),
    }


def _print_summary(report: dict) -> None:
    print(
        "NOTE: no labelled reference set exists for this dataset -- 'recovery' here means\n"
        "recovering the pipeline's OWN filtered detector output, not ground truth. See this\n"
        "module's docstring.\n"
    )
    print(f"{len(report['per_clip'])} clips evaluated\n")
    print("=== Overall, by gap length (frames masked) ===")
    for g, s in sorted(report["overall"]["by_gap"].items()):
        rr = s["recovery_rate"]
        ffr = s["false_fill_rate"]
        iou_med = s["iou"].get(50)
        cd_med = s["center_distance_px"].get(50)
        print(
            f"  gap={g:>2d}  n_placements={s['n_placements']:5d}  "
            f"recovery_rate={'n/a' if rr is None else f'{rr:.1%}'}  "
            f"median_iou={'n/a' if iou_med is None else f'{iou_med:.3f}'}  "
            f"median_center_dist_px={'n/a' if cd_med is None else f'{cd_med:.1f}'}  "
            f"false_fill_rate={'n/a' if ffr is None else f'{ffr:.2%}'}"
        )

    print("\n=== By job (recovery_rate per gap length) ===")
    for job, agg in sorted(report["by_job"].items()):
        parts = []
        for g, s in sorted(agg["by_gap"].items()):
            rr = s["recovery_rate"]
            parts.append(f"{g}={'n/a' if rr is None else f'{rr:.0%}'}")
        print(f"  {job:20s} (n_clips={agg['n_clips']:2d})  {', '.join(parts)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dropout_recovery_results.json"))
    args = ap.parse_args()

    report = run_full_report(args.data_dir)
    _print_summary(report)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
