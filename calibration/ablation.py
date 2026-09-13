#!/usr/bin/env python3
"""Ablation + sensitivity analysis for the hand pipeline, across all 39 real
clips, without any labelled ground truth (see `metrics.py`'s docstring for
why none exists for this dataset). Two questions, both answerable from the
outcome distribution alone:

  1. ABLATION -- disable one rule at a time (dedup, size filter, shape
     filter, implausible-displacement, unsupported/flicker, static,
     interpolation, selection) and see how kept/rejected-by-reason/
     interpolated shifts vs. the full pipeline. Every rule is disabled
     purely through a `hand_config(**overrides)` value that makes it a
     structural no-op -- nothing under `adapter/` is touched or bypassed.
  2. SENSITIVITY -- sweep each key threshold across a plausible range and
     watch kept%/interpolated% respond, to find which thresholds are
     knife-edge (a small change swings the outcome a lot) vs. which sit on
     a flat plateau (the current default could move quite a bit with little
     effect) -- the basis for a defensible robustness claim.

**Critical gotcha (already bit this codebase once, see planning.md and
EVAL_LOG.md):** `Detection.tag` is mutated in place by every stage. Every
single data point here -- every ablation, every sweep point -- calls
`load_clip()` fresh. Never reuse a `ClipData`/`Detection` across two
different `Config`s; the second run would silently see the first run's tags
baked into `Detection.tag` and produce plausible-looking but wrong numbers.

Stage 6 (stereo depth) stays off throughout, per the project's own findings
that it's uncalibrated outside one clip -- see `stereo_depth.py`.

Run directly for the full report + JSON results file against real data:

    conda activate koshalabs
    python calibration/ablation.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapter.association import track_detections
from adapter.geometric import (
    reject_duplicates,
    reject_implausible_shape,
    reject_implausible_size,
)
from adapter.hand_config import hand_config
from adapter.ingest import ClipData, load_clip
from adapter.interpolation import apply_stage4
from adapter.selection import apply_selection
from adapter.temporal import (
    reject_implausible_displacement,
    reject_static,
    reject_unsupported,
)
from adapter.types import Config, Tag
from calibration.sweep_thresholds import discover_clip_ids

OUTCOME_KEYS = (
    "kept",
    "interpolated",
    "dropped_duplicate",
    "rejected_size",
    "rejected_shape",
    "rejected_displacement",
    "rejected_unsupported",
    "rejected_static",
    "rejected_selection",
)


def classify_clip(clip: ClipData, config: Config) -> Counter:
    """Run the hand pipeline (stages 1-5, stage 6 off) on one already-loaded
    clip, attributing every raw + fabricated detection to exactly one
    outcome bucket. Calls each stage's already-public sub-rule functions
    directly, in the exact fixed order `apply_stage1`/`apply_stage3` already
    use, purely to snapshot tag state between sub-rules -- this doesn't
    change pipeline behavior, it only lets ablation attribute a stage-3
    rejection to the specific sub-rule that caused it (the fixed
    `rejection_reason_frequency` in `sweep_thresholds.py` only distinguishes
    stage 1 / stage 3 / stage 5, not stage 3's three sub-rules).
    """
    counts: Counter = Counter()
    frame_count = len(clip.detections)

    stage1_frames = []
    for frame_dets in clip.detections:
        frame_dets = list(frame_dets)

        after_dedup = reject_duplicates(frame_dets, config)
        survivor_ids = {id(d) for d in after_dedup}
        counts["dropped_duplicate"] += sum(1 for d in frame_dets if id(d) not in survivor_ids)

        after_size = reject_implausible_size(after_dedup, config)
        survivor_ids = {id(d) for d in after_size}
        counts["rejected_size"] += sum(1 for d in after_dedup if id(d) not in survivor_ids)

        after_shape = reject_implausible_shape(after_size, config)
        survivor_ids = {id(d) for d in after_shape}
        counts["rejected_shape"] += sum(1 for d in after_size if id(d) not in survivor_ids)

        stage1_frames.append(after_shape)

    tracks = track_detections(stage1_frames, config)
    pre_stage3_ids = [id(d) for t in tracks for d in t.detections]

    reject_implausible_displacement(tracks, config)
    after_disp = {id(d): d.tag for t in tracks for d in t.detections}
    reject_unsupported(tracks, config)
    after_unsup = {id(d): d.tag for t in tracks for d in t.detections}
    reject_static(tracks, clip.pose, config)
    after_static = {id(d): d.tag for t in tracks for d in t.detections}

    stage3_reason: dict[int, str] = {}
    for key in pre_stage3_ids:
        if after_disp[key] == Tag.REJECTED:
            stage3_reason[key] = "rejected_displacement"
        elif after_unsup[key] == Tag.REJECTED:
            stage3_reason[key] = "rejected_unsupported"
        elif after_static[key] == Tag.REJECTED:
            stage3_reason[key] = "rejected_static"

    before_stage4_ids = set(pre_stage3_ids)
    apply_stage4(tracks, config, frame_count=frame_count)
    after_stage4 = {id(d): d.tag for t in tracks for d in t.detections}
    fabricated_ids = set(after_stage4) - before_stage4_ids

    apply_selection(tracks, config)
    after_stage5 = {id(d): d.tag for t in tracks for d in t.detections}

    for key, final_tag in after_stage5.items():
        if final_tag == Tag.INTERPOLATED:
            counts["interpolated"] += 1
        elif final_tag != Tag.REJECTED:
            counts["kept"] += 1
        elif key in fabricated_ids:
            counts["rejected_selection"] += 1
        elif stage3_reason.get(key) is not None and after_stage4[key] == Tag.REJECTED:
            counts[stage3_reason[key]] += 1
        else:
            counts["rejected_selection"] += 1

    return counts


def run_dataset(data_dir: str, clip_ids: list[str], config: Config) -> Counter:
    """Sum `classify_clip` across every clip, loading each one fresh -- the
    one rule this whole module exists to enforce (see module docstring).
    """
    total: Counter = Counter()
    for cid in clip_ids:
        clip = load_clip(os.path.join(data_dir, cid))
        total.update(classify_clip(clip, config))
    return total


def _pct(counts: Counter, key: str) -> float:
    total = sum(counts.values())
    return counts.get(key, 0) / total if total else 0.0


def outcome_summary(counts: Counter) -> dict:
    total = sum(counts.values())
    return {
        "total": total,
        "counts": dict(counts),
        "pct": {k: (counts.get(k, 0) / total if total else 0.0) for k in OUTCOME_KEYS},
    }


# ---------------------------------------------------------------------------
# 1. Ablation: one rule disabled per config, purely through hand_config()
#    overrides that make the rule a structural no-op. See each entry's
#    comment for why the chosen value can never fire.
# ---------------------------------------------------------------------------

ABLATIONS: dict[str, dict] = {
    "no_dedup": {
        # IoU is bounded [0, 1]; 1.1 is unreachable, so reject_duplicates'
        # match condition (IoU AND containment) is never satisfied.
        "duplicate_iou_threshold": 1.1,
    },
    "no_size_filter": {
        "plausible_size": (0.0, float("inf")),
    },
    "no_shape_filter": {
        # A degenerate zero-height box is still rejected unconditionally by
        # reject_implausible_shape regardless of config -- negligible on
        # real data, noted in the report rather than hidden.
        "plausible_shape": (0.0, float("inf")),
    },
    "no_displacement_check": {
        "max_speed_px_per_frame": float("inf"),
    },
    "no_flicker_check": {
        # len(track.detections) < 0 is never true.
        "min_supported_track_length": 0,
    },
    "no_static_check": {
        # No real track has a billion-frame still-run, so the sustained-run
        # requirement never completes.
        "min_static_run_frames": 10**9,
    },
    "no_interpolation": {
        # The smallest real gap the fill logic ever considers has dt=2
        # (dt<=1 means "no gap"); max_dropout_frames=0 rejects every gap
        # before the position-prediction check even runs. Exit-detection
        # (track.state) is untouched -- it doesn't affect the tag counts
        # this report measures.
        #
        # NOT A CLEAN ISOLATION OF STAGE 4, and the report says so: this
        # field is also association.py's own patience window for expiring
        # a stale track (`frame_idx - track.last_frame > max_dropout_frames`,
        # association.py line ~89). At 0, a track expires the instant a
        # single frame passes with no matching detection -- which, given
        # this dataset's real per-frame detection dropout, fragments nearly
        # every track down to 1-2 detections, which stage 3's flicker rule
        # (`min_supported_track_length`) then rejects almost entirely. The
        # result (kept collapses to ~0%, rejected_unsupported balloons to
        # ~94%) is a real, reportable finding about this specific config
        # field's coupling across stages 2 and 4, not a demonstration of
        # what stage 4 alone contributes -- there is no config-only lever
        # that disables stage 4's fill without also touching stage 2's
        # patience, since both read the same field. See the sensitivity
        # sweep on `max_dropout_frames` (which includes 0) for the same
        # cliff in isolation.
        "max_dropout_frames": 0,
    },
    "no_selection_cap": {
        # No frame in this dataset has anywhere near a billion candidates.
        "class_max_instances": 10**9,
    },
}


def run_ablation_suite(data_dir: str, clip_ids: list[str]) -> dict:
    results = {}
    baseline_config = hand_config()
    results["full_pipeline"] = outcome_summary(run_dataset(data_dir, clip_ids, baseline_config))
    for name, overrides in ABLATIONS.items():
        config = hand_config(**overrides)
        results[name] = outcome_summary(run_dataset(data_dir, clip_ids, config))
    return results


# ---------------------------------------------------------------------------
# 2. Sensitivity: sweep one threshold at a time, holding everything else at
#    hand_config() defaults. Ranges are centered on the current default
#    (included exactly, so the sweep reproduces the full-pipeline baseline
#    at one point) and span roughly 3-5x above/below it, informed by the
#    real percentiles already documented in hand_config.py/types.py.
# ---------------------------------------------------------------------------

_DEFAULTS = hand_config()

SCALAR_SWEEPS: dict[str, list[float]] = {
    # default 110; real p99=82.6, p99.5=97.7, p99.9=132 px/frame (types.py).
    "max_speed_px_per_frame": [50, 70, 90, 110, 130, 150, 200, 300, 500],
    # default 350; fastest single observed jump across 39 clips is 305.8.
    "track_gate_speed_px_per_frame": [150, 200, 250, 300, 350, 400, 500, 700, 1000],
    # default 15; real gap-length percentiles p50=3, p75=13, p90=54
    # (hand_config.py). 0 is included deliberately: `max_dropout_frames` is
    # NOT purely stage 4's parameter -- association.py's tracker also uses
    # it as its own patience window for expiring a stale track
    # (`frame_idx - track.last_frame > config.max_dropout_frames`). At 0,
    # ANY single-frame dropout expires a track immediately, which is a much
    # more severe regime change than 1 (tolerates exactly one missed
    # frame) -- see this module's docstring on `ABLATIONS["no_interpolation"]`
    # for the full account of this coupling, first found via that ablation.
    "max_dropout_frames": [0, 1, 3, 5, 10, 15, 20, 30, 50, 100],
    # default 15 (Milestone 4's sustained-run fix).
    "min_static_run_frames": [1, 3, 5, 10, 15, 20, 30, 50, 100],
    # default 4.0; real p25 on long clean tracks is 3.68px (hand_config.py).
    "static_px_threshold": [1, 2, 3, 4, 5, 6, 8, 12, 20],
    # default 0.5; real pair-IoU distribution has a cluster at 0.5-0.7 (hand_config.py).
    "duplicate_iou_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # default 0.7; real duplicate pairs found at containment >= 0.73 (planning.md).
    "duplicate_containment_threshold": [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# Tuple-valued thresholds: swept one bound at a time, the other held at its
# hand_config() default.
TUPLE_SWEEPS: dict[tuple[str, str], list[float]] = {
    ("plausible_size", "lo"): [0.0, 20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 300.0],
    ("plausible_size", "hi"): [800.0, 900.0, 1000.0, 1100.0, 1150.0, 1200.0, 1400.0, 1800.0, 2500.0],
    ("plausible_shape", "lo"): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    ("plausible_shape", "hi"): [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
}


def _config_for_scalar(param: str, value: float) -> Config:
    return hand_config(**{param: value})


def _config_for_tuple(param: str, side: str, value: float) -> Config:
    lo, hi = getattr(_DEFAULTS, param)
    if side == "lo":
        lo = value
    else:
        hi = value
    return hand_config(**{param: (lo, hi)})


def _series_verdict(points: list[dict]) -> dict:
    """Characterize a swept series as knife-edge or flat, from kept%'s
    step-to-step deltas alone (interpolated% moves far less throughout,
    per the actual numbers -- see EVAL_LOG.md). A step is called knife-edge
    if it swings kept% by more than 5 percentage points; "sits on a flat
    region around the default" if the default's neighboring steps both
    swing it by less than 1 point.
    """
    kept_pcts = [p["outcome"]["pct"]["kept"] for p in points]
    deltas = [abs(kept_pcts[i + 1] - kept_pcts[i]) for i in range(len(kept_pcts) - 1)]
    max_delta = max(deltas) if deltas else 0.0
    default_idx = next((i for i, p in enumerate(points) if p.get("is_default")), None)
    default_neighbor_delta = None
    if default_idx is not None:
        neighbor_deltas = []
        if default_idx > 0:
            neighbor_deltas.append(deltas[default_idx - 1])
        if default_idx < len(deltas):
            neighbor_deltas.append(deltas[default_idx])
        default_neighbor_delta = max(neighbor_deltas) if neighbor_deltas else 0.0
    return {
        "max_step_delta_kept_pct": max_delta,
        "kept_pct_range": (min(kept_pcts), max(kept_pcts)) if kept_pcts else (0.0, 0.0),
        "default_neighbor_max_delta": default_neighbor_delta,
        "knife_edge": max_delta > 0.05,
        "flat_at_default": default_neighbor_delta is not None and default_neighbor_delta < 0.01,
    }


def run_sensitivity_suite(data_dir: str, clip_ids: list[str]) -> dict:
    results: dict[str, dict] = {}

    for param, values in SCALAR_SWEEPS.items():
        default_value = getattr(_DEFAULTS, param)
        points = []
        for value in values:
            config = _config_for_scalar(param, value)
            counts = run_dataset(data_dir, clip_ids, config)
            points.append({
                "value": value,
                "is_default": value == default_value,
                "outcome": outcome_summary(counts),
            })
        results[param] = {"points": points, "verdict": _series_verdict(points)}

    for (param, side), values in TUPLE_SWEEPS.items():
        default_lo, default_hi = getattr(_DEFAULTS, param)
        default_value = default_lo if side == "lo" else default_hi
        label = f"{param}.{side}"
        points = []
        for value in values:
            config = _config_for_tuple(param, side, value)
            counts = run_dataset(data_dir, clip_ids, config)
            points.append({
                "value": value,
                "is_default": value == default_value,
                "outcome": outcome_summary(counts),
            })
        results[label] = {"points": points, "verdict": _series_verdict(points)}

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_ablation_report(results: dict) -> None:
    baseline = results["full_pipeline"]["pct"]
    print("=== Ablation: outcome distribution vs. full pipeline ===")
    header = f"{'config':22s} " + " ".join(f"{k:>13s}" for k in OUTCOME_KEYS)
    print(header)
    for name, summary in results.items():
        row = f"{name:22s} "
        cells = []
        for k in OUTCOME_KEYS:
            pct = summary["pct"][k]
            if name == "full_pipeline":
                cells.append(f"{pct:>12.1%} ")
            else:
                delta = pct - baseline[k]
                cells.append(f"{pct:>7.1%}{delta:+6.1%} ")
        print(row + "".join(cells))
    print()


def _print_sensitivity_report(results: dict) -> None:
    print("=== Sensitivity: kept%% / interpolated%% across each threshold's range ===")
    for name, data in results.items():
        verdict = data["verdict"]
        tag = "KNIFE-EDGE" if verdict["knife_edge"] else "flat"
        print(f"\n{name}  [{tag}]  max single-step kept%% swing: {verdict['max_step_delta_kept_pct']:.1%}"
              f"  kept%% range: {verdict['kept_pct_range'][0]:.1%}-{verdict['kept_pct_range'][1]:.1%}")
        for point in data["points"]:
            mark = " <- default" if point["is_default"] else ""
            pct = point["outcome"]["pct"]
            print(f"    {point['value']!r:>10}  kept={pct['kept']:.1%}  interpolated={pct['interpolated']:.1%}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-file", default=os.path.join(os.path.dirname(__file__), "ablation_results.json"))
    ap.add_argument("--clip-ids", nargs="*", default=None, help="subset of clip ids, default: all discovered")
    args = ap.parse_args()

    clip_ids = args.clip_ids or discover_clip_ids(args.data_dir)
    print(f"{len(clip_ids)} clips under {args.data_dir}\n")

    t0 = time.time()
    ablation_results = run_ablation_suite(args.data_dir, clip_ids)
    t1 = time.time()
    print(f"(ablation suite: {t1 - t0:.1f}s)\n")
    _print_ablation_report(ablation_results)

    sensitivity_results = run_sensitivity_suite(args.data_dir, clip_ids)
    t2 = time.time()
    print(f"(sensitivity suite: {t2 - t1:.1f}s)\n")
    _print_sensitivity_report(sensitivity_results)

    with open(args.results_file, "w") as f:
        json.dump(
            {
                "n_clips": len(clip_ids),
                "clip_ids": clip_ids,
                "ablation": ablation_results,
                "sensitivity": sensitivity_results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.results_file}")


if __name__ == "__main__":
    main()
