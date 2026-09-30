#!/usr/bin/env python3
"""Reproduce every number reported in the paper and check it.

    python paper/reproduce_paper.py --data-dir data            # ~1-2 min
    python paper/reproduce_paper.py --data-dir data --rerun-calibration
                                   # also re-runs ablation/sensitivity and the
                                   # synthetic-dropout test (~10-20 min) instead
                                   # of reading calibration/*_results.json

Writes results/paper_numbers.json plus the CSVs behind Figs. 2-4 and prints a
PASS/FAIL line for every value in paper/expected_values.json. Exit code is 0
only if every check passes. Stage 6 (stereo depth, opt-in, ~4 h with video) is
not re-run here; see README.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapter.association import track_detections  # noqa: E402
from adapter.geometric import apply_stage1  # noqa: E402
from adapter.hand_config import hand_config  # noqa: E402
from adapter.ingest import load_clip  # noqa: E402
from adapter.pipeline import run_hand_pipeline  # noqa: E402
from adapter.types import Tag, TrackState  # noqa: E402
from calibration import sweep_thresholds as SW  # noqa: E402
from calibration.ablation import run_dataset  # noqa: E402
from calibration.metrics import interpolated_proportion  # noqa: E402

FAST_CAMERA_CLIP = "clip_21"  # excluded from the speed percentiles (paper Table VI)


def _iou_containment(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    A = (a[2] - a[0]) * (a[3] - a[1])
    B = (b[2] - b[0]) * (b[3] - b[1])
    union = A + B - inter
    return (inter / union if union > 0 else 0.0), (inter / min(A, B) if min(A, B) > 0 else 0.0)


def dataset_stats(data_dir, clip_ids):
    per_frame = collections.Counter()
    frames, raw, jobs, batches, maxn = [], 0, [], collections.Counter(), 0
    for cid in clip_ids:
        ft = json.load(open(os.path.join(data_dir, cid, "frame_ts.json")))
        hb = json.load(open(os.path.join(data_dir, cid, "hand_boxes.json")))
        meta = json.load(open(os.path.join(data_dir, cid, "meta.json")))
        n = {f["frame"]: len(f["detections"]) for f in hb["frames"]}
        for i in range(ft["frame_count"]):
            k = n.get(i, 0)
            per_frame["3+" if k >= 3 else str(k)] += 1
            maxn = max(maxn, k)
        frames.append(ft["frame_count"])
        raw += sum(n.values())
        jobs.append(meta["job"].strip().lower().replace("_", " "))
        batches[meta["batch"]] += 1
    F = sum(frames)
    return {
        "clips": len(clip_ids), "occupations": len(set(jobs)), "frames": F, "raw_detections": raw,
        "frames_min": min(frames), "frames_max": max(frames), "max_boxes_on_frame": maxn,
        "batches": dict(batches),
        "boxes_per_frame_pct": {k: 100 * per_frame[k] / F for k in ("2", "1", "3+", "0")},
        "boxes_per_frame_n": {k: per_frame[k] for k in ("2", "1", "3+", "0")},
    }


def threshold_evidence(data_dir, clip_ids):
    clips = lambda: ((cid, load_clip(os.path.join(data_dir, cid))) for cid in clip_ids)  # fresh loads
    sides = np.array([max(d.width, d.height) for _, c in clips() for fr in c.detections for d in fr])
    old_shape = hand_config(plausible_shape=(0.3, 3.0))
    ar = np.array([d.width / d.height for _, c in clips() for fr in c.detections
                   for d in apply_stage1(list(fr), old_shape)])
    gaps = SW.dropout_length_distribution(data_dir, clip_ids)
    # speeds were measured before retuning: shared 150 px/frame gate, previous size/shape bounds
    pre = hand_config(track_gate_speed_px_per_frame=150, max_speed_px_per_frame=150,
                      plausible_size=(20.0, 800.0), plausible_shape=(0.3, 3.0))
    speeds, fastest = [], 0.0
    for cid, c in clips():
        tracks = track_detections([apply_stage1(list(fr), pre) for fr in c.detections], pre)
        loc = [float(np.hypot(b.center[0] - a.center[0], b.center[1] - a.center[1]))
               for t in tracks for a, b in zip(t.detections, t.detections[1:]) if b.frame - a.frame == 1]
        fastest = max([fastest] + loc)
        if cid != FAST_CAMERA_CLIP:
            speeds += loc
    conts = []
    for _, c in clips():
        for fr in c.detections:
            for i in range(len(fr)):
                for j in range(i + 1, len(fr)):
                    u, k = _iou_containment(fr[i].xyxy, fr[j].xyxy)
                    if u >= 0.5:
                        conts.append(k)
    conts = np.array(conts)
    return {
        "side_px": {"min": float(sides.min()), "p1": float(np.percentile(sides, 1)), "max": float(sides.max()),
                    "n_over_800": int((sides > 800).sum()), "pct_over_800": float(100 * (sides > 800).mean())},
        "aspect_stage1_old_bounds": {q: float(np.percentile(ar, p)) for q, p in (("p1", 1), ("median", 50), ("p99", 99))},
        "gap_frames": {"n": gaps["n"], "p50": gaps["gap_length_frames"][50], "p75": gaps["gap_length_frames"][75],
                       "p90": gaps["gap_length_frames"][90]},
        "speed_px_per_frame": {q: float(np.percentile(speeds, p)) for q, p in (("p99", 99), ("p99_5", 99.5), ("p99_9", 99.9))},
        "fastest_jump_px_per_frame": fastest,
        "containment_pairs_iou_ge_0_5": {"n": int(len(conts)), "min": float(conts.min()),
                                          "p1": float(np.percentile(conts, 1)),
                                          "pct_ge_0_73": float(100 * (conts >= 0.73).mean()),
                                          "n_below_0_70": int((conts < 0.70).sum())},
    }


def final_pipeline(data_dir, clip_ids):
    cfg = hand_config()
    rows, secs = [], 0.0
    for cid in clip_ids:
        clip = load_clip(os.path.join(data_dir, cid))
        meta = clip.meta
        raw_ids = {id(d) for fr in clip.detections for d in fr}
        t0 = time.perf_counter()
        tracks = run_hand_pipeline(clip.detections, clip.pose, cfg)
        secs += time.perf_counter() - t0
        dets = [d for t in tracks for d in t.detections]
        rows.append({"clip": cid, "job": meta["job"], "tracks": len(tracks),
                     "exits": sum(t.state == TrackState.EXITING for t in tracks),
                     "in_tracks": len(dets),
                     "fabricated_surviving": sum(1 for d in dets if id(d) not in raw_ids and d.tag == Tag.INTERPOLATED),
                     "interp_share": interpolated_proportion(tracks)})
    shares = np.array([r["interp_share"] for r in rows])
    imax = int(shares.argmax())
    return {"tracks": sum(r["tracks"] for r in rows), "exits": sum(r["exits"] for r in rows),
            "in_track_detections": sum(r["in_tracks"] for r in rows),
            "interp_share_pct": {"mean": float(100 * shares.mean()), "median": float(100 * np.median(shares)),
                                 "max": float(100 * shares.max()), "max_clip_job": rows[imax]["job"]},
            "stages_1_5_seconds": secs}, rows


def load_or_run_calibration(data_dir, rerun):
    cal = os.path.join(ROOT, "calibration")
    if rerun:
        for s in ("ablation", "dropout_recovery"):
            subprocess.run([sys.executable, os.path.join(cal, f"{s}.py"), "--data-dir", data_dir], check=True)
    return (json.load(open(os.path.join(cal, "ablation_results.json"))),
            json.load(open(os.path.join(cal, "dropout_recovery_results.json"))))


def get(d, path):
    for p in path.split("."):
        d = d[p]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--rerun-calibration", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    clip_ids = SW.discover_clip_ids(args.data_dir)

    out = {"dataset": dataset_stats(args.data_dir, clip_ids),
           "thresholds": threshold_evidence(args.data_dir, clip_ids)}
    out["final"], per_clip = final_pipeline(args.data_dir, clip_ids)

    full = run_dataset(args.data_dir, clip_ids, hand_config())
    total = sum(full.values())
    out["outcomes"] = {"total": total, "raw": out["dataset"]["raw_detections"],
                       "stage4_new_boxes": total - out["dataset"]["raw_detections"],
                       "counts": dict(full), "pct_of_total": {k: 100 * v / total for k, v in full.items()}}
    it = out["final"]["in_track_detections"]
    temporal = full["rejected_static"] + full["rejected_displacement"] + full["rejected_unsupported"]
    out["table7_pct_of_tracked"] = {"kept": 100 * full["kept"] / it, "temporal": 100 * temporal / it,
                                    "interpolated": 100 * full["interpolated"] / it,
                                    "selection": 100 * full["rejected_selection"] / it,
                                    "exits_pct_of_tracks": 100 * out["final"]["exits"] / out["final"]["tracks"]}

    abl, drop = load_or_run_calibration(args.data_dir, args.rerun_calibration)
    out["ablation"] = {k: {"total": v["total"], "kept": 100 * v["pct"]["kept"],
                           "interpolated": 100 * v["pct"]["interpolated"],
                           "selection": 100 * v["pct"]["rejected_selection"],
                           "unsupported": 100 * v["pct"]["rejected_unsupported"]}
                       for k, v in abl["ablation"].items()}
    sens = {}
    for param, series in abl["sensitivity"].items():
        sens[param] = {str(p["value"]): {"kept": 100 * p["outcome"]["pct"]["kept"],
                                          "interpolated": 100 * p["outcome"]["pct"]["interpolated"],
                                          "static": 100 * p["outcome"]["pct"]["rejected_static"]}
                       for p in series["points"]}
    out["sensitivity"] = sens
    out["dropout"] = {g: {"placements": v["n_placements"], "recovery_pct": 100 * v["recovery_rate"],
                          "false_fill_pct": 100 * v["false_fill_rate"], "iou_p25": v["iou"]["25"],
                          "iou_median": v["iou"]["50"], "iou_p75": v["iou"]["75"],
                          "center_err_median_px": v["center_distance_px"]["50"]}
                      for g, v in drop["overall"]["by_gap"].items()}

    json.dump(out, open(os.path.join(args.out_dir, "paper_numbers.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, "fig2_boxes_per_frame.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["boxes_per_frame", "n_frames", "pct_of_frames"])
        for k in ("2", "1", "3+", "0"):
            w.writerow([k, out["dataset"]["boxes_per_frame_n"][k], out["dataset"]["boxes_per_frame_pct"][k]])
    with open(os.path.join(args.out_dir, "fig3_outcomes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["outcome", "count", "pct_of_total"])
        c = full
        for name, n in (("kept", c["kept"]), ("rejected_temporal", temporal),
                        ("dropped_duplicate", c["dropped_duplicate"]),
                        ("rejected_size_shape", c["rejected_size"] + c["rejected_shape"]),
                        ("interpolated", c["interpolated"]), ("rejected_selection", c["rejected_selection"])):
            w.writerow([name, n, 100 * n / total])
    with open(os.path.join(args.out_dir, "fig4_dropout_recovery.csv"), "w", newline="") as f:
        cols = ["gap_frames", "placements", "recovery_pct", "false_fill_pct", "iou_p25", "iou_median", "iou_p75"]
        w = csv.writer(f); w.writerow(cols)
        for g in sorted(out["dropout"], key=int):
            w.writerow([g] + [out["dropout"][g][k] for k in cols[1:]])
    with open(os.path.join(args.out_dir, "per_clip.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_clip[0])); w.writeheader(); w.writerows(per_clip)

    expected = json.load(open(os.path.join(ROOT, "paper", "expected_values.json")))
    fails = 0
    for e in expected:
        try:
            got = get(out, e["path"])
        except (KeyError, TypeError):
            got = None
        ok = got is not None and (got == e["value"] if isinstance(e["value"], str)
                                  else abs(got - e["value"]) <= e.get("tol", 0))
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {e['where']:<22} {e['path']:<55} paper={e['value']!s:<10} got={got}")
    print(f"\n{len(expected) - fails}/{len(expected)} checks passed; results in {args.out_dir}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
