"""
Demo: run the Milestone 6 stereo-depth rule against a real clip bundle and
print a per-detection report. Run from the repo root:

    python scripts/demo_stereo_depth.py /path/to/clip_dir_with_left_right_and_hand_boxes

Or point it at individual files with --hand-boxes/--left/--right if your
files aren't in a single folder (as with the uploaded bundle, which prefixes
filenames with an upload id).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapter"))

import cv2

from exp.scafholds.stereo_depth import reject_beyond_reach
from exp.scafholds.nominal_calibration import DEFAULT_CALIBRATION


def find_by_suffix(directory, suffix):
    for f in os.listdir(directory):
        if f.endswith(suffix):
            return os.path.join(directory, f)
    raise FileNotFoundError(f"no file ending in {suffix} found in {directory}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", nargs="?", help="directory containing hand_boxes.json, video_left.mp4, video_right.mp4")
    ap.add_argument("--hand-boxes", default=None)
    ap.add_argument("--left", default=None)
    ap.add_argument("--right", default=None)
    ap.add_argument("--max-reach-m", type=float, default=1.8)
    ap.add_argument("--every-n-frames", type=int, default=30)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    args = ap.parse_args()

    hb_path = args.hand_boxes or find_by_suffix(args.clip_dir, "hand_boxes.json")
    left_path = args.left or find_by_suffix(args.clip_dir, "video_left.mp4")
    right_path = args.right or find_by_suffix(args.clip_dir, "video_right.mp4")

    with open(hb_path) as f:
        hb = json.load(f)
    frames_by_idx = {fr["frame"]: fr["detections"] for fr in hb["frames"]}

    left_cap = cv2.VideoCapture(left_path)
    right_cap = cv2.VideoCapture(right_path)
    frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"calibration: {DEFAULT_CALIBRATION.name}  "
          f"(baseline={DEFAULT_CALIBRATION.baseline_m}m, focal={DEFAULT_CALIBRATION.focal_px:.1f}px)")
    print(f"max_reach_m: {args.max_reach_m}")
    print(f"{'frame':>6} {'hand':>6} {'conf':>5} {'depth_m':>8} {'score':>6} {'status':>10}")

    n_rejected = 0
    n_total = 0
    for fidx in range(0, frame_count, args.every_n_frames):
        dets = frames_by_idx.get(fidx)
        if not dets:
            continue
        left_cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok_l, fl = left_cap.read()
        ok_r, fr = right_cap.read()
        if not (ok_l and ok_r):
            continue
        gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        annotated = [{"xyxy": tuple(d["xyxy"]), "handedness": d["handedness"], "confidence": d["confidence"]}
                     for d in dets if d["confidence"] >= args.min_confidence]
        if not annotated:
            continue
        reject_beyond_reach(annotated, gl, gr, max_reach_m=args.max_reach_m)

        for d in annotated:
            n_total += 1
            status = d.get("status", "reported")
            if status == "rejected":
                n_rejected += 1
            depth_str = f"{d['depth_m']:.2f}" if d.get("depth_m") is not None else "n/a"
            print(f"{fidx:>6} {d['handedness']:>6} {d['confidence']:>5.2f} "
                  f"{depth_str:>8} {d['depth_match_score']:>6.2f} {status:>10}")

    print(f"\n{n_rejected}/{n_total} detections rejected as beyond arm's reach "
          f"({n_rejected/max(1,n_total):.1%})")


if __name__ == "__main__":
    main()
