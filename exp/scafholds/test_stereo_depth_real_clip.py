"""
Integration test against the real uploaded clip (0c54a47b_t010). Skips itself
if the clip isn't present at the expected path, so the synthetic test suite
still runs standalone without the dataset.

This test encodes the empirical validation described in
nominal_calibration.py and stereo_depth.py's docstrings: sampling ~150
high-confidence own-hand detections should produce depths in a plausible
close-range envelope (roughly 0.2m-3m for this task), not near-zero or
kilometers-away nonsense, and match quality should be high on real hand
patches (median >= 0.7).
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapter"))

CLIP_DIR_CANDIDATES = ["/mnt/user-data/uploads"]
HAND_BOXES_SUFFIX = "hand_boxes.json"
LEFT_VIDEO_SUFFIX = "video_left.mp4"
RIGHT_VIDEO_SUFFIX = "video_right.mp4"


def _find_clip_files():
    for d in CLIP_DIR_CANDIDATES:
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        hb = next((f for f in files if f.endswith(HAND_BOXES_SUFFIX)), None)
        vl = next((f for f in files if f.endswith(LEFT_VIDEO_SUFFIX)), None)
        vr = next((f for f in files if f.endswith(RIGHT_VIDEO_SUFFIX)), None)
        if hb and vl and vr:
            return os.path.join(d, hb), os.path.join(d, vl), os.path.join(d, vr)
    return None


def test_real_clip_depth_distribution_is_plausible():
    found = _find_clip_files()
    if found is None:
        print("SKIPPED: no real clip bundle found at", CLIP_DIR_CANDIDATES)
        return

    import cv2
    import numpy as np
    from exp.scafholds.stereo_depth import estimate_box_depth

    hb_path, left_path, right_path = found
    with open(hb_path) as f:
        hb = json.load(f)
    frames_by_idx = {fr["frame"]: fr["detections"] for fr in hb["frames"]}

    left_cap = cv2.VideoCapture(left_path)
    right_cap = cv2.VideoCapture(right_path)
    frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    depths, scores = [], []
    for fidx in range(0, frame_count, 30):
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
        for d in dets:
            if d["confidence"] < 0.5:
                continue
            est = estimate_box_depth(gl, gr, tuple(d["xyxy"]))
            if est.valid:
                depths.append(est.depth_m)
                scores.append(est.match_score)

    assert len(depths) > 20, "too few valid estimates to judge the distribution"
    depths = np.array(depths)
    scores = np.array(scores)

    assert np.median(scores) > 0.7, f"match quality too low: median={np.median(scores):.2f}"
    assert depths.min() > 0.05, f"implausibly close depth: {depths.min():.3f}m"
    assert np.median(depths) < 3.0, f"median depth implausibly far for close hand work: {np.median(depths):.2f}m"
    print(f"n={len(depths)} depth median={np.median(depths):.2f}m "
          f"range=[{depths.min():.2f}, {depths.max():.2f}] score median={np.median(scores):.2f}")


if __name__ == "__main__":
    test_real_clip_depth_distribution_is_plausible()
    print("OK")
