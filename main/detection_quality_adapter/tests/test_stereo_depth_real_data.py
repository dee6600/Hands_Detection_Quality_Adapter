"""Milestone 6: stereo depth against a real clip bundle.

Migrated from `exp/scafholds/test_stereo_depth_real_clip.py`, which pointed
at an old sandbox upload path (`/mnt/user-data/uploads`) that doesn't exist
in this repo -- fixed to use the real `data/` directory, same pattern as
every other real-data test in this suite. Encodes the empirical validation
described in `nominal_calibration.py`/`stereo_depth.py`'s docstrings:
sampling high-confidence own-hand detections should produce depths in a
plausible close-range envelope, not near-zero or kilometers-away nonsense,
with high match quality on real hand patches.
"""

import os

import cv2
import numpy as np
import pytest

from adapter.hand_config import hand_config
from adapter.ingest import load_clip
from adapter.pipeline import run_hand_pipeline
from adapter.stereo_depth import estimate_box_depth
from adapter.types import Tag

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
CLIP_ID = "0c54a47b_t010"


def _clip_dir() -> str:
    return os.path.join(DATA_DIR, CLIP_ID)


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_DIR, CLIP_ID)), reason="real dataset not present")
def test_real_clip_depth_distribution_is_plausible():
    clip = load_clip(_clip_dir())
    left_cap = cv2.VideoCapture(os.path.join(_clip_dir(), "video_left.mp4"))
    right_cap = cv2.VideoCapture(os.path.join(_clip_dir(), "video_right.mp4"))

    # Repeated cv2.CAP_PROP_POS_FRAMES seeks are the slow part of this test
    # (each one may decode forward from the nearest keyframe) -- stop as
    # soon as there's comfortably enough to judge the distribution, rather
    # than scanning the whole clip regardless. Keeps this test at a couple
    # of seconds instead of tens of seconds.
    target_samples = 40
    depths, scores = [], []
    for frame_idx in range(0, len(clip.detections), 30):
        if len(depths) >= target_samples:
            break
        dets = [d for d in clip.detections[frame_idx] if d.confidence >= 0.5]
        if not dets:
            continue
        left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok_l, left_frame = left_cap.read()
        ok_r, right_frame = right_cap.read()
        if not (ok_l and ok_r):
            continue
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        for d in dets:
            est = estimate_box_depth(left_gray, right_gray, d.xyxy)
            if est.valid:
                depths.append(est.depth_m)
                scores.append(est.match_score)

    left_cap.release()
    right_cap.release()

    assert len(depths) > 20, "too few valid estimates to judge the distribution"
    depths_arr = np.array(depths)
    scores_arr = np.array(scores)

    assert np.median(scores_arr) > 0.7, f"match quality too low: median={np.median(scores_arr):.2f}"
    assert depths_arr.min() > 0.05, f"implausibly close depth: {depths_arr.min():.3f}m"
    assert np.median(depths_arr) < 3.0, f"median depth implausibly far for close hand work: {np.median(depths_arr):.2f}m"


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_DIR, CLIP_ID)), reason="real dataset not present")
def test_run_hand_pipeline_with_video_paths_runs_end_to_end():
    """Full integration: `run_hand_pipeline` with real video paths actually
    invokes stage 6 (stereo depth) rather than silently skipping it. Capped
    to a short clip prefix -- video seeking per surviving detection is the
    slow part of this stage, and this is a structural smoke test, not a
    replacement for the distribution check above.
    """
    clip = load_clip(_clip_dir())
    config = hand_config()
    n = 60

    tracks = run_hand_pipeline(
        clip.detections[:n],
        clip.pose[:n],
        config,
        video_left_path=os.path.join(_clip_dir(), "video_left.mp4"),
        video_right_path=os.path.join(_clip_dir(), "video_right.mp4"),
    )

    all_dets = [d for t in tracks for d in t.detections]
    assert len(all_dets) > 0
    n_rejected = sum(1 for d in all_dets if d.tag == Tag.REJECTED)
    assert 0 < n_rejected < len(all_dets)  # some rejected (stages 1/3/5/6 combined), not all
