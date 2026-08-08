import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'adapter'))

import numpy as np
import cv2
from exp.scafholds.stereo_depth import estimate_box_depth, reject_beyond_reach
from exp.scafholds.nominal_calibration import ZED_X_2_2MM


def _make_stereo_pair(shape=(400, 600), patch_size=40, true_disparity=50, vertical_offset=0):
    """
    Build a synthetic stereo pair: a distinctive random patch pasted into a
    plain-noise left image, and pasted into the right image shifted left by
    true_disparity px (simulating a near object) plus an optional vertical
    offset (simulating this dataset's imperfect vertical alignment).
    """
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, size=shape, dtype=np.uint8)
    right = left.copy()

    patch = rng.integers(0, 255, size=(patch_size, patch_size), dtype=np.uint8)
    cy, cx = shape[0] // 2, shape[0]  # placeholder, set below
    cy, cx = shape[0] // 2, shape[1] // 2

    left[cy - patch_size // 2: cy + patch_size // 2, cx - patch_size // 2: cx + patch_size // 2] = patch

    rcx = cx - true_disparity
    rcy = cy + vertical_offset
    right[rcy - patch_size // 2: rcy + patch_size // 2, rcx - patch_size // 2: rcx + patch_size // 2] = patch

    box = (cx - patch_size // 2, cy - patch_size // 2, cx + patch_size // 2, cy + patch_size // 2)
    return left, right, box


def test_recovers_known_disparity_no_vertical_offset():
    left, right, box = _make_stereo_pair(true_disparity=60, vertical_offset=0)
    est = estimate_box_depth(left, right, box, patch_half_size=20, vertical_tolerance_px=10)
    assert est.valid
    assert abs(est.disparity_px - 60) <= 1


def test_recovers_known_disparity_with_small_vertical_offset():
    # This dataset's real eyes aren't perfectly vertically aligned -- confirm
    # the vertical_tolerance_px search band actually tolerates that.
    left, right, box = _make_stereo_pair(true_disparity=60, vertical_offset=4)
    est = estimate_box_depth(left, right, box, patch_half_size=20, vertical_tolerance_px=10)
    assert est.valid
    assert abs(est.disparity_px - 60) <= 1


def test_fails_gracefully_when_vertical_offset_exceeds_tolerance():
    left, right, box = _make_stereo_pair(true_disparity=60, vertical_offset=30)
    est = estimate_box_depth(left, right, box, patch_half_size=20, vertical_tolerance_px=5)
    # match should be poor/wrong since the true match is outside the search band
    assert (not est.valid) or abs(est.disparity_px - 60) > 5


def test_closer_object_larger_disparity_smaller_depth():
    left_near, right_near, box_near = _make_stereo_pair(true_disparity=120)
    left_far, right_far, box_far = _make_stereo_pair(true_disparity=20)
    est_near = estimate_box_depth(left_near, right_near, box_near, patch_half_size=20)
    est_far = estimate_box_depth(left_far, right_far, box_far, patch_half_size=20)
    assert est_near.valid and est_far.valid
    assert est_near.depth_m < est_far.depth_m


def _make_two_object_stereo_pair(shape=(400, 800), patch_size=40,
                                  near_center=(200, 200), near_disparity=120,
                                  far_center=(200, 600), far_disparity=8):
    """Two independent, distinguishable patches in one stereo pair, at
    different depths, so reject_beyond_reach can be tested against both
    within a single realistic frame."""
    rng = np.random.default_rng(2)
    left = rng.integers(0, 255, size=shape, dtype=np.uint8)
    right = left.copy()

    def paste(img, patch, cy, cx):
        h2 = patch_size // 2
        img[cy - h2:cy + h2, cx - h2:cx + h2] = patch

    near_patch = rng.integers(0, 255, size=(patch_size, patch_size), dtype=np.uint8)
    far_patch = rng.integers(0, 255, size=(patch_size, patch_size), dtype=np.uint8)

    paste(left, near_patch, *near_center)
    paste(left, far_patch, *far_center)
    paste(right, near_patch, near_center[0], near_center[1] - near_disparity)
    paste(right, far_patch, far_center[0], far_center[1] - far_disparity)

    h2 = patch_size // 2
    box_near = (near_center[1] - h2, near_center[0] - h2, near_center[1] + h2, near_center[0] + h2)
    box_far = (far_center[1] - h2, far_center[0] - h2, far_center[1] + h2, far_center[0] + h2)
    return left, right, box_near, box_far


def test_reject_beyond_reach_flags_far_detection_only():
    # near: disparity 120px -> ~0.73m at default calib, well under 1.8m
    # far: disparity 8px -> ~11m, well beyond 1.8m
    left, right, box_near, box_far = _make_two_object_stereo_pair()
    dets = [{"xyxy": box_near}, {"xyxy": box_far}]

    reject_beyond_reach(dets, left, right, max_reach_m=1.8, calib=ZED_X_2_2MM, patch_half_size=20)

    assert dets[0].get("status") != "rejected"
    assert dets[1].get("status") == "rejected"
    assert dets[1].get("reject_reason") == "beyond_arm_reach"


def test_low_confidence_match_defaults_to_keep_not_reject():
    # Two random, unrelated images -- no real match exists anywhere.
    rng = np.random.default_rng(1)
    left = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    right = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    dets = [{"xyxy": (100, 80, 160, 140)}]
    reject_beyond_reach(dets, left, right)
    assert dets[0].get("status") != "rejected"
    assert dets[0]["depth_m"] is None
