"""
Per-box stereo depth estimation and the "beyond arm's reach" hand-detection
selective_rejection rule described in the spec (Part 2, "A second eye is
available").

This is NOT a full stereo-matching pipeline (no rectification, no dense depth
map). It answers one narrow question per detection box: roughly how far is
this hand from the camera, well enough to separate "the wearer's own hand"
from "someone else's hand, out of reach". See nominal_calibration.py for why
the numbers behind this are approximate.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from exp.scafholds.nominal_calibration import NominalCalibration, DEFAULT_CALIBRATION, depth_from_disparity


@dataclass
class DepthEstimate:
    disparity_px: float | None
    depth_m: float | None
    match_score: float  # normalized cross-correlation score, higher = more trustworthy
    valid: bool  # False if match_score too low to trust this estimate at all


def estimate_box_depth(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    calib: NominalCalibration = DEFAULT_CALIBRATION,
    patch_half_size: int = 60,
    vertical_tolerance_px: int = 15,
    max_disparity_px: int = 500,
    min_disparity_px: int = -20,
    min_match_score: float = 0.25,
) -> DepthEstimate:
    """
    Estimate depth for one box by template-matching a patch around the box
    center from the left frame against a horizontal band in the right frame.

    vertical_tolerance_px exists because the two eyes are not perfectly
    vertically aligned in this dataset (see nominal_calibration.py docstring) --
    a strict same-row search would silently produce wrong disparities.

    min_disparity_px is slightly negative (not zero) to tolerate far-background
    points, which showed small negative apparent disparity in validation,
    likely from that same imperfect alignment, not real geometry.
    """
    x1, y1, x2, y2 = box_xyxy
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = left_gray.shape

    half = patch_half_size
    py1, py2 = max(0, cy - half), min(h, cy + half)
    px1, px2 = max(0, cx - half), min(w, cx + half)
    patch = left_gray[py1:py2, px1:px2]

    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return DepthEstimate(None, None, 0.0, valid=False)

    band_y1 = max(0, py1 - vertical_tolerance_px)
    band_y2 = min(h, py2 + vertical_tolerance_px)
    search_band = right_gray[band_y1:band_y2, :]

    if search_band.shape[0] < patch.shape[0] or search_band.shape[1] < patch.shape[1]:
        return DepthEstimate(None, None, 0.0, valid=False)

    result = cv2.matchTemplate(search_band, patch, cv2.TM_CCOEFF_NORMED)
    _, match_score, _, top_left = cv2.minMaxLoc(result)

    match_cx = top_left[0] + patch.shape[1] / 2.0
    disparity = (px1 + patch.shape[1] / 2.0) - match_cx

    if match_score < min_match_score:
        return DepthEstimate(disparity, None, match_score, valid=False)
    if not (min_disparity_px <= disparity <= max_disparity_px):
        return DepthEstimate(disparity, None, match_score, valid=False)

    depth_m = depth_from_disparity(disparity, calib)
    return DepthEstimate(disparity, depth_m, match_score, valid=depth_m is not None)


def reject_beyond_reach(
    detections: list,
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_reach_m: float = 1.8,
    calib: NominalCalibration = DEFAULT_CALIBRATION,
    on_low_confidence_match: str = "keep",
    **estimate_kwargs,
) -> list:
    """
    Apply the stereo-depth rejection rule to one frame's detections.

    detections: list of dict-like objects, each with a 'xyxy' box (mutated
    in place with 'depth_m' and, if rejected, 'status'='rejected',
    'reject_reason'='beyond_arm_reach').

    on_low_confidence_match: what to do when the stereo match itself is too
    unreliable to trust (match_score below threshold) -- 'keep' (default,
    conservative: don't reject on a depth estimate you don't trust) or
    'reject' (aggressive: treat unmatchable as suspicious). Default is 'keep'
    because a false reject discards a real hand permanently, while a missed
    reject just leaves one more candidate for later stages / manual review.

    max_reach_m default of 1.8m (not the more intuitive-sounding ~0.8m "arm's
    length") is empirically grounded: sampling ~150 high-confidence own-hand
    detections across clip t010 (a hairstyling task -- arms frequently
    extended toward a seated client) gave a depth median of 1.17m, with 72%
    beyond 1.0m and up to 2.44m. A tight threshold would reject the wearer's
    own hands during completely normal reaching motion. This is still a
    placeholder, not a calibrated value -- see Milestone 7. It almost
    certainly needs to vary by task/job (a seated desk task has a much
    shorter reach envelope than hairstyling), which is exactly why the spec
    requires deriving it from a labeled reference set rather than guessing.

    estimate_kwargs: passed through to estimate_box_depth (patch_half_size,
    vertical_tolerance_px, etc.) for callers that need non-default patch/search
    sizing -- e.g. small boxes need a smaller patch_half_size than the 60px
    default, or matching quality degrades from including too much non-hand
    background in the template.

    Returns the same list, annotated.
    """
    for det in detections:
        est = estimate_box_depth(left_gray, right_gray, det["xyxy"], calib=calib, **estimate_kwargs)
        det["depth_m"] = est.depth_m
        det["depth_match_score"] = est.match_score

        if not est.valid:
            if on_low_confidence_match == "reject":
                det["status"] = "rejected"
                det["reject_reason"] = "stereo_match_unreliable"
            # else: leave untouched, low-confidence match is not evidence either way
            continue

        if est.depth_m is not None and est.depth_m > max_reach_m:
            det["status"] = "rejected"
            det["reject_reason"] = "beyond_arm_reach"

    return detections
