"""Milestone 6: per-box stereo depth and the "beyond arm's reach" rejection
rule (spec Part 2, "a second eye is available").

Migrated from the `exp/scafholds/` prototype (built and validated there
first against the real `t010` clip) and adapted to this package's real
`Detection`/`Config`/`Tag` types instead of ad-hoc dicts. See
`nominal_calibration.py`'s module docstring for the calibration derivation
(nominal ZED X, 2.2mm lens, 12cm baseline -- not exact, good enough for a
coarse arm's-reach threshold).

This is NOT a full stereo-matching pipeline (no rectification, no dense
depth map). It answers one narrow question per detection: roughly how far
is this hand from the camera, well enough to separate "the wearer's own
hand" from "someone else's, out of reach." Per the spec's own edge case,
position and entry direction alone don't separate them -- this rule exists
because nothing else in the pipeline can.

Needs actual video frames, not just detections/pose, so it's a separate,
optional stage rather than baked into `pipeline.run_hand_pipeline`'s default
signature -- see `apply_stereo_depth_stage` below, which that function calls
when video paths are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from adapter.nominal_calibration import DEFAULT_CALIBRATION, NominalCalibration, depth_from_disparity
from adapter.types import Config, Detection, Tag, Track


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
    """Estimate depth for one box by template-matching a patch around the
    box center from the left frame against a horizontal band in the right
    frame.

    `vertical_tolerance_px` exists because the two eyes are not perfectly
    vertically aligned in this dataset (see `nominal_calibration.py`) -- a
    strict same-row search would silently produce wrong disparities.
    `min_disparity_px` is slightly negative (not zero) to tolerate
    far-background points, which showed small negative apparent disparity
    in validation, likely from that same imperfect alignment, not real
    geometry.
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
    detections: list[Detection],
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    config: Config,
    calib: NominalCalibration = DEFAULT_CALIBRATION,
    on_low_confidence_match: str = "keep",
    **estimate_kwargs,
) -> dict[int, DepthEstimate]:
    """Apply the stereo-depth rejection rule to one frame's detections,
    tagging any box beyond `config.max_reach_m` as `rejected`. A no-op if
    `config.max_reach_m` is None (the rule isn't configured -- true for the
    generic, non-hand pipeline, which has no second eye to use anyway).

    `on_low_confidence_match`: what to do when the stereo match itself is
    too unreliable to trust (match_score below threshold) -- `"keep"`
    (default, conservative: don't reject on a depth estimate you don't
    trust) or `"reject"` (aggressive: treat unmatchable as suspicious).
    Default is `"keep"` because a false reject discards a real hand
    permanently, while a missed reject just leaves one more candidate for
    later review.

    Returns `{id(detection): DepthEstimate}` for every detection checked,
    for diagnostics/visualization -- `Detection` itself doesn't carry a
    depth field, matching how other stages keep per-rule diagnostic detail
    out of the core type (see e.g. `scripts/visualize_temporal.py`'s
    before/after tag diffing for the same pattern).
    """
    estimates: dict[int, DepthEstimate] = {}
    if config.max_reach_m is None:
        return estimates

    for det in detections:
        est = estimate_box_depth(left_gray, right_gray, det.xyxy, calib=calib, **estimate_kwargs)
        estimates[id(det)] = est

        if not est.valid:
            if on_low_confidence_match == "reject":
                det.tag = Tag.REJECTED
            continue  # low-confidence match is not evidence either way

        if est.depth_m is not None and est.depth_m > config.max_reach_m:
            det.tag = Tag.REJECTED

    return estimates


def apply_stereo_depth_stage(
    tracks: list[Track],
    video_left_path: str,
    video_right_path: str,
    config: Config,
    calib: NominalCalibration = DEFAULT_CALIBRATION,
) -> dict[int, DepthEstimate]:
    """Run `reject_beyond_reach` across an entire clip's tracks. Only checks
    detections not already `rejected` by an earlier stage (no point spending
    a video seek + template match on a box already excluded), grouped by
    frame so each frame's pair of video frames is decoded once regardless of
    how many detections land on it.

    A no-op (returns `{}` immediately, no video opened) if
    `config.max_reach_m` is None.
    """
    if config.max_reach_m is None:
        return {}

    by_frame: dict[int, list[Detection]] = {}
    for track in tracks:
        for det in track.detections:
            if det.tag != Tag.REJECTED:
                by_frame.setdefault(det.frame, []).append(det)

    if not by_frame:
        return {}

    left_cap = cv2.VideoCapture(video_left_path)
    right_cap = cv2.VideoCapture(video_right_path)
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError(f"could not open video(s): {video_left_path}, {video_right_path}")

    estimates: dict[int, DepthEstimate] = {}
    try:
        for frame_idx, dets in by_frame.items():
            left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok_l, left_frame = left_cap.read()
            ok_r, right_frame = right_cap.read()
            if not (ok_l and ok_r):
                continue
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            estimates.update(reject_beyond_reach(dets, left_gray, right_gray, config, calib=calib))
    finally:
        left_cap.release()
        right_cap.release()

    return estimates
