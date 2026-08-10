"""
Nominal (approximate) stereo calibration for the ZED-family head-mounted camera
used to record this dataset.

WHY NOMINAL, NOT EXACT
-----------------------
True per-unit intrinsics/extrinsics are factory-calibrated per physical camera
serial number and are NOT published anywhere generic online, and no calibration
file ships with this dataset bundle. These constants are derived from the public
ZED X datasheet (Stereolabs, Rev 2.1, 2023) and from validating disparity against
real hand boxes in clip 0c54a47b_t010 -- they are good enough to separate "close
to the wearer" from "several meters away" (the actual task), NOT for precision
3D reconstruction.

HOW THE LENS VARIANT WAS CHOSEN
--------------------------------
The ZED X ships with two fixed-focal-length lens options: 2.2mm (min depth 0.3m)
and 4mm (min depth 1.5m). The dataset's hand boxes frequently fill most of the
frame and touch the image border (e.g. clip t010 frame 0: a box spanning
y=[912, 1200] out of 1200px height -- a hand only centimeters from the camera).
A 4mm lens (1.5m minimum focus/depth) could not produce that. So the 2.2mm wide
lens variant is assumed. This was cross-checked empirically: template-matching a
large, high-confidence hand box (frame 230, t010) against the right eye gave a
disparity of ~267px, which under the 2.2mm assumption resolves to ~0.33m -- right
at the 2.2mm lens's 0.3m minimum depth spec, which is exactly the kind of value
you'd expect for a hand box that large. The 4mm assumption would have put the same
box at ~0.60m, which is *possible* but doesn't explain how much closer boxes get
in other frames. Prefer the 2.2mm assumption as the default.

BASELINE
--------
ZED X (non-mini): 12cm. ZED X Mini: 5cm. The full-size ZED X is used as the
default; if depth estimates look systematically off by roughly 2.4x once labels
exist, that's the signature of the Mini variant instead, and MINI_2_2MM should be
swapped in.

STEREO RECTIFICATION -- IMPORTANT CAVEAT
------------------------------------------
Feature-matching background points between video_left.mp4 and video_right.mp4 in
t010 shows a small but non-zero vertical offset (~2.4px mean, std ~1.7-2.5px, up
to ~9px), not the ~0px you'd expect from perfectly rectified stereo output. This
could be residual un-rectified geometry, or sub-frame temporal misalignment
between the two eyes (this is a moving head-mounted rig, so even a 1-frame skew
shows up as apparent parallax on background). Practical consequence: disparity
matching must NOT assume a strict horizontal epipolar line -- use a vertical
search tolerance (see stereo_depth.py). Do not treat this module's depth values
as metrically precise; treat them as ranked/thresholded estimates.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NominalCalibration:
    name: str
    baseline_m: float
    focal_px: float
    notes: str


def _focal_px_from_lens(focal_length_mm: float, pixel_size_um: float = 3.0) -> float:
    """f_px = f_mm / pixel_size_mm. ZED X sensor: 1/2.6", 3um pixels (datasheet)."""
    pixel_size_mm = pixel_size_um / 1000.0
    return focal_length_mm / pixel_size_mm


# ZED X, 2.2mm lens (wide, min depth 0.3m) -- DEFAULT, see module docstring.
ZED_X_2_2MM = NominalCalibration(
    name="ZED X / 2.2mm lens (default assumption)",
    baseline_m=0.12,
    focal_px=_focal_px_from_lens(2.2),  # ~733 px
    notes="Wide lens, 0.3m min depth. Matches large/near-border hand boxes seen in t010.",
)

# ZED X, 4mm lens (narrower, min depth 1.5m) -- kept for comparison/fallback only.
ZED_X_4MM = NominalCalibration(
    name="ZED X / 4mm lens (fallback, unlikely for close hand work)",
    baseline_m=0.12,
    focal_px=_focal_px_from_lens(4.0),  # ~1333 px
    notes="1.5m min depth -- inconsistent with near-frame-filling hand boxes observed.",
)

# ZED X Mini, 2.2mm lens -- swap to this if empirical depths look ~2.4x too far
# once labeled data is available (baseline is the only thing that changes).
ZED_X_MINI_2_2MM = NominalCalibration(
    name="ZED X Mini / 2.2mm lens (alternate baseline)",
    baseline_m=0.05,
    focal_px=_focal_px_from_lens(2.2),
    notes="Same lens/focal assumption, smaller baseline. Try this if depths run high.",
)

DEFAULT_CALIBRATION = ZED_X_2_2MM


def depth_from_disparity(disparity_px: float, calib: NominalCalibration = DEFAULT_CALIBRATION) -> float | None:
    """
    Z = f * B / disparity.  Returns meters, or None if disparity is at/below zero
    (object effectively at infinity, or a bad/negative match -- not a valid depth).
    """
    if disparity_px is None or disparity_px <= 0:
        return None
    return (calib.focal_px * calib.baseline_m) / disparity_px
