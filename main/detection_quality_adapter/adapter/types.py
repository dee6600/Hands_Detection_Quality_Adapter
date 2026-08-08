"""Milestone 1: shared data model.

Shapes only, no behavior — every later stage (Milestones 2-6) is unit
tested against hand-built fixtures of these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tag(str, Enum):
    """What happened to a detection as it moved through the pipeline stages."""

    REPORTED = "reported"
    MERGED = "merged"
    REJECTED = "rejected"
    INTERPOLATED = "interpolated"


class TrackState(str, Enum):
    ACTIVE = "active"
    EXITING = "exiting"
    ENDED = "ended"


@dataclass
class Detection:
    """One detector output box, at input, or a stage's transformation of one.

    `xyxy` matches the raw dataset format (pixels, x1/y1/x2/y2). `class_label`
    carries the detector's raw class guess (e.g. hand_boxes.json's 0=left/
    1=right) through for debugging/visualization only — per spec, no stage
    may use it to decide identity or association.
    """

    frame: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_label: int | None = None
    tag: Tag = Tag.REPORTED

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


@dataclass
class Track:
    """One object's history across frames, plus tracker state for association."""

    track_id: int
    detections: list[Detection] = field(default_factory=list)
    state: TrackState = TrackState.ACTIVE
    predicted_position: tuple[float, float] | None = None
    predicted_velocity: tuple[float, float] = (0.0, 0.0)

    @property
    def last_detection(self) -> Detection | None:
        return self.detections[-1] if self.detections else None

    @property
    def last_frame(self) -> int | None:
        d = self.last_detection
        return d.frame if d is not None else None


@dataclass
class PoseSample:
    """One frame's 6DoF VIO pose, matching vio_pose.json's columnar fields."""

    t: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    speed: float


@dataclass
class Config:
    """Per-class expectations driving stages 1-4. Generic placeholders here —
    `hand_config.py` (Milestone 6) overrides these for the hand-specialized
    pipeline (e.g. class_max_instances=2, tighter candidate_pool_size).
    """

    plausible_size: tuple[float, float] = (20.0, 800.0)  # box side length, px
    plausible_shape: tuple[float, float] = (0.3, 3.0)  # aspect ratio w/h bounds
    duplicate_iou_threshold: float = 0.5  # IoU at/above this = same object
    candidate_pool_size: int = 5
    class_max_instances: int = 4
    # Plausibility check (stage 3, temporal.py): is this specific jump too fast to
    # trust, given an already-established track? Calibrated from real adjacent-frame
    # (dt=1) hand speeds across 38 of the 39 delivered clips: p99=82.6, p99.5=97.7,
    # p99.9=132.0 px/frame -- 110 sits just above the p99.5 mark. One clip
    # (ae580129_t057, a maintenance task) was excluded from that calibration: its
    # camera moves ~2-4x faster on average (VIO speed) than the other 38 clips, which
    # inflates apparent hand speed in image space with no change in real hand
    # behavior -- a flat px/frame threshold can't distinguish "fast hand" from "normal
    # hand, fast camera." Scaling this threshold by the clip's own VIO camera speed at
    # each frame would fix that properly; deferred to Milestone 7, where labeled data
    # can validate the scaling factor instead of more manual eyeballing.
    max_speed_px_per_frame: float = 110.0
    # Match gate (stage 2, association.py): is this detection even a plausible
    # candidate to extend an existing track? Deliberately generous -- comfortably
    # above the single fastest adjacent-frame jump observed in any of the 39 clips
    # (305.8 px/frame) -- because over-tightening the gate FRAGMENTS a genuine track
    # right at the moment it's moving fastest (see README: real motion-blur/low-
    # confidence "wobble" moments observed within otherwise long, clearly-continuous
    # tracks). The plausibility check above is what should flag those moments as
    # untrustworthy; the gate's only job is to not sever the track over it.
    track_gate_speed_px_per_frame: float = 350.0
    exit_border_margin_px: float = 20.0
    max_dropout_frames: int = 10
    min_supported_track_length: int = 3  # tracks shorter than this = flicker
    static_px_threshold: float = 4.0  # box movement below this, per frame, = "not moving"
    camera_moving_speed_mps: float = 0.05  # VIO speed at/above this = camera is moving
    camera_moving_angular_deg_per_frame: float = 1.0  # roll/pitch/yaw delta at/above this = camera is moving
    min_static_run_frames: int = 15  # consecutive still-while-moving frames needed before tagging static

    # Milestone 5 (interpolation.py): frame bounds for the border-exit test.
    # Matches this dataset's fixed ZED resolution; generic Part 1 default, not a
    # hand-specific one.
    frame_size: tuple[float, float] = (1920.0, 1200.0)
    # Per-border overrides of exit_border_margin_px / whether outward motion is
    # required to count as an exit, keyed by "left"/"right"/"top"/"bottom". Missing
    # borders fall back to exit_border_margin_px / True (motion required). Added for
    # hand_config.py (Milestone 6) per the spec cross-check: a hand disappearing
    # below the camera is occluded by the wearer's own torso, not walking out of
    # frame like a side exit, so it may need a larger margin and no outward-motion
    # requirement -- see planning.md's Milestone 5 design notes.
    exit_border_margin_overrides: dict[str, float] = field(default_factory=dict)
    exit_requires_outward_motion: dict[str, bool] = field(default_factory=dict)
