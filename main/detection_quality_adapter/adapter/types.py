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
    candidate_pool_size: int = 5
    class_max_instances: int = 4
    max_speed_px_per_frame: float = 150.0
    exit_border_margin_px: float = 20.0
    max_dropout_frames: int = 10
