"""Milestone 4: stage 3 - displacement/flicker/static rejection.

Three rules that only make sense once tracks exist (spec S3, rules 4-6), in
the order planning.md lists them: displacement -> unsupported -> static.
Each rule tags offending Detections `rejected` in place; a track's
`detections` list is never restructured here -- the full history stays
intact for Milestone 5's interpolation/exit logic, and downstream consumers
filter by tag.

Camera motion is a real per-frame signal (`ClipData.pose`, from
`vio_pose.json`), not a stub -- see `_camera_moving`.
"""

from __future__ import annotations

import math

from adapter.types import Config, PoseSample, Tag, Track


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_delta_deg(a: float, b: float) -> float:
    """Signed difference b - a in degrees, wrapped to (-180, 180]."""
    return (b - a + 180.0) % 360.0 - 180.0


def reject_implausible_displacement(tracks: list[Track], config: Config) -> list[Track]:
    """A jump between consecutive detections in one track faster than
    `Config.max_speed_px_per_frame` (distance normalized by elapsed frames,
    so a jump across a coasted gap is judged against the whole gap) gets its
    later detection tagged `rejected` -- that's the position that can't be
    trusted, not the one before it.
    """
    for track in tracks:
        dets = track.detections
        for prev, curr in zip(dets, dets[1:]):
            dt = curr.frame - prev.frame
            if dt <= 0:
                continue
            if _distance(prev.center, curr.center) / dt > config.max_speed_px_per_frame:
                curr.tag = Tag.REJECTED
    return tracks


def reject_unsupported(tracks: list[Track], config: Config) -> list[Track]:
    """A track shorter than `Config.min_supported_track_length` has no track
    before or after it to lean on -- flicker, not a real object. Every
    detection in it is rejected.
    """
    for track in tracks:
        if len(track.detections) < config.min_supported_track_length:
            for det in track.detections:
                det.tag = Tag.REJECTED
    return tracks


def _camera_moving(pose: list[PoseSample], prev_frame: int, curr_frame: int, config: Config) -> bool:
    """Fails open (assumes moving) when pose data for either frame is
    missing, so the static rule never fires on a signal it doesn't have.
    """
    if not (0 <= prev_frame < len(pose)) or not (0 <= curr_frame < len(pose)):
        return True
    prev, curr = pose[prev_frame], pose[curr_frame]
    if curr.speed >= config.camera_moving_speed_mps:
        return True
    angular = max(
        abs(_angle_delta_deg(prev.roll, curr.roll)),
        abs(_angle_delta_deg(prev.pitch, curr.pitch)),
        abs(_angle_delta_deg(prev.yaw, curr.yaw)),
    )
    return angular >= config.camera_moving_angular_deg_per_frame


def reject_static(tracks: list[Track], pose: list[PoseSample], config: Config) -> list[Track]:
    """A box that stays essentially fixed in image space for a SUSTAINED run
    of at least `Config.min_static_run_frames` consecutive frames while the
    camera is moving reads as background structure riding along in frame.

    A single low-displacement step is deliberately not enough on its own: a
    head-mounted camera's apparent motion is often dominated by rotation
    (the wearer turning their head) rather than translation, and whatever
    the wearer is actively looking at -- for hand-eye tasks, usually their
    own hands -- tends to sit near the center of gaze and show the LEAST
    apparent motion of anything in the shot. A brief real pause (a grip
    adjustment, a steady precision hold) can easily dip under
    `static_px_threshold` for a frame or two without being background; only
    a run sustained across many consecutive frames is treated as evidence of
    a genuinely fixed object rather than an actively-used hand momentarily
    holding still. Every detection in a confirmed run is rejected, not just
    the tail, once the run is long enough to count.
    """
    for track in tracks:
        dets = track.detections
        run_start = None  # index into dets where the current still-run began
        for i in range(1, len(dets)):
            prev, curr = dets[i - 1], dets[i]
            still = (
                _camera_moving(pose, prev.frame, curr.frame, config)
                and _distance(prev.center, curr.center) <= config.static_px_threshold
            )
            if not still:
                run_start = None
                continue
            if run_start is None:
                run_start = i - 1
            if (i - run_start + 1) >= config.min_static_run_frames:
                for d in dets[run_start : i + 1]:
                    d.tag = Tag.REJECTED
    return tracks


def apply_stage3(tracks: list[Track], pose: list[PoseSample], config: Config | None = None) -> list[Track]:
    """Run the three stage-3 rules in fixed order on a clip's tracks."""
    config = config or Config()
    tracks = reject_implausible_displacement(tracks, config)
    tracks = reject_unsupported(tracks, config)
    tracks = reject_static(tracks, pose, config)
    return tracks
