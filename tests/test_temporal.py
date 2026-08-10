"""Milestone 4: hand-built synthetic tracks for stage 3 (temporal rejection).

One track per rule, in the fixed order the spec requires: displacement ->
unsupported -> static, plus a clean track that survives all three untouched
and one exercising `apply_stage3`'s order together. Camera motion is
exercised with real `PoseSample` sequences (not a boolean stub), per
planning.md's note that `vio_pose.json` is a real signal now.
"""

from adapter.temporal import (
    apply_stage3,
    reject_implausible_displacement,
    reject_static,
    reject_unsupported,
)
from adapter.types import Config, Detection, PoseSample, Tag, Track


def _det(frame, xy, confidence=0.8):
    x, y = xy
    return Detection(frame=frame, xyxy=(x - 20.0, y - 20.0, x + 20.0, y + 20.0), confidence=confidence)


def _track(track_id, positions_by_frame):
    t = Track(track_id=track_id)
    for frame, xy in positions_by_frame:
        t.detections.append(_det(frame, xy))
    return t


def _still_pose(n, speed=0.0, yaw=0.0):
    return [PoseSample(t=i / 30.0, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=yaw, speed=speed) for i in range(n)]


def _moving_pose(n, speed=1.0):
    return [PoseSample(t=i / 30.0, x=0.1 * i, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, speed=speed) for i in range(n)]


def test_reject_implausible_displacement_flags_the_jump():
    config = Config(max_speed_px_per_frame=50.0)
    track = _track(0, [(0, (0.0, 0.0)), (1, (30.0, 0.0)), (2, (500.0, 0.0)), (3, (530.0, 0.0))])

    reject_implausible_displacement([track], config)

    tags = [d.tag for d in track.detections]
    assert tags == [Tag.REPORTED, Tag.REPORTED, Tag.REJECTED, Tag.REPORTED]


def test_reject_unsupported_flags_short_tracks_entirely():
    config = Config(min_supported_track_length=3)
    flicker = _track(0, [(5, (100.0, 100.0)), (6, (105.0, 100.0))])  # length 2
    supported = _track(1, [(0, (0.0, 0.0)), (1, (10.0, 0.0)), (2, (20.0, 0.0))])  # length 3

    reject_unsupported([flicker, supported], config)

    assert all(d.tag == Tag.REJECTED for d in flicker.detections)
    assert all(d.tag == Tag.REPORTED for d in supported.detections)


def test_reject_static_flags_a_sustained_stationary_run():
    """`min_static_run_frames` kept small here for a compact fixture -- the
    real behavior (a brief pause must NOT trip this) is covered by
    `test_reject_static_does_not_flag_a_brief_pause` below.
    """
    config = Config(static_px_threshold=4.0, camera_moving_speed_mps=0.05, min_static_run_frames=3)
    pose = _moving_pose(5, speed=1.0)  # camera translating throughout
    # box barely moves (1px/frame) while camera moves -- background structure
    track = _track(0, [(0, (500.0, 500.0)), (1, (501.0, 500.0)), (2, (502.0, 500.0)), (3, (503.0, 500.0))])

    reject_static([track], pose, config)

    # once the run is confirmed long enough, the WHOLE run is rejected,
    # including its first frame -- it was part of the same static episode
    assert all(d.tag == Tag.REJECTED for d in track.detections)


def test_reject_static_does_not_flag_a_brief_pause():
    """The whole point of the sustained-run requirement: a real hand
    naturally pausing for a couple of frames (a grip adjustment, a steady
    hold) must survive, even while the camera is moving -- only a run that
    reaches `min_static_run_frames` should ever be rejected.
    """
    config = Config(static_px_threshold=4.0, camera_moving_speed_mps=0.05, min_static_run_frames=5)
    pose = _moving_pose(6, speed=1.0)
    # holds still for 3 frames (short of the 5-frame requirement), then moves again
    track = _track(0, [
        (0, (500.0, 500.0)),
        (1, (501.0, 500.0)),
        (2, (502.0, 500.0)),
        (3, (600.0, 500.0)),  # resumes real motion
        (4, (650.0, 500.0)),
    ])

    reject_static([track], pose, config)

    assert all(d.tag == Tag.REPORTED for d in track.detections)


def test_reject_static_does_not_flag_stationary_box_when_camera_is_still():
    config = Config(static_px_threshold=4.0, camera_moving_speed_mps=0.05, min_static_run_frames=3)
    pose = _still_pose(5, speed=0.0)  # camera not moving
    track = _track(0, [(0, (500.0, 500.0)), (1, (501.0, 500.0)), (2, (502.0, 500.0))])

    reject_static([track], pose, config)

    assert all(d.tag == Tag.REPORTED for d in track.detections)


def test_reject_static_uses_angular_motion_too():
    config = Config(
        static_px_threshold=4.0,
        camera_moving_speed_mps=0.05,
        camera_moving_angular_deg_per_frame=1.0,
        min_static_run_frames=2,
    )
    # zero translational speed, but camera panning (yaw changing) -- still "moving"
    pose = [
        PoseSample(t=0.0, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, speed=0.0),
        PoseSample(t=0.033, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=5.0, speed=0.0),
    ]
    track = _track(0, [(0, (500.0, 500.0)), (1, (501.0, 500.0))])

    reject_static([track], pose, config)

    assert all(d.tag == Tag.REJECTED for d in track.detections)


def test_clean_track_survives_all_three_rules_untouched():
    config = Config()
    pose = _moving_pose(5, speed=1.0)
    track = _track(0, [(0, (0.0, 0.0)), (1, (30.0, 0.0)), (2, (60.0, 0.0)), (3, (90.0, 0.0))])

    apply_stage3([track], pose, config)

    assert all(d.tag == Tag.REPORTED for d in track.detections)


def test_apply_stage3_runs_all_three_rules_together():
    config = Config(
        max_speed_px_per_frame=50.0,
        min_supported_track_length=3,
        static_px_threshold=4.0,
        min_static_run_frames=3,
    )
    pose = _moving_pose(10, speed=1.0)

    jump_track = _track(0, [(0, (0.0, 0.0)), (1, (30.0, 0.0)), (2, (500.0, 0.0))])
    flicker_track = _track(1, [(0, (900.0, 900.0)), (1, (905.0, 900.0))])
    static_track = _track(2, [(0, (200.0, 200.0)), (1, (201.0, 200.0)), (2, (202.0, 200.0))])
    clean_track = _track(3, [(0, (0.0, 500.0)), (1, (30.0, 500.0)), (2, (60.0, 500.0))])

    apply_stage3([jump_track, flicker_track, static_track, clean_track], pose, config)

    assert [d.tag for d in jump_track.detections] == [Tag.REPORTED, Tag.REPORTED, Tag.REJECTED]
    assert all(d.tag == Tag.REJECTED for d in flicker_track.detections)
    assert all(d.tag == Tag.REJECTED for d in static_track.detections)
    assert all(d.tag == Tag.REPORTED for d in clean_track.detections)
