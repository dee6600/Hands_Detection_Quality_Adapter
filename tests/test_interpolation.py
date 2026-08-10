"""Milestone 5: hand-built synthetic tracks for stage 4 (interpolation + exit).

Covers the four scenarios from planning.md (exit at a border, brief occlusion
resumed near prediction, gap exceeding max dropout, gap resuming far from
prediction) plus the design notes added at the spec cross-check checkpoint:
`rejected` detections must be treated as gap material, and a border can be
configured to skip the outward-motion requirement (the bottom-border/torso-
occlusion case Milestone 6 will actually turn on for hands).

Positions in the non-exit tests deliberately stay well clear of any frame
border (well past `exit_border_margin_px`) so the exit test never
incidentally fires and confuses what's actually being checked.
"""

from adapter.interpolation import apply_stage4
from adapter.types import Config, Detection, Tag, Track, TrackState


def _det(frame, xy, confidence=0.8, tag=Tag.REPORTED, w=40.0, h=40.0):
    x, y = xy
    return Detection(frame=frame, xyxy=(x - w / 2, y - h / 2, x + w / 2, y + h / 2), confidence=confidence, tag=tag)


def _track(track_id, dets, state=TrackState.ENDED):
    t = Track(track_id=track_id, state=state)
    t.detections = list(dets)
    return t


def test_exit_at_border_with_outward_motion_is_not_interpolated():
    """A track approaching the right border, moving outward, then a LATER
    detection appears near where the trajectory would have predicted (as if
    by coincidence) -- this must NOT be bridged, because the pre-gap anchor
    already confirms an exit.
    """
    config = Config(frame_size=(1920.0, 1200.0), exit_border_margin_px=20.0, max_dropout_frames=10)
    dets = [
        _det(0, (1850.0, 600.0)),
        _det(1, (1870.0, 600.0)),
        _det(2, (1905.0, 600.0)),  # within 20px of the right border, moving outward (rightward)
        _det(8, (1905.0, 600.0)),  # "coincidentally" resumes at a plausible spot
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    frames_present = [d.frame for d in track.detections]
    assert frames_present == [0, 1, 2, 8]  # no frames 3-7 fabricated
    assert all(d.tag != Tag.INTERPOLATED for d in track.detections)


def test_terminal_exit_marks_track_exiting():
    config = Config(frame_size=(1920.0, 1200.0), exit_border_margin_px=20.0)
    dets = [
        _det(0, (1850.0, 600.0)),
        _det(1, (1870.0, 600.0)),
        _det(2, (1905.0, 600.0)),  # last detection, near right border, moving outward
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    assert track.state == TrackState.EXITING


def test_occluded_mid_frame_resumes_near_prediction_is_interpolated():
    config = Config(max_dropout_frames=10, track_gate_speed_px_per_frame=100.0)
    dets = [
        _det(0, (100.0, 500.0)),
        _det(1, (120.0, 500.0)),  # velocity (20, 0) px/frame established here
        # occluded for frames 2-4
        _det(5, (200.0, 500.0)),  # resumes exactly where 20px/frame x 4 predicts
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    frames_present = [d.frame for d in track.detections]
    assert frames_present == [0, 1, 2, 3, 4, 5]
    filled = [d for d in track.detections if 2 <= d.frame <= 4]
    assert all(d.tag == Tag.INTERPOLATED for d in filled)
    # linear interpolation between the actual endpoints (120@f1, 200@f5):
    # frame 2 sits 1/4 of the way from 120 to 200
    frame2 = next(d for d in track.detections if d.frame == 2)
    assert frame2.center[0] == 140.0
    assert frame2.center[1] == 500.0


def test_gap_exceeding_max_dropout_is_left_untouched():
    config = Config(max_dropout_frames=3, track_gate_speed_px_per_frame=1000.0)
    dets = [
        _det(0, (100.0, 500.0)),
        _det(1, (120.0, 500.0)),
        _det(10, (300.0, 500.0)),  # 9-frame gap, well past max_dropout_frames=3
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    frames_present = [d.frame for d in track.detections]
    assert frames_present == [0, 1, 10]  # nothing fabricated in between


def test_gap_resuming_far_from_prediction_is_left_untouched():
    config = Config(max_dropout_frames=10, track_gate_speed_px_per_frame=50.0)
    dets = [
        _det(0, (100.0, 500.0)),
        _det(1, (120.0, 500.0)),  # velocity (20, 0) -- predicts frame 4 at x=180
        _det(4, (900.0, 900.0)),  # nowhere near the prediction -- a different object
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    frames_present = [d.frame for d in track.detections]
    assert frames_present == [0, 1, 4]


def test_rejected_detections_are_treated_as_gap_material():
    """A detection tagged `rejected` by Milestone 4 (e.g. a motion-blur
    displacement flag) sits between two trustworthy anchors. Its position
    shouldn't be trusted, but the gap it's part of should still be eligible
    for interpolation -- and filling it OVERWRITES the rejected box rather
    than leaving a spurious untrustworthy one alongside a new one.
    """
    config = Config(max_dropout_frames=10, track_gate_speed_px_per_frame=200.0)
    dets = [
        _det(0, (100.0, 500.0)),
        _det(1, (120.0, 500.0)),  # velocity (20, 0) established here
        _det(2, (900.0, 900.0), tag=Tag.REJECTED),  # implausible wobble, already flagged
        _det(3, (160.0, 500.0)),  # consistent with continuing at 20px/frame
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    assert [d.frame for d in track.detections] == [0, 1, 2, 3]
    frame2 = next(d for d in track.detections if d.frame == 2)
    assert frame2.tag == Tag.INTERPOLATED
    assert frame2.center == (140.0, 500.0)  # linear interpolation between (120,500)@f1 and (160,500)@f3


def test_bottom_border_can_skip_outward_motion_requirement():
    """Mechanism-level test for the Milestone 6 hand override: a track
    disappearing near the bottom border with velocity that is NOT outward
    (simulating torso occlusion, not a clean walk-out-of-frame) still counts
    as an exit once that border's outward-motion requirement is turned off.
    """
    config = Config(
        frame_size=(1920.0, 1200.0),
        exit_border_margin_overrides={"bottom": 150.0},
        exit_requires_outward_motion={"bottom": False},
    )
    dets = [
        _det(0, (900.0, 1000.0)),
        _det(1, (895.0, 1080.0)),  # drifting slightly sideways/up, NOT downward-outward
        _det(2, (890.0, 1100.0)),  # last detection, within 150px of the bottom edge (1200)
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    assert track.state == TrackState.EXITING


def test_bottom_border_exit_not_detected_without_the_override():
    """Same geometry as above, but with generic (non-hand) config: the
    default border margin is much tighter and outward motion is required
    everywhere, so this does NOT count as a confirmed exit -- it's an
    ambiguous dropout instead.
    """
    config = Config(frame_size=(1920.0, 1200.0))
    dets = [
        _det(0, (900.0, 1000.0)),
        _det(1, (895.0, 1080.0)),
        _det(2, (890.0, 1100.0)),
    ]
    track = _track(0, dets)

    apply_stage4([track], config)

    assert track.state == TrackState.ENDED


def test_track_running_to_clip_end_is_not_misclassified_as_exit():
    """Even though the last detection sits right at the border with outward
    motion, if it's also the clip's literal last frame, there's no dropout
    to explain -- the recording ended, not the object leaving it.
    """
    config = Config(frame_size=(1920.0, 1200.0), exit_border_margin_px=20.0)
    dets = [
        _det(0, (1850.0, 600.0)),
        _det(1, (1870.0, 600.0)),
        _det(2, (1905.0, 600.0)),
    ]
    track = _track(0, dets)

    apply_stage4([track], config, frame_count=3)  # last detection IS frame_count - 1

    assert track.state == TrackState.ENDED


def test_clean_continuous_track_is_untouched():
    config = Config()
    dets = [_det(i, (100.0 + 10.0 * i, 500.0)) for i in range(5)]
    track = _track(0, dets)

    apply_stage4([track], config)

    assert [d.frame for d in track.detections] == [0, 1, 2, 3, 4]
    assert all(d.tag == Tag.REPORTED for d in track.detections)
    assert track.state == TrackState.ENDED
