"""Milestone 3: synthetic multi-frame sequences for the tracker.

Covers the four scenarios called out in planning.md: continuous motion, a
short gap, two objects crossing paths, and a new object entering mid-sequence
-- plus two extra cases: a gap that exceeds patience (must NOT bridge), and
an explicit check that swapped handedness labels never affect association
(the tracker must only ever use position/motion, per spec).
"""

from adapter.association import track_detections
from adapter.types import Config, Detection, TrackState


def _det(frame, xy, confidence=0.8, class_label=None, w=40.0, h=40.0):
    x, y = xy
    return Detection(frame=frame, xyxy=(x - w / 2, y - h / 2, x + w / 2, y + h / 2), confidence=confidence, class_label=class_label)


def test_continuous_motion_forms_one_track():
    frames = [[_det(i, (10.0 * i, 100.0))] for i in range(6)]

    tracks = track_detections(frames)

    assert len(tracks) == 1
    track = tracks[0]
    assert len(track.detections) == 6
    assert [d.frame for d in track.detections] == list(range(6))
    assert track.state == TrackState.ENDED


def test_short_gap_continues_same_track_not_two():
    # object seen on frames 0,1,2, missing frame 3, resumes near prediction on frame 4
    frames = [
        [_det(0, (0.0, 100.0))],
        [_det(1, (20.0, 100.0))],
        [_det(2, (40.0, 100.0))],
        [],  # dropout
        [_det(4, (80.0, 100.0))],  # resumes where velocity (20px/frame) predicts
    ]

    tracks = track_detections(frames)

    assert len(tracks) == 1
    track = tracks[0]
    assert [d.frame for d in track.detections] == [0, 1, 2, 4]


def test_gap_exceeding_patience_does_not_bridge():
    config = Config(max_dropout_frames=2)
    frames = [
        [_det(0, (0.0, 100.0))],
        [_det(1, (20.0, 100.0))],
        [],
        [],
        [],
        [],
        [_det(6, (100.0, 100.0))],  # reappears well past patience
    ]

    tracks = track_detections(frames, config)

    assert len(tracks) == 2
    first, second = sorted(tracks, key=lambda t: t.detections[0].frame)
    assert [d.frame for d in first.detections] == [0, 1]
    assert first.state == TrackState.ENDED
    assert [d.frame for d in second.detections] == [6]


def test_crossing_paths_do_not_swap_identity():
    # A moves left->right along y=100, B moves right->left along y=110.
    # Their x-order swaps around frame 1-2 ("crossing"), but velocity-based
    # prediction should keep each track glued to its own object.
    a_positions = [(0.0, 100.0), (60.0, 100.0), (120.0, 100.0), (180.0, 100.0), (240.0, 100.0)]
    b_positions = [(200.0, 110.0), (140.0, 110.0), (80.0, 110.0), (20.0, 110.0), (-40.0, 110.0)]
    frames = [
        [_det(i, a_positions[i]), _det(i, b_positions[i])]
        for i in range(5)
    ]

    tracks = track_detections(frames)

    assert len(tracks) == 2
    for track in tracks:
        xs = [d.center[0] for d in track.detections]
        # each track's x should be monotonic (either always increasing, like
        # A, or always decreasing, like B) -- a swap would break monotonicity
        assert xs == sorted(xs) or xs == sorted(xs, reverse=True)
        assert len(track.detections) == 5


def test_new_object_entering_mid_sequence_gets_its_own_track():
    frames = [
        [_det(0, (0.0, 100.0))],
        [_det(1, (20.0, 100.0))],
        [_det(2, (40.0, 100.0)), _det(2, (900.0, 900.0))],  # new, far-away object
        [_det(3, (60.0, 100.0)), _det(3, (920.0, 900.0))],
    ]

    tracks = track_detections(frames)

    assert len(tracks) == 2
    by_start = sorted(tracks, key=lambda t: t.detections[0].frame)
    original, newcomer = by_start
    assert [d.frame for d in original.detections] == [0, 1, 2, 3]
    assert [d.frame for d in newcomer.detections] == [2, 3]
    assert newcomer.track_id != original.track_id


def test_handedness_label_never_influences_association():
    """Two hands cross with their class_label (handedness guess) swapped
    exactly at the crossing frame -- if the tracker looked at class_label at
    all, this would tempt it to mis-assign. Position/velocity alone must
    still keep each track glued to its own trajectory (same check as the
    crossing-paths test, but with adversarial labels).
    """
    a_positions = [(0.0, 100.0), (60.0, 100.0), (120.0, 100.0), (180.0, 100.0)]
    b_positions = [(200.0, 110.0), (140.0, 110.0), (80.0, 110.0), (20.0, 110.0)]
    # labels swap at frame 2, right at the crossing point
    a_labels = [0, 0, 1, 1]
    b_labels = [1, 1, 0, 0]
    frames = [
        [
            _det(i, a_positions[i], class_label=a_labels[i]),
            _det(i, b_positions[i], class_label=b_labels[i]),
        ]
        for i in range(4)
    ]

    tracks = track_detections(frames)

    assert len(tracks) == 2
    for track in tracks:
        xs = [d.center[0] for d in track.detections]
        assert xs == sorted(xs) or xs == sorted(xs, reverse=True)
        assert len(track.detections) == 4
