"""Milestone 6: one test per row of the spec's edge-case table (S6),
using `hand_config()` instead of the generic `Config()`, on synthetic data.

Row "another person's hands in view" (stereo depth) is covered separately in
`test_stereo_depth.py`, since it needs actual video frames, not just
detections. Row "gloved or partially occluded hand" is explicitly marked
`xfail` below, per the spec's own instruction not to guess at logic there.
"""

import pytest

from adapter.geometric import apply_stage1
from adapter.hand_config import hand_config
from adapter.interpolation import apply_stage4
from adapter.pipeline import run_hand_pipeline
from adapter.types import Detection, PoseSample, Tag, TrackState


def _det(frame, xy, confidence=0.8, w=60.0, h=60.0):
    x, y = xy
    return Detection(frame=frame, xyxy=(x - w / 2, y - h / 2, x + w / 2, y + h / 2), confidence=confidence)


def _still_pose(n):
    return [PoseSample(t=i / 30.0, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, speed=0.0) for i in range(n)]


def test_plausible_size_admits_a_real_near_frame_filling_hand():
    """Milestone 7 sweep: raw box side lengths across all 39 clips run up to
    1110px, and this dataset's own README/stereo-depth calibration establish
    near-frame-filling hands as a genuine, expected pattern (hands very
    close to a head-mounted camera), not noise. The generic Config's 800px
    upper bound would incorrectly reject a box this large; hand_config()'s
    1150px bound (set from that real distribution) keeps it.
    """
    from adapter.types import Config as GenericConfig

    big_hand = _det(0, (500.0, 500.0), w=1000.0, h=900.0)  # side=1000, within (50, 1150)

    hand_survivors = apply_stage1([big_hand], hand_config())
    assert len(hand_survivors) == 1

    generic_survivors = apply_stage1([_det(0, (500.0, 500.0), w=1000.0, h=900.0)], GenericConfig())
    assert len(generic_survivors) == 0  # confirms the generic 800px cap really would reject it


def test_duplicate_boxes_on_one_hand_merge_in_the_candidate_pool():
    config = hand_config()
    strong = _det(0, (500.0, 500.0), confidence=0.9)
    weak = _det(0, (505.0, 502.0), confidence=0.4)  # near-identical -- one hand, echoed

    survivors = apply_stage1([weak, strong], config)

    assert len(survivors) == 1
    assert survivors[0].tag == Tag.MERGED


def test_hand_exits_at_a_side_border_terminates_without_interpolating():
    config = hand_config(frame_size=(1920.0, 1200.0))
    # moving right, off the side, then nothing -- a few trailing empty frames
    # so the clip doesn't just end exactly where the track does (which would
    # correctly be classified as ambiguous, not a confirmed exit -- see
    # Milestone 5's frame_count-aware skip).
    n_frames = 8
    frames: list[list[Detection]] = [[] for _ in range(n_frames)]
    for f in range(3):
        frames[f].append(_det(f, (1850.0 + 20.0 * f, 600.0)))
    tracks = run_hand_pipeline(frames, _still_pose(n_frames), config)

    track = next(t for t in tracks if t.detections)
    assert len(track.detections) == 3  # nothing fabricated
    assert track.state == TrackState.EXITING


def test_hand_passes_below_the_camera_terminates_without_interpolating():
    """The bottom-border case: occluded by the wearer's torso, so the last
    real motion isn't necessarily downward/outward -- exactly the scenario
    `hand_config()`'s bottom-border override exists for. A few trailing
    empty frames so the clip doesn't end exactly where the track does.
    """
    config = hand_config(frame_size=(1920.0, 1200.0))
    n_frames = 8
    frames: list[list[Detection]] = [[] for _ in range(n_frames)]
    frames[0].append(_det(0, (900.0, 1000.0)))
    frames[1].append(_det(1, (895.0, 1080.0)))  # drifting sideways, NOT downward
    frames[2].append(_det(2, (890.0, 1100.0)))  # last detection, near the bottom edge
    tracks = run_hand_pipeline(frames, _still_pose(n_frames), config)

    track = next(t for t in tracks if t.detections)
    assert track.state == TrackState.EXITING


def test_brief_occlusion_by_an_object_is_interpolated():
    config = hand_config()
    frames: list[list[Detection]] = [[] for _ in range(6)]
    frames[0].append(_det(0, (100.0, 500.0)))
    frames[1].append(_det(1, (120.0, 500.0)))
    # occluded, frames 2-4
    frames[5].append(_det(5, (200.0, 500.0)))

    tracks = run_hand_pipeline(frames, _still_pose(6), config)

    track = next(t for t in tracks if t.detections)
    frames_present = sorted(d.frame for d in track.detections)
    assert frames_present == [0, 1, 2, 3, 4, 5]
    assert all(d.tag == Tag.INTERPOLATED for d in track.detections if 2 <= d.frame <= 4)


def test_hands_cross_or_overlap_are_retained_as_two_detections():
    """Two similarly-sized, laterally-offset boxes (like two real crossing
    hands) must survive as two. Chosen to sit right in the gap between the
    two gates: IoU ~0.52 (above `duplicate_iou_threshold`=0.5, so this WOULD
    merge under the generic pipeline) but containment ~0.68 (below
    hand_config's tightened 0.7), so hand_config's extra gate is what saves
    it -- not just low IoU alone.
    """
    config = hand_config()
    hand_a = _det(0, (500.0, 500.0), confidence=0.8, w=60.0, h=60.0)
    hand_b = _det(0, (519.0, 500.0), confidence=0.75, w=60.0, h=60.0)

    survivors = apply_stage1([hand_a, hand_b], config)

    assert len(survivors) == 2

    from adapter.types import Config as GenericConfig

    generic_survivors = apply_stage1([_det(0, (500.0, 500.0)), _det(0, (519.0, 500.0))], GenericConfig())
    assert len(generic_survivors) == 1  # confirms this pair really would merge without the hand-specific gate


def test_hands_cross_but_a_true_duplicate_still_merges():
    """Sanity check for the test above: a real duplicate pattern (one
    strong box, one much weaker, nearly nested) still merges under
    hand_config's tighter containment gate -- the fix narrows what counts
    as a duplicate, it doesn't disable dedup.
    """
    config = hand_config()
    strong = _det(0, (500.0, 500.0), confidence=0.85, w=60.0, h=60.0)
    weak = _det(0, (505.0, 503.0), confidence=0.3, w=55.0, h=55.0)  # nearly nested inside strong

    survivors = apply_stage1([weak, strong], config)

    assert len(survivors) == 1
    assert survivors[0].tag == Tag.MERGED


def test_hand_reenters_after_long_absence_starts_a_new_track():
    config = hand_config(max_dropout_frames=3)
    frames: list[list[Detection]] = [[] for _ in range(10)]
    frames[0].append(_det(0, (100.0, 500.0)))
    frames[1].append(_det(1, (120.0, 500.0)))
    frames[8].append(_det(8, (120.0, 500.0)))  # same spot, but well past patience

    tracks = run_hand_pipeline(frames, _still_pose(10), config)

    non_empty = [t for t in tracks if t.detections]
    assert len(non_empty) == 2
    first, second = sorted(non_empty, key=lambda t: t.detections[0].frame)
    assert [d.frame for d in first.detections] == [0, 1]
    assert [d.frame for d in second.detections] == [8]


def test_motion_blur_confidence_dip_is_interpolated_not_lost():
    """A frame flagged `rejected` by stage 3 (e.g. an implausible-looking
    jump during blur) still gets recovered by stage 4 as long as the
    trajectory around it is continuous -- see Milestone 5's design notes.
    """
    config = hand_config(max_speed_px_per_frame=50.0, track_gate_speed_px_per_frame=500.0)
    frames: list[list[Detection]] = [[] for _ in range(4)]
    frames[0].append(_det(0, (100.0, 500.0)))
    frames[1].append(_det(1, (120.0, 500.0)))
    frames[2].append(_det(2, (900.0, 900.0), confidence=0.2))  # blur-induced wobble
    frames[3].append(_det(3, (160.0, 500.0)))

    tracks = run_hand_pipeline(frames, _still_pose(4), config)

    track = next(t for t in tracks if t.detections)
    assert [d.frame for d in track.detections] == [0, 1, 2, 3]
    frame2 = next(d for d in track.detections if d.frame == 2)
    assert frame2.tag == Tag.INTERPOLATED


@pytest.mark.xfail(reason="spec: gloved/partially-occluded hand behavior is unmeasured, needs labelled examples", strict=True)
def test_gloved_or_partially_occluded_hand():
    """Deliberately not implemented -- the spec explicitly says not to guess
    at logic here without labelled data. This test exists so the row isn't
    silently missing from the edge-case table's coverage; it should keep
    failing until Milestone 7 provides real examples to design against.
    """
    raise NotImplementedError("needs a labelled reference set (spec S6, S7)")
