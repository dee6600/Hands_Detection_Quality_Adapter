"""Milestone 5 exit criteria: the full generic Part 1 pipeline (stages 1-4)
runs end to end on one synthetic multi-frame sequence and produces correctly
tagged output. One deliberately busy synthetic clip, several objects with
different fates, checked against `run_pipeline` directly rather than one
stage at a time.
"""

from adapter.pipeline import run_pipeline
from adapter.types import Config, Detection, PoseSample, Tag, TrackState


def _det(frame, xy, confidence=0.8, w=40.0, h=40.0):
    x, y = xy
    return Detection(frame=frame, xyxy=(x - w / 2, y - h / 2, x + w / 2, y + h / 2), confidence=confidence)


def _still_pose(n):
    return [PoseSample(t=i / 30.0, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, speed=0.0) for i in range(n)]


def test_full_pipeline_on_a_synthetic_clip_with_several_object_fates():
    n_frames = 12
    frames: list[list[Detection]] = [[] for _ in range(n_frames)]

    # Object A: duplicate pair on frame 0 (should merge), then a clean,
    # continuous, uneventful track through frame 11 (survives everything).
    frames[0].append(_det(0, (400.0, 400.0), confidence=0.9))
    frames[0].append(_det(0, (405.0, 402.0), confidence=0.4))  # near-duplicate of the box above
    for f in range(1, n_frames):
        frames[f].append(_det(f, (400.0 + 10.0 * f, 400.0), confidence=0.85))

    # Object B: occluded for frames 3-5, resumes near prediction -> interpolated.
    b_positions = {0: (100.0, 700.0), 1: (120.0, 700.0), 2: (140.0, 700.0), 6: (260.0, 700.0)}
    for f, xy in b_positions.items():
        frames[f].append(_det(f, xy, confidence=0.8))
    for f in range(7, n_frames):
        frames[f].append(_det(f, (260.0 + 20.0 * (f - 6), 700.0), confidence=0.8))

    # Object C: a flicker -- a single isolated detection, no track around it.
    frames[4].append(_det(4, (1500.0, 300.0), confidence=0.6))

    config = Config()
    tracks = run_pipeline(frames, _still_pose(n_frames), config)

    # Object A and B both start on frame 0, so look each up by its own
    # trajectory rather than a frame-0-keyed dict (which would collide).
    track_a = next(t for t in tracks if t.detections and t.detections[0].center == (400.0, 400.0))
    by_first_frame = {t.detections[0].frame: t for t in tracks if t.detections and t is not track_a}

    # Object A: the duplicate collapses to one detection on frame 0 (tagged
    # `merged`), and the track survives cleanly through the rest of the clip.
    frame0_dets = [d for d in track_a.detections if d.frame == 0]
    assert len(frame0_dets) == 1
    assert frame0_dets[0].tag == Tag.MERGED
    assert len(track_a.detections) == n_frames
    assert all(d.tag in (Tag.MERGED, Tag.REPORTED) for d in track_a.detections)

    # Object B: frames 3-5 got fabricated back in as `interpolated`.
    track_b = next(t for t in tracks if t.detections and t.detections[0].center == (100.0, 700.0))
    frames_present = sorted(d.frame for d in track_b.detections)
    assert frames_present == list(range(0, n_frames))
    filled = [d for d in track_b.detections if 3 <= d.frame <= 5]
    assert len(filled) == 3
    assert all(d.tag == Tag.INTERPOLATED for d in filled)

    # Object C: an isolated one-frame flicker, rejected as unsupported.
    track_c = by_first_frame[4]
    assert len(track_c.detections) == 1
    assert track_c.detections[0].tag == Tag.REJECTED

    # Every track ends up in a terminal state -- this is a closed batch.
    assert all(t.state in (TrackState.ENDED, TrackState.EXITING) for t in tracks)


def test_full_pipeline_exit_at_border_is_not_interpolated():
    n_frames = 10
    frames: list[list[Detection]] = [[] for _ in range(n_frames)]
    # moves toward the right border and exits; nothing at all afterward
    for f in range(4):
        frames[f].append(_det(f, (1880.0 + 10.0 * f, 500.0), confidence=0.8))

    config = Config()
    tracks = run_pipeline(frames, _still_pose(n_frames), config)

    track = next(t for t in tracks if t.detections)
    assert len(track.detections) == 4  # nothing fabricated after the exit
    assert track.state == TrackState.EXITING
