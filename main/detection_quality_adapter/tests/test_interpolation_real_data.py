"""Milestone 5: stage 4 (interpolation + exit) against real clip bundles.

Synthetic fixtures in `test_interpolation.py` are the primary correctness
check; this is the secondary real-data pass (same pattern as the other
stages). Runs the real pipeline order: stage 1 -> stage 2 (tracker) ->
stage 3 (temporal, real VIO pose) -> stage 4, using each clip's own real
frame count.
"""

import os

import pytest

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.ingest import load_clip
from adapter.interpolation import apply_stage4
from adapter.temporal import apply_stage3
from adapter.types import Config, Tag, TrackState

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

CLIP_IDS = [
    "0c54a47b_t010",
    "6cd0b236_t000",
    "130d64c7_t054",
]


def _clip_dir(cid: str) -> str:
    return os.path.join(DATA_DIR, cid)


def _run_through_stage4(cid, config):
    clip = load_clip(_clip_dir(cid))
    frames = [apply_stage1(list(dets), config) for dets in clip.detections]
    tracks = track_detections(frames, config)
    apply_stage3(tracks, clip.pose, config)
    apply_stage4(tracks, config, frame_count=len(clip.detections))
    return tracks, len(clip.detections)


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_stage4_runs_without_crashing(cid):
    tracks, _ = _run_through_stage4(cid, Config())
    assert len(tracks) > 0


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_interpolated_detections_sit_strictly_between_real_neighbors(cid):
    """Every interpolated detection must be sandwiched between two
    non-rejected, frame-ordered real detections in its own track -- proof
    the fill logic never fabricates something dangling off one end.
    """
    tracks, _ = _run_through_stage4(cid, Config())
    for track in tracks:
        dets = sorted(track.detections, key=lambda d: d.frame)
        for i, d in enumerate(dets):
            if d.tag != Tag.INTERPOLATED:
                continue
            assert 0 < i < len(dets) - 1
            assert dets[i - 1].tag != Tag.REJECTED
            assert dets[i + 1].tag != Tag.REJECTED
            assert dets[i - 1].frame < d.frame < dets[i + 1].frame


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_no_track_exceeds_the_clips_own_frame_range(cid):
    tracks, frame_count = _run_through_stage4(cid, Config())
    for track in tracks:
        for d in track.detections:
            assert 0 <= d.frame < frame_count


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_tracks_ending_at_the_real_clips_last_frame_are_never_marked_exiting(cid):
    """A track that survives to the clip's actual last frame shouldn't be
    classified as a confirmed exit -- the recording ended, not the object.
    """
    tracks, frame_count = _run_through_stage4(cid, Config())
    for track in tracks:
        if track.detections and track.detections[-1].frame == frame_count - 1:
            assert track.state != TrackState.EXITING


def test_stage4_meaningfully_fills_some_gaps_on_a_real_clip():
    """t010 has plenty of short tracked gaps (motion blur, brief occlusion);
    confirm stage 4 actually recovers a non-trivial number of them, not zero
    (a silent no-op) and not an implausibly large fraction (fabricating).
    """
    tracks, _ = _run_through_stage4("0c54a47b_t010", Config())
    all_dets = [d for t in tracks for d in t.detections]
    n_interpolated = sum(1 for d in all_dets if d.tag == Tag.INTERPOLATED)
    assert 0 < n_interpolated < len(all_dets) * 0.2


@pytest.mark.parametrize("cid", ["0c54a47b_t010", "407258cd_t036", "130d64c7_t054"])
def test_stage4_actually_classifies_some_real_exits(cid):
    """Regression test for a real bug found while building this stage: the
    exit test originally compared box CENTER distance to the frame border,
    which misses this dataset's large/near-frame-filling boxes entirely (a
    box with its bottom edge exactly at the frame's bottom edge can still
    have its center 100+ px inland). That version of the code made this
    assertion fail on every real clip -- zero tracks were ever classified
    as EXITING despite hands clearly leaving frame in every one of them.
    Fixed by measuring from the box's own edges instead of its center.

    Deliberately not run against `6cd0b236_t000` (in `CLIP_IDS` above): it's
    a dense clip with only 2 tracks, both running to the clip's own last
    frame with no dropouts at all -- genuinely nothing to exit from, not a
    bug. This test is about clips that DO have early-ending tracks.
    """
    tracks, _ = _run_through_stage4(cid, Config())
    n_exiting = sum(1 for t in tracks if t.state == TrackState.EXITING)
    assert n_exiting > 0
