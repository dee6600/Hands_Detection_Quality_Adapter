"""Milestone 3: tracker sanity checks against real clip bundles.

Synthetic sequences in `test_association.py` are the primary correctness
check; this is the secondary real-data pass (same pattern as
`test_geometric_real_data.py`). Real detections have no ground-truth track
labels yet (that needs Milestone 7), so these are structural invariants the
tracker must hold, not "is this the correct hand" checks. Detections are run
through stage 1 (geometric rejection) first, matching the real pipeline
order.
"""

import os

import pytest

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.ingest import load_clip
from adapter.types import Config, TrackState

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

CLIP_IDS = [
    "0c54a47b_t010",
    "6cd0b236_t000",
    "130d64c7_t054",
]


def _clip_dir(cid: str) -> str:
    return os.path.join(DATA_DIR, cid)


def _stage1_frames(cid, config):
    clip = load_clip(_clip_dir(cid))
    return [apply_stage1(list(dets), config) for dets in clip.detections]


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_tracker_runs_without_crashing(cid):
    config = Config()
    frames = _stage1_frames(cid, config)
    tracks = track_detections(frames, config)
    assert len(tracks) > 0


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_every_track_ends_up_ended_and_has_unique_id(cid):
    config = Config()
    frames = _stage1_frames(cid, config)
    tracks = track_detections(frames, config)

    assert all(t.state == TrackState.ENDED for t in tracks)
    ids = [t.track_id for t in tracks]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_track_detections_are_frame_ordered_and_within_dropout_gaps(cid):
    """Each track's own detections must be strictly frame-increasing, and
    consecutive detections within one track can never be separated by more
    than `max_dropout_frames` (else the tracker itself has bridged a gap it
    shouldn't have).
    """
    config = Config()
    frames = _stage1_frames(cid, config)
    tracks = track_detections(frames, config)

    for track in tracks:
        track_frames = [d.frame for d in track.detections]
        assert track_frames == sorted(set(track_frames))
        for prev, nxt in zip(track_frames, track_frames[1:]):
            assert nxt - prev <= config.max_dropout_frames


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_every_surviving_detection_is_assigned_to_exactly_one_track(cid):
    config = Config()
    frames = _stage1_frames(cid, config)
    tracks = track_detections(frames, config)

    total_survivors = sum(len(f) for f in frames)
    total_tracked = sum(len(t.detections) for t in tracks)
    assert total_tracked == total_survivors

    seen_ids = set()
    for track in tracks:
        for det in track.detections:
            assert id(det) not in seen_ids
            seen_ids.add(id(det))
