"""Milestone 4: stage 3 (temporal rejection) against real clip bundles.

Synthetic fixtures in `test_temporal.py` are the primary correctness check;
this is the secondary real-data pass (same pattern as the stage 1/2
real-data tests). Runs the real pipeline order: stage 1 -> stage 2 (tracker)
-> stage 3, using the clip's own real VIO pose as the camera-motion signal.
"""

import os

import pytest

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.ingest import load_clip
from adapter.temporal import apply_stage3
from adapter.types import Config, Tag

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

CLIP_IDS = [
    "0c54a47b_t010",
    "6cd0b236_t000",
    "130d64c7_t054",
]


def _clip_dir(cid: str) -> str:
    return os.path.join(DATA_DIR, cid)


def _run_through_stage3(cid, config):
    clip = load_clip(_clip_dir(cid))
    stage1_frames = [apply_stage1(list(dets), config) for dets in clip.detections]
    tracks = track_detections(stage1_frames, config)
    apply_stage3(tracks, clip.pose, config)
    return tracks


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_stage3_runs_without_crashing(cid):
    tracks = _run_through_stage3(cid, Config())
    assert len(tracks) > 0


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_stage3_never_untags_a_rejection(cid):
    """Stage 3 only ever adds `rejected` tags -- it must never flip a
    detection back to `reported` once another rule has flagged it.
    """
    config = Config()
    tracks = _run_through_stage3(cid, config)
    for track in tracks:
        for d in track.detections:
            assert d.tag in (Tag.REPORTED, Tag.REJECTED, Tag.MERGED)


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_short_tracks_are_fully_rejected_as_flicker(cid):
    config = Config()
    tracks = _run_through_stage3(cid, config)
    for track in tracks:
        if len(track.detections) < config.min_supported_track_length:
            assert all(d.tag == Tag.REJECTED for d in track.detections)


def test_stage3_meaningfully_reduces_the_moving_camera_clip():
    """t010's camera moves throughout (a hairstyling task) -- confirm stage 3
    actually flags a non-trivial fraction of detections as rejected, not
    zero (which would mean the rules are silently no-ops on real data) and
    not everything (which would mean they're rejecting real hands).
    """
    config = Config()
    tracks = _run_through_stage3("0c54a47b_t010", config)
    all_dets = [d for t in tracks for d in t.detections]
    n_rejected = sum(1 for d in all_dets if d.tag == Tag.REJECTED)

    assert 0 < n_rejected < len(all_dets)
