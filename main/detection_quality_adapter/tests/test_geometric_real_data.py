"""Milestone 2: stage 1 (geometric rejection) against real clip bundles.

Synthetic fixtures in `test_geometric.py` are the primary correctness check
(per planning.md); this file is the secondary real-data sanity pass. Real
boxes don't have ground truth yet (Milestone 7), so these assertions are
structural invariants stage 1 must hold on real data, not "is this the
correct hand" checks:

  - never crashes on any frame of a real clip, including 0-box and 5-box
    frames
  - never leaves two survivors overlapping above the dedup IoU threshold
  - only ever removes detections, never invents or duplicates one
  - a real, hand-confirmed near-duplicate pair (t010 frame 5) actually
    collapses to one survivor, tagged `merged`, keeping the higher-confidence
    box
"""

import os

import pytest

from adapter.geometric import _iou, apply_stage1
from adapter.ingest import load_clip
from adapter.types import Config, Tag

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

CLIP_IDS = [
    "0c54a47b_t010",  # has a confirmed near-duplicate pair at frame 5
    "6cd0b236_t000",  # dense, no zero-detection frames
    "130d64c7_t054",  # has real detection gaps
]


def _clip_dir(cid: str) -> str:
    return os.path.join(DATA_DIR, cid)


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_stage1_runs_on_every_frame_without_crashing(cid):
    clip = load_clip(_clip_dir(cid))
    config = Config()
    for frame_dets in clip.detections:
        survivors = apply_stage1(list(frame_dets), config)
        assert len(survivors) <= len(frame_dets)


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_stage1_leaves_no_residual_duplicates(cid):
    """After stage 1, no two survivors on the same frame should still overlap
    above the dedup threshold -- that would mean a duplicate slipped through.
    """
    clip = load_clip(_clip_dir(cid))
    config = Config()
    for frame_dets in clip.detections:
        survivors = apply_stage1(list(frame_dets), config)
        for i, a in enumerate(survivors):
            for b in survivors[i + 1 :]:
                assert _iou(a, b) < config.duplicate_iou_threshold


def test_stage1_collapses_the_confirmed_t010_frame5_duplicate():
    """Frame 5 of t010 has two right-hand boxes (confidence 0.20 and 0.12)
    heavily overlapping each other -- a real near-duplicate detection, not a
    synthetic stand-in. Confirm stage 1 actually merges them.
    """
    clip = load_clip(_clip_dir("0c54a47b_t010"))
    frame5 = clip.detections[5]
    assert len(frame5) == 3, "fixture assumption changed -- re-check t010 frame 5"

    survivors = apply_stage1(list(frame5), Config())

    # the two overlapping right-hand boxes collapse to one, the separate
    # left-hand box is untouched
    assert len(survivors) == 2
    merged = [d for d in survivors if d.tag == Tag.MERGED]
    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(0.20)


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_stage1_never_increases_detection_count_across_whole_clip(cid):
    clip = load_clip(_clip_dir(cid))
    config = Config()
    total_in = sum(len(f) for f in clip.detections)
    total_out = sum(len(apply_stage1(list(f), config)) for f in clip.detections)
    assert total_out <= total_in
