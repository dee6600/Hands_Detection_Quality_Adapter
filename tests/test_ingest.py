"""Milestone 1.5: `load_clip` against real bundles.

Deliberately exercises more than one clip so nothing is tuned to a single
recording's quirks. `6cd0b236_t000` has zero zero-detection frames (a dense
clip); `130d64c7_t054` and `7fe10737_t009` both have real gaps (82 and 123
zero-detection frames respectively), exercising the reconstruction path.
`0c54a47b_t010` is kept too for continuity with earlier exploration, but is
not the only case any assertion depends on. Expected values are read
straight from each bundle's own raw JSON, not hardcoded, so nothing here is
invented placeholder data.
"""

import json
import os

import pytest

from adapter.ingest import load_clip
from adapter.types import Detection, PoseSample

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CLIP_IDS = [
    "6cd0b236_t000",
    "130d64c7_t054",
    "7fe10737_t009",
    "0c54a47b_t010",
]


def _clip_dir(cid: str) -> str:
    return os.path.join(DATA_DIR, cid)


def _raw(cid: str, name: str) -> dict:
    with open(os.path.join(_clip_dir(cid), name)) as f:
        return json.load(f)


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_frame_count_matches_meta_and_frame_ts(cid):
    meta = _raw(cid, "meta.json")
    frame_ts_raw = _raw(cid, "frame_ts.json")
    clip = load_clip(_clip_dir(cid))

    frame_count = frame_ts_raw["frame_count"]
    assert frame_count == meta["hand_boxes"]["video_frame_count"]
    assert len(clip.detections) == frame_count
    assert len(clip.pose) == frame_count
    assert len(clip.frame_ts) == frame_count


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_zero_detection_frames_become_empty_lists(cid):
    hand_boxes_raw = _raw(cid, "hand_boxes.json")
    clip = load_clip(_clip_dir(cid))

    frames_with_boxes = {f["frame"] for f in hand_boxes_raw["frames"]}
    frame_count = len(clip.detections)
    expected_zero_frames = frame_count - hand_boxes_raw["frames_with_detections"]
    actual_zero_frames = sum(1 for dets in clip.detections if len(dets) == 0)

    assert actual_zero_frames == expected_zero_frames
    for i in range(frame_count):
        if i not in frames_with_boxes:
            assert clip.detections[i] == []
        else:
            assert len(clip.detections[i]) > 0


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_detections_carry_real_box_values_through(cid):
    hand_boxes_raw = _raw(cid, "hand_boxes.json")
    clip = load_clip(_clip_dir(cid))

    first_frame_with_boxes = hand_boxes_raw["frames"][0]
    frame_idx = first_frame_with_boxes["frame"]
    raw_dets = first_frame_with_boxes["detections"]

    loaded_dets = clip.detections[frame_idx]
    assert len(loaded_dets) == len(raw_dets)
    for loaded, raw in zip(loaded_dets, raw_dets):
        assert isinstance(loaded, Detection)
        assert loaded.frame == frame_idx
        assert loaded.xyxy == tuple(raw["xyxy"])
        assert loaded.confidence == raw["confidence"]
        assert loaded.class_label == raw["class"]


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_pose_aligns_with_frames_and_drops_trailing_sample(cid):
    vio_raw = _raw(cid, "vio_pose.json")
    clip = load_clip(_clip_dir(cid))

    frame_count = len(clip.pose)
    assert vio_raw["n"] == frame_count + 1

    for i in (0, frame_count // 2, frame_count - 1):
        sample = clip.pose[i]
        assert isinstance(sample, PoseSample)
        assert sample.t == vio_raw["t"][i]
        assert sample.x == vio_raw["x"][i]
        assert sample.yaw == vio_raw["yaw"][i]
        assert sample.speed == vio_raw["speed"][i]


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_frame_ts_is_unix_ns_keyed_by_frame_index(cid):
    frame_ts_raw = _raw(cid, "frame_ts.json")
    clip = load_clip(_clip_dir(cid))

    for key in ("0", str(frame_ts_raw["frame_count"] - 1)):
        idx = int(key)
        assert clip.frame_ts[idx] == frame_ts_raw["frame_ts"][key]


@pytest.mark.parametrize("cid", CLIP_IDS)
def test_meta_passed_through_as_is(cid):
    meta_raw = _raw(cid, "meta.json")
    clip = load_clip(_clip_dir(cid))
    assert clip.meta == meta_raw


def test_handedness_class_label_carried_but_not_relied_on():
    """Spot check on the user-suggested t000 clip: class 0/1 passes through
    for debugging but load_clip itself makes no claim about its stability
    (downstream stages must ignore it for identity, per spec).
    """
    clip = load_clip(_clip_dir("6cd0b236_t000"))
    first_populated = next(dets for dets in clip.detections if dets)
    assert all(d.class_label in (0, 1) for d in first_populated)
