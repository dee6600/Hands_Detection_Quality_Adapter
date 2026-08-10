"""Milestone 1.5: loads a real clip bundle into the internal types.

`load_clip(clip_dir)` is the single entry point Milestones 2-6 build on top
of. See `../data/README.md` for the on-disk bundle schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from adapter.types import Detection, PoseSample

_POSE_FIELDS = ("t", "x", "y", "z", "roll", "pitch", "yaw", "speed")


@dataclass
class ClipData:
    """One clip bundle, frame-aligned and ready for the pipeline stages."""

    detections: list[list[Detection]]  # index i = detections on frame i, [] if none
    frame_ts: dict[int, int]  # frame index -> unix ns
    pose: list[PoseSample]  # index i = pose at frame i
    meta: dict


def load_clip(clip_dir: str) -> ClipData:
    meta = _load_json(clip_dir, "meta.json")
    frame_ts_raw = _load_json(clip_dir, "frame_ts.json")
    hand_boxes_raw = _load_json(clip_dir, "hand_boxes.json")
    vio_raw = _load_json(clip_dir, "vio_pose.json")

    frame_count = frame_ts_raw["frame_count"]
    frame_ts = {int(k): v for k, v in frame_ts_raw["frame_ts"].items()}

    return ClipData(
        detections=_load_detections(hand_boxes_raw, frame_count),
        frame_ts=frame_ts,
        pose=_load_pose(vio_raw, frame_count),
        meta=meta,
    )


def _load_json(clip_dir: str, name: str) -> dict:
    with open(os.path.join(clip_dir, name)) as f:
        return json.load(f)


def _load_detections(hand_boxes_raw: dict, frame_count: int) -> list[list[Detection]]:
    """Reconstruct the full [0..frame_count) sequence. `hand_boxes.json` omits
    frames with zero detections rather than listing them as empty arrays.
    """
    by_frame = {frame["frame"]: frame["detections"] for frame in hand_boxes_raw["frames"]}
    return [
        [
            Detection(
                frame=i,
                xyxy=tuple(d["xyxy"]),
                confidence=d["confidence"],
                class_label=d["class"],
            )
            for d in by_frame.get(i, [])
        ]
        for i in range(frame_count)
    ]


def _load_pose(vio_raw: dict, frame_count: int) -> list[PoseSample]:
    """`vio_pose.json` has always been observed with exactly one more sample
    than `frame_ts.json`'s frame_count (n == frame_count + 1) across all 39
    clips in the delivered set — the pose window's closing-boundary sample,
    one tick past the last frame. Pose index i aligns with frame index i for
    i in [0, frame_count); the trailing sample is dropped. Fail loudly rather
    than silently truncate/pad if a future bundle doesn't match this.
    """
    n = vio_raw["n"]
    if n != frame_count + 1:
        raise ValueError(
            f"unexpected vio_pose.json length: n={n}, frame_ts frame_count={frame_count} "
            "(expected n == frame_count + 1)"
        )
    return [
        PoseSample(**{key: vio_raw[key][i] for key in _POSE_FIELDS})
        for i in range(frame_count)
    ]
