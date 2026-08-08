"""Orchestrates the adapter stages in fixed order: geometric (stage 1) ->
association (stage 2) -> temporal (stage 3) -> interpolation (stage 4).
Milestone 6 will add a selection step after association for the
hand-specialized pipeline; the generic Part 1 pipeline ends at stage 4.
"""

from __future__ import annotations

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.interpolation import apply_stage4
from adapter.temporal import apply_stage3
from adapter.types import Config, Detection, PoseSample, Track


def run_pipeline(
    detections: list[list[Detection]],
    pose: list[PoseSample],
    config: Config | None = None,
) -> list[Track]:
    """Run the full generic Part 1 pipeline on one clip.

    `detections[i]` is frame i's raw candidate boxes and `pose[i]` is that
    frame's VIO sample -- both match `ClipData`'s shape (`ingest.py`), so
    this can be called directly as `run_pipeline(clip.detections, clip.pose)`.
    Returns every track that ever existed, fully tagged and gap-filled.
    """
    config = config or Config()
    frame_count = len(detections)

    stage1_frames = [apply_stage1(list(frame_dets), config) for frame_dets in detections]
    tracks = track_detections(stage1_frames, config)
    apply_stage3(tracks, pose, config)
    apply_stage4(tracks, config, frame_count=frame_count)
    return tracks
