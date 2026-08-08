"""Orchestrates the adapter stages in fixed order: geometric (stage 1) ->
association (stage 2) -> temporal (stage 3) -> interpolation (stage 4) ->
selection (stage 5, hand-specialized pipeline only).
"""

from __future__ import annotations

from adapter.association import track_detections
from adapter.geometric import apply_stage1
from adapter.hand_config import hand_config
from adapter.interpolation import apply_stage4
from adapter.selection import apply_selection
from adapter.temporal import apply_stage3
from adapter.types import Config, Detection, PoseSample, Track


def run_pipeline(
    detections: list[list[Detection]],
    pose: list[PoseSample],
    config: Config | None = None,
) -> list[Track]:
    """Run the full generic Part 1 pipeline (stages 1-4) on one clip.

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


def run_hand_pipeline(
    detections: list[list[Detection]],
    pose: list[PoseSample],
    config: Config | None = None,
    video_left_path: str | None = None,
    video_right_path: str | None = None,
) -> list[Track]:
    """Run the hand-specialized pipeline (Milestone 6, Part 2): stages 1-4
    exactly as `run_pipeline`, then stage 5 (`selection.py`) enforcing the
    low, constantly-reached per-frame instance cap by track quality. Uses
    `hand_config()` by default instead of the generic `Config()`.

    The stereo-depth "beyond arm's reach" rule (stage 6) needs the clip's
    actual video frames, not just detections/pose, so it only runs when
    both `video_left_path` and `video_right_path` are given -- silently
    skipped otherwise (selection still ran, so output is still valid, just
    without the bystander-hand check). Also skipped if `config.max_reach_m`
    is `None`, even with video paths given.
    """
    config = config or hand_config()
    tracks = run_pipeline(detections, pose, config)
    apply_selection(tracks, config)

    if video_left_path is not None and video_right_path is not None:
        from adapter.stereo_depth import apply_stereo_depth_stage

        apply_stereo_depth_stage(tracks, video_left_path, video_right_path, config)

    return tracks
