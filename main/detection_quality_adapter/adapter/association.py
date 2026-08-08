"""Milestone 3: stage 2 - the tracker (association).

Turns per-frame detection lists into persistent Tracks. Deliberately the
simplest thing that can work: greedy nearest-neighbor matching against each
active track's linearly-extrapolated predicted position. No handedness
(class_label) is ever used to decide a match -- only position/velocity, per
spec (tracker must not use the detector's unreliable left/right guess for
identity).

NOTE: greedy NN + linear prediction can mis-assign in dense/ambiguous scenes
(many overlapping tracks with near-identical predicted positions). A Kalman
filter (smoother velocity estimate under detector noise) or Hungarian
matching (globally optimal assignment instead of greedy) would improve
robustness there -- not built here per the plan; upgrade only if synthetic
or real-clip tests show greedy NN isn't good enough.
"""

from __future__ import annotations

import math

from adapter.types import Config, Detection, Track, TrackState


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _predict_position(track: Track, frame: int) -> tuple[float, float]:
    """Linearly extrapolate the track's last known position/velocity to
    `frame`. A track coasting through a gap predicts further ahead the
    longer the gap has been open.
    """
    last = track.last_detection
    dt = frame - last.frame
    vx, vy = track.predicted_velocity
    lx, ly = last.center
    return lx + vx * dt, ly + vy * dt


def _gate_radius(config: Config, dt: int) -> float:
    """Max allowed distance between a track's prediction and a candidate
    detection, scaled by frames elapsed since the track was last seen.

    Deliberately uses `track_gate_speed_px_per_frame`, not the tighter
    `max_speed_px_per_frame` plausibility threshold from temporal.py's
    displacement rule -- see Config's docstring for why those two need to be
    different numbers.
    """
    return config.track_gate_speed_px_per_frame * max(dt, 1)


def _extend_track(track: Track, det: Detection) -> None:
    last = track.last_detection
    dt = det.frame - last.frame
    if dt > 0:
        lx, ly = last.center
        dx, dy = det.center
        track.predicted_velocity = ((dx - lx) / dt, (dy - ly) / dt)
    track.detections.append(det)
    track.predicted_position = det.center
    track.state = TrackState.ACTIVE


def _start_track(det: Detection, track_id: int) -> Track:
    track = Track(track_id=track_id, state=TrackState.ACTIVE, predicted_velocity=(0.0, 0.0))
    track.detections.append(det)
    track.predicted_position = det.center
    return track


def track_detections(frames: list[list[Detection]], config: Config | None = None) -> list[Track]:
    """`frames[i]` = detections on frame i (possibly empty), in frame order.

    Returns every track that ever existed, all with `state == ENDED` (a
    clip is a closed batch -- there's no "still running" at the end).
    """
    config = config or Config()
    active: list[Track] = []
    finished: list[Track] = []
    next_id = 0

    for frame_idx, detections in enumerate(frames):
        # 1. expire tracks whose patience has already run out -- must happen
        # before matching, not after, else a stale track could still snap up
        # a detection via an ever-widening gate radius and never actually end.
        still_active = []
        for track in active:
            if frame_idx - track.last_frame > config.max_dropout_frames:
                track.state = TrackState.ENDED
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        # 2. candidate (distance, track, detection) triples within gate
        candidates = []
        for track in active:
            dt = frame_idx - track.last_frame
            predicted = _predict_position(track, frame_idx)
            radius = _gate_radius(config, dt)
            for det in detections:
                dist = _distance(predicted, det.center)
                if dist <= radius:
                    candidates.append((dist, track, det))

        # 3. greedy assignment, closest pairs first, one match each
        candidates.sort(key=lambda c: c[0])
        matched_det_ids: set[int] = set()
        for dist, track, det in candidates:
            if track.last_frame == frame_idx or id(det) in matched_det_ids:
                continue
            _extend_track(track, det)
            matched_det_ids.add(id(det))

        # 4. unmatched detections start new tracks; unmatched tracks simply
        # remain in `active`, coasting, to be re-checked for patience above
        # on the next frame.
        for det in detections:
            if id(det) not in matched_det_ids:
                active.append(_start_track(det, next_id))
                next_id += 1

    for track in active:
        track.state = TrackState.ENDED
    finished.extend(active)
    return finished
