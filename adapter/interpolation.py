"""Milestone 5: stage 4 - gap fill + exit detection.

Runs last, after tracks exist (stage 2) and have been tagged (stage 3). For
each track, walks its trustworthy detections (anything not already tagged
`rejected` -- a rejected detection's position can't be trusted, so it's
treated as gap material to potentially re-fill, exactly like a frame with no
detection at all; see planning.md's Milestone 5 design notes) and, for every
gap between two trustworthy anchors:

  1. Exit test first, always: is the anchor *before* the gap near a frame
     border with motion carrying it further out? If so, the object left the
     frame -- never fill this gap, no matter how well the far side matches.
  2. Otherwise, only fill if the gap is short enough (`max_dropout_frames`)
     and the resumed position lands close enough to where the pre-gap
     trajectory predicted (`track_gate_speed_px_per_frame`, the same
     question association.py already asks when building tracks -- kept as
     an independent check here so this stage is correctly testable on its
     own hand-built fixtures, not just tracks our own tracker produced).
  3. A gap that fails either test is left exactly as it was -- no fabricated
     detections, no forced closure, per spec.

A track's *final* gap (nothing follows within this track) is handled the
same way, but there's nothing to interpolate toward -- the only thing to
decide is whether it counts as a confirmed exit (`TrackState.EXITING`) or an
unexplained, ambiguous dropout (state left as `ENDED`, untouched). If
`frame_count` is given and the track's last detection is already the clip's
last frame, there's no dropout to explain at all -- the recording simply
ended, not the object leaving it.
"""

from __future__ import annotations

import math

from adapter.types import Config, Detection, Tag, Track, TrackState

_BORDERS = ("left", "right", "top", "bottom")


def _distances_to_borders(xyxy: tuple[float, float, float, float], frame_size: tuple[float, float]) -> dict[str, float]:
    """Distance from each of the box's own EDGES to the matching frame edge
    -- not the box center. Real boxes in this dataset are often large/
    near-frame-filling, so a box can be touching or past the frame edge
    while its center is still 100+ px inland; center-based distance missed
    every real exit in initial real-clip testing (a box with y2=1200,
    exactly at the 1200px-tall frame's bottom edge, had center_y only
    ~1093 -- 107px short of the default 20px margin). Edge distance can go
    negative if the box already extends past the frame edge, which is fine:
    still "at or past the border," just more so.
    """
    x1, y1, x2, y2 = xyxy
    w, h = frame_size
    return {"left": x1, "right": w - x2, "top": y1, "bottom": h - y2}


def _nearest_border(xyxy: tuple[float, float, float, float], frame_size: tuple[float, float]) -> tuple[str, float]:
    distances = _distances_to_borders(xyxy, frame_size)
    border = min(distances, key=distances.get)
    return border, distances[border]


def _margin_for(border: str, config: Config) -> float:
    return config.exit_border_margin_overrides.get(border, config.exit_border_margin_px)


def _requires_outward_motion(border: str, config: Config) -> bool:
    return config.exit_requires_outward_motion.get(border, True)


def _is_outward(border: str, velocity: tuple[float, float]) -> bool:
    vx, vy = velocity
    eps = 1e-9
    if border == "left":
        return vx < -eps
    if border == "right":
        return vx > eps
    if border == "top":
        return vy < -eps
    if border == "bottom":
        return vy > eps
    return False


def _is_exit(detection: Detection, velocity: tuple[float, float], config: Config) -> bool:
    border, dist = _nearest_border(detection.xyxy, config.frame_size)
    if dist > _margin_for(border, config):
        return False
    if _requires_outward_motion(border, config):
        return _is_outward(border, velocity)
    return True


def _velocity(a: Detection, b: Detection) -> tuple[float, float]:
    dt = b.frame - a.frame
    if dt <= 0:
        return (0.0, 0.0)
    return ((b.center[0] - a.center[0]) / dt, (b.center[1] - a.center[1]) / dt)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _fill_gap(track: Track, prev: Detection, curr: Detection) -> None:
    """Linearly interpolate position and size for every frame strictly
    between `prev` and `curr`. A frame that already has a (rejected)
    Detection there gets its box overwritten; a truly missing frame gets a
    brand new one. Confidence is the average of the two anchors -- the spec
    doesn't prescribe a value, only that consumers can tell it's `interpolated`.
    """
    dt_total = curr.frame - prev.frame
    by_frame = {d.frame: d for d in track.detections}
    for f in range(prev.frame + 1, curr.frame):
        t = (f - prev.frame) / dt_total
        cx = prev.center[0] + (curr.center[0] - prev.center[0]) * t
        cy = prev.center[1] + (curr.center[1] - prev.center[1]) * t
        w = prev.width + (curr.width - prev.width) * t
        h = prev.height + (curr.height - prev.height) * t
        xyxy = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
        confidence = (prev.confidence + curr.confidence) / 2.0

        existing = by_frame.get(f)
        if existing is not None:
            existing.xyxy = xyxy
            existing.tag = Tag.INTERPOLATED
        else:
            track.detections.append(Detection(frame=f, xyxy=xyxy, confidence=confidence, tag=Tag.INTERPOLATED))

    track.detections.sort(key=lambda d: d.frame)


def _process_track(track: Track, config: Config, frame_count: int | None) -> None:
    anchors = [d for d in track.detections if d.tag != Tag.REJECTED]
    if not anchors:
        return

    for i in range(len(anchors) - 1):
        prev, curr = anchors[i], anchors[i + 1]
        dt = curr.frame - prev.frame
        if dt <= 1:
            continue  # no gap between these two anchors

        # velocity LEADING INTO prev, from whatever came before it -- not
        # prev-to-curr, which would make "does curr match the prediction"
        # tautologically true (curr is one of the two points defining it).
        incoming_velocity = _velocity(anchors[i - 1], prev) if i > 0 else (0.0, 0.0)

        if _is_exit(prev, incoming_velocity, config):
            continue  # confirmed exit -- never fill, regardless of what follows
        if dt > config.max_dropout_frames:
            continue  # too long a break to trust
        predicted = (prev.center[0] + incoming_velocity[0] * dt, prev.center[1] + incoming_velocity[1] * dt)
        if _distance(predicted, curr.center) > config.track_gate_speed_px_per_frame * dt:
            continue  # resumed too far from prediction -- not the same object
        _fill_gap(track, prev, curr)

    last = anchors[-1]
    if frame_count is not None and last.frame >= frame_count - 1:
        return  # track ran to the end of the clip -- nothing to classify

    incoming_velocity_at_last = _velocity(anchors[-2], last) if len(anchors) >= 2 else (0.0, 0.0)
    if _is_exit(last, incoming_velocity_at_last, config):
        track.state = TrackState.EXITING
    # else: ambiguous, unresolved dropout -- left untouched, state stays ENDED


def apply_stage4(tracks: list[Track], config: Config | None = None, frame_count: int | None = None) -> list[Track]:
    """Run gap-fill + exit detection over every track. Mutates tracks in
    place (new `interpolated` Detections appended/overwritten, `state`
    updated on confirmed exits) and returns the same list.
    """
    config = config or Config()
    for track in tracks:
        _process_track(track, config, frame_count)
    return tracks
