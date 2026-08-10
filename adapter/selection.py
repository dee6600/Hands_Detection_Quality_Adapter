"""Milestone 6: instance-cap selection after association.

Runs last, after stages 1-4 (and therefore after `rejected`/`interpolated`
tags already reflect everything earlier stages know). Per spec S3: a
detector asked for the class maximum picks by single-frame confidence,
which is exactly what's unreliable here -- a duplicate or background object
can outscore a real one on a given frame. So the adapter requests a pool
slightly above the maximum (`Config.candidate_pool_size`) and enforces the
cap here instead, ranking by the TRACK supporting each candidate rather than
its score on that one frame: an isolated flicker is dropped even if it
scored well, and a long, well-supported track wins even on a frame where its
box wasn't the most confident thing around.
"""

from __future__ import annotations

from collections import defaultdict

from adapter.types import Config, Tag, Track


def _track_quality(track: Track) -> tuple[int, float]:
    """(supported detection count, mean confidence), both "higher is
    better". Count dominates -- a short-but-confident candidate is exactly
    the single-frame flicker the spec wants ranked below a real trajectory;
    confidence only breaks ties between tracks of comparable length.
    Detections already `rejected` don't count as support.
    """
    supported = [d for d in track.detections if d.tag != Tag.REJECTED]
    if not supported:
        return (0, 0.0)
    mean_confidence = sum(d.confidence for d in supported) / len(supported)
    return (len(supported), mean_confidence)


def apply_selection(tracks: list[Track], config: Config) -> list[Track]:
    """Enforce `Config.class_max_instances` on every frame. Track quality is
    computed once up front, from state at the start of selection -- ranking
    reflects how well-established each track already was, not a moving
    target as candidates get trimmed frame by frame.
    """
    quality = {id(track): _track_quality(track) for track in tracks}

    by_frame: dict[int, list[tuple[Track, object]]] = defaultdict(list)
    for track in tracks:
        for det in track.detections:
            if det.tag != Tag.REJECTED:
                by_frame[det.frame].append((track, det))

    for candidates in by_frame.values():
        if len(candidates) <= config.class_max_instances:
            continue
        candidates.sort(key=lambda pair: quality[id(pair[0])], reverse=True)
        for _, det in candidates[config.class_max_instances :]:
            det.tag = Tag.REJECTED

    return tracks
