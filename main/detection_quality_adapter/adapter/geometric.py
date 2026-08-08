"""Milestone 2: stage 1 - duplicate/size/shape rejection.

Per-frame only, no tracking. Fixed order (spec S3):
  1. reject_duplicates       - heavily overlapping boxes -> merge, keep the
                                stronger, tag `merged`.
  2. reject_implausible_size - box far outside plausible size -> drop, tag
                                `rejected`.
  3. reject_implausible_shape - box markedly more elongated than plausible ->
                                drop, tag `rejected`.

`apply_stage1` runs all three in order. Detections not touched by any rule
keep their incoming tag (`reported` at input).
"""

from __future__ import annotations

from adapter.types import Config, Detection, Tag


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def reject_duplicates(detections: list[Detection], config: Config) -> list[Detection]:
    """Greedy NMS: process boxes highest-confidence first; any lower-confidence
    box overlapping an already-kept box above `duplicate_iou_threshold` is
    treated as a duplicate of it and dropped. The surviving box is tagged
    `merged` so the output makes clear a duplicate was absorbed into it.
    Boxes with no duplicate keep their incoming tag untouched.

    Returns survivors in their original input order.
    """
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        winner = next((k for k in kept if _iou(det, k) >= config.duplicate_iou_threshold), None)
        if winner is None:
            kept.append(det)
        else:
            winner.tag = Tag.MERGED

    kept_ids = {id(d) for d in kept}
    return [d for d in detections if id(d) in kept_ids]


def reject_implausible_size(detections: list[Detection], config: Config) -> list[Detection]:
    """Reject boxes whose larger side falls outside `Config.plausible_size`."""
    lo, hi = config.plausible_size
    survivors = []
    for d in detections:
        side = max(d.width, d.height)
        if lo <= side <= hi:
            survivors.append(d)
        else:
            d.tag = Tag.REJECTED
    return survivors


def reject_implausible_shape(detections: list[Detection], config: Config) -> list[Detection]:
    """Reject boxes more elongated than `Config.plausible_shape` (w/h ratio)."""
    lo, hi = config.plausible_shape
    survivors = []
    for d in detections:
        if d.height <= 0.0:
            d.tag = Tag.REJECTED
            continue
        ratio = d.width / d.height
        if lo <= ratio <= hi:
            survivors.append(d)
        else:
            d.tag = Tag.REJECTED
    return survivors


def apply_stage1(detections: list[Detection], config: Config | None = None) -> list[Detection]:
    """Run the three stage-1 rules in fixed order on one frame's detections."""
    config = config or Config()
    survivors = reject_duplicates(detections, config)
    survivors = reject_implausible_size(survivors, config)
    survivors = reject_implausible_shape(survivors, config)
    return survivors
