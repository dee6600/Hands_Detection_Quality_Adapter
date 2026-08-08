"""Orchestrates the adapter stages in fixed order.

Milestone 1: no-op pass-through over real `Detection` objects. Each stage is
wired in as it's built (Milestones 2-6), in this fixed order: geometric ->
association -> temporal -> interpolation -> selection.
"""

from adapter.types import Detection, Tag


def run_pipeline(detections: list[Detection]) -> list[Detection]:
    """Pass detections through untouched, tagged `reported`."""
    for d in detections:
        d.tag = Tag.REPORTED
    return detections
