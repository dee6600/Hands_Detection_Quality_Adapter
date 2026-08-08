"""Milestone 0 smoke test: every module imports, no-op pipeline runs."""

from adapter import (
    association,
    geometric,
    hand_config,
    ingest,
    interpolation,
    pipeline,
    selection,
    temporal,
    types,
)
from adapter.types import Detection, Tag


def test_modules_import():
    for module in (
        types,
        ingest,
        geometric,
        association,
        temporal,
        interpolation,
        selection,
        hand_config,
        pipeline,
    ):
        assert module is not None


def test_noop_pipeline_passthrough():
    detections = [
        Detection(frame=0, xyxy=(0, 0, 10, 10), confidence=0.9),
        Detection(frame=1, xyxy=(1, 1, 11, 11), confidence=0.8),
    ]
    out = pipeline.run_pipeline(detections)
    assert len(out) == len(detections)
    assert all(d.tag == Tag.REPORTED for d in out)
