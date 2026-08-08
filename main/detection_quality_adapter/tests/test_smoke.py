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
    detections = [{"frame": 0, "box": [0, 0, 10, 10]}, {"frame": 1, "box": [1, 1, 11, 11]}]
    out = pipeline.run_pipeline(detections)
    assert len(out) == len(detections)
    assert all(d["tag"] == "reported" for d in out)
