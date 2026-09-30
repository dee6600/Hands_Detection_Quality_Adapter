"""Milestone 0 smoke test: every module imports, pipeline runs end to end.

Milestone 5 wired `pipeline.py` for real, superseding the Milestone 0/1
no-op placeholder this file originally checked -- see `tests/test_pipeline_
synthetic.py` for actual multi-stage correctness coverage; this file stays a
broad "does it run at all" check.
"""

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
from adapter.types import Detection, PoseSample, Track


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


def test_pipeline_runs_end_to_end_on_a_minimal_synthetic_clip():
    detections = [
        [Detection(frame=0, xyxy=(100.0, 100.0, 140.0, 140.0), confidence=0.9)],
        [Detection(frame=1, xyxy=(110.0, 100.0, 150.0, 140.0), confidence=0.8)],
        [Detection(frame=2, xyxy=(120.0, 100.0, 160.0, 140.0), confidence=0.85)],
    ]
    pose = [PoseSample(t=i / 30.0, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, speed=0.0) for i in range(3)]

    out = pipeline.run_pipeline(detections, pose)

    assert isinstance(out, list)
    assert all(isinstance(t, Track) for t in out)
