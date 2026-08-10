"""Milestone 1: hand-built fixtures exercising the shared data model."""

from adapter.types import Config, Detection, PoseSample, Tag, Track, TrackState


def test_detection_geometry_helpers():
    d = Detection(frame=5, xyxy=(100.0, 200.0, 140.0, 260.0), confidence=0.5, class_label=0)
    assert d.width == 40.0
    assert d.height == 60.0
    assert d.xywh == (100.0, 200.0, 40.0, 60.0)
    assert d.center == (120.0, 230.0)
    assert d.tag == Tag.REPORTED


def test_track_accumulates_detections_and_exposes_last():
    t = Track(track_id=1)
    assert t.last_detection is None
    assert t.last_frame is None

    t.detections.append(Detection(frame=0, xyxy=(0, 0, 10, 10), confidence=0.9))
    t.detections.append(Detection(frame=1, xyxy=(2, 2, 12, 12), confidence=0.85))

    assert t.last_frame == 1
    assert t.last_detection.confidence == 0.85
    assert t.state == TrackState.ACTIVE


def test_track_state_transitions():
    t = Track(track_id=2, state=TrackState.EXITING)
    assert t.state == TrackState.EXITING
    t.state = TrackState.ENDED
    assert t.state == TrackState.ENDED


def test_pose_sample_matches_vio_pose_json_columns():
    p = PoseSample(t=0.033, x=1.3858, y=2.9793, z=1.7948, roll=-142.72, pitch=3.27, yaw=-48.36, speed=0.562)
    assert p.t == 0.033
    assert p.speed == 0.562


def test_config_defaults_are_generic_placeholders():
    c = Config()
    assert c.plausible_size[0] < c.plausible_size[1]
    assert c.plausible_shape[0] < c.plausible_shape[1]
    assert c.candidate_pool_size > 0
    assert c.class_max_instances > 0


def test_hand_built_detection_list_through_real_pipeline():
    """Milestone 5 wired `pipeline.py` for real; this fixture (real box
    values from t010 frame 0) now exercises the actual pipeline instead of
    a no-op. Appearing on a single frame with nothing before or after, both
    detections are correctly flagged `rejected` as unsupported/flicker by
    stage 3 -- there's no track to support them, which is the right answer,
    not the old placeholder's blanket `reported`.
    """
    from adapter.pipeline import run_pipeline

    detections = [
        [
            Detection(frame=0, xyxy=(912.3, 912.16, 1245.5, 1200.0), confidence=0.78, class_label=0),
            Detection(frame=0, xyxy=(1150.79, 1048.0, 1481.36, 1199.51), confidence=0.25, class_label=1),
        ]
    ]
    pose = [PoseSample(t=0.0, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, speed=0.0)]

    tracks = run_pipeline(detections, pose)

    all_dets = [d for t in tracks for d in t.detections]
    assert len(all_dets) == 2
    assert all(d.tag == Tag.REJECTED for d in all_dets)
