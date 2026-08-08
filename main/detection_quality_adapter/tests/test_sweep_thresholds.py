"""Milestone 7: sweep_thresholds.py's real-data functions, structural checks
against a small clip subset (fast) -- the full 39-clip report is generated
by running the module directly (`python calibration/sweep_thresholds.py`),
not on every test run.
"""

import os

from calibration.sweep_thresholds import (
    discover_clip_ids,
    dropout_length_distribution,
    interpolated_proportion_report,
    raw_box_geometry,
    rejection_reason_frequency,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
CLIP_IDS = ["0c54a47b_t010", "407258cd_t036"]


def test_discover_clip_ids_finds_all_39():
    ids = discover_clip_ids(DATA_DIR)
    assert len(ids) == 39
    assert "0c54a47b_t010" in ids


def test_raw_box_geometry_returns_sane_percentiles():
    result = raw_box_geometry(DATA_DIR, CLIP_IDS)
    assert result["n"] > 0
    sides = result["side_px"]
    assert sides[1] <= sides[50] <= sides[99]  # percentiles are monotonic
    assert sides[1] > 0  # no zero/negative box sizes
    ratios = result["aspect_ratio"]
    assert ratios[1] <= ratios[50] <= ratios[99]


def test_dropout_length_distribution_returns_sane_percentiles():
    result = dropout_length_distribution(DATA_DIR, CLIP_IDS)
    assert result["n"] > 0
    gaps = result["gap_length_frames"]
    assert gaps[1] >= 1  # a "gap" here is defined as >=1 missing frame
    assert gaps[1] <= gaps[50] <= gaps[99]


def test_rejection_reason_frequency_accounts_for_every_detection():
    result = rejection_reason_frequency(DATA_DIR, CLIP_IDS)
    assert result["total"] > 0
    assert sum(result["counts"].values()) == result["total"]
    assert all(n >= 0 for n in result["counts"].values())


def test_rejection_reason_frequency_total_matches_raw_plus_fabricated():
    """Independent cross-check against the exact accounting bug found while
    building this function (see its docstring): total counted must equal
    the true raw detection count PLUS however many brand-new detections
    stage 4 fabricated. Checking against the raw count alone isn't enough
    -- a version that silently dropped fabricated `interpolated` detections
    from every bucket would still match the raw total by coincidence, since
    it wouldn't count them anywhere at all. Deliberately reruns the
    pipeline by hand here rather than calling the function under test, so
    this is a genuinely independent check, not a restatement of it.
    """
    from adapter.association import track_detections
    from adapter.geometric import apply_stage1
    from adapter.hand_config import hand_config
    from adapter.ingest import load_clip
    from adapter.interpolation import apply_stage4
    from adapter.temporal import apply_stage3

    config = hand_config()
    true_raw_total = 0
    true_fabricated_total = 0
    for cid in CLIP_IDS:
        clip = load_clip(os.path.join(DATA_DIR, cid))
        true_raw_total += sum(len(dets) for dets in clip.detections)

        stage1_frames = [apply_stage1(list(d), config) for d in clip.detections]
        tracks = track_detections(stage1_frames, config)
        before_stage4 = sum(len(t.detections) for t in tracks)

        apply_stage3(tracks, clip.pose, config)
        apply_stage4(tracks, config, frame_count=len(clip.detections))
        after_stage4 = sum(len(t.detections) for t in tracks)

        true_fabricated_total += after_stage4 - before_stage4

    result = rejection_reason_frequency(DATA_DIR, CLIP_IDS, config)

    assert result["total"] == true_raw_total + true_fabricated_total


def test_interpolated_proportion_report_covers_every_requested_clip():
    result = interpolated_proportion_report(DATA_DIR, CLIP_IDS)
    assert set(result["per_clip"]) == set(CLIP_IDS)
    assert all(0.0 <= v <= 1.0 for v in result["per_clip"].values())
    assert 0.0 <= result["mean"] <= 1.0
    assert result["clip_with_max"] in CLIP_IDS
