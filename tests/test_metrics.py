"""Milestone 7: hand-built synthetic ground truth for calibration/metrics.py.

There's no real labelled reference set for this dataset (see metrics.py's
module docstring) -- these tests verify the matching/scoring MACHINERY is
correct, using invented ground truth, so it's ready the moment real labels
exist. They are not, and cannot be, a claim about this project's real
precision/recall.
"""

from calibration.metrics import (
    evaluate_against_ground_truth,
    flag_wrong_direction_stages,
    interpolated_proportion,
    per_stage_report,
)
from adapter.types import Detection, Tag, Track


def _det(frame, xyxy, tag=Tag.REPORTED, confidence=0.8):
    return Detection(frame=frame, xyxy=xyxy, confidence=confidence, tag=tag)


def test_perfect_match_gives_precision_and_recall_of_one():
    preds = {0: [_det(0, (10.0, 10.0, 50.0, 50.0))]}
    truth = {0: [(10.0, 10.0, 50.0, 50.0)]}

    m = evaluate_against_ground_truth(preds, truth, stage="test")

    assert m.true_positives == 1
    assert m.precision == 1.0
    assert m.recall == 1.0


def test_extra_prediction_with_no_match_is_a_false_positive():
    preds = {0: [_det(0, (10.0, 10.0, 50.0, 50.0)), _det(0, (900.0, 900.0, 940.0, 940.0))]}
    truth = {0: [(10.0, 10.0, 50.0, 50.0)]}

    m = evaluate_against_ground_truth(preds, truth, stage="test")

    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.precision == 0.5
    assert m.recall == 1.0


def test_missing_ground_truth_box_is_a_false_negative():
    preds = {0: [_det(0, (10.0, 10.0, 50.0, 50.0))]}
    truth = {0: [(10.0, 10.0, 50.0, 50.0), (900.0, 900.0, 940.0, 940.0)]}

    m = evaluate_against_ground_truth(preds, truth, stage="test")

    assert m.true_positives == 1
    assert m.false_negatives == 1
    assert m.precision == 1.0
    assert m.recall == 0.5


def test_rejected_detections_are_excluded_from_scoring():
    """A detection tagged `rejected` isn't a live prediction any more -- it
    should count as neither a true positive nor a false positive, and the
    ground truth box it would have matched becomes a false negative instead.
    """
    preds = {0: [_det(0, (10.0, 10.0, 50.0, 50.0), tag=Tag.REJECTED)]}
    truth = {0: [(10.0, 10.0, 50.0, 50.0)]}

    m = evaluate_against_ground_truth(preds, truth, stage="test")

    assert m.true_positives == 0
    assert m.false_positives == 0
    assert m.false_negatives == 1


def test_greedy_matching_resolves_an_ambiguous_pair_by_highest_iou_first():
    """Two overlapping ground-truth boxes (like two crossing hands) and two
    predictions, where one prediction (`mid`) is plausibly close to BOTH
    truths (IoU 0.667 vs truth_a, 0.702 vs truth_b -- both above threshold)
    while the other (`clear`) is an almost-exact match to truth_a alone
    (IoU 0.951) but not a candidate for truth_b at all (IoU 0.481, below
    threshold). Matching globally by highest IoU first resolves this
    correctly (clear->truth_a, mid->truth_b, 2 true positives); matching
    predictions in naive list order without considering `clear`'s much
    better claim on truth_a could instead let `mid` grab truth_a first and
    leave `clear` either double-counted or stranded.
    """
    truth_a = (0.0, 0.0, 40.0, 40.0)
    truth_b = (15.0, 0.0, 55.0, 40.0)
    pred_mid = _det(0, (8.0, 0.0, 48.0, 40.0))
    pred_clear = _det(0, (1.0, 0.0, 41.0, 40.0))

    m = evaluate_against_ground_truth({0: [pred_mid, pred_clear]}, {0: [truth_a, truth_b]}, stage="test")

    assert m.true_positives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 0


def test_per_stage_report_preserves_order():
    truth = {0: [(0.0, 0.0, 40.0, 40.0)]}
    stage_detections = {
        "raw": {0: [_det(0, (0.0, 0.0, 40.0, 40.0))]},
        "final": {0: [_det(0, (0.0, 0.0, 40.0, 40.0))]},
    }

    reports = per_stage_report(stage_detections, truth)

    assert [r.stage for r in reports] == ["raw", "final"]


def test_flag_wrong_direction_stages_catches_a_precision_regression():
    truth = {0: [(0.0, 0.0, 40.0, 40.0)]}
    true_det = _det(0, (0.0, 0.0, 40.0, 40.0))
    false_det = _det(0, (900.0, 900.0, 940.0, 940.0))
    # "raw": 1 TP + 1 FP -> precision 0.5. A correctly-behaving rejection
    # stage would drop the false positive (precision rises to 1.0). This one
    # is misconfigured -- it keeps the false positive and wrongly rejects
    # the true one instead, so precision actually DROPS to 0.0.
    stage_detections = {
        "raw": {0: [true_det, false_det]},
        "rejection": {0: [_det(0, true_det.xyxy, tag=Tag.REJECTED), false_det]},
    }

    reports = per_stage_report(stage_detections, truth)
    warnings = flag_wrong_direction_stages(reports, expected={"rejection": "precision"})

    assert len(warnings) == 1
    assert "rejection" in warnings[0]


def test_flag_wrong_direction_stages_silent_when_behaving():
    truth = {0: [(0.0, 0.0, 40.0, 40.0)]}
    stage_detections = {
        "raw": {0: [_det(0, (0.0, 0.0, 40.0, 40.0)), _det(0, (900.0, 900.0, 940.0, 940.0))]},
        "rejection": {0: [_det(0, (0.0, 0.0, 40.0, 40.0))]},  # false positive correctly dropped
    }

    reports = per_stage_report(stage_detections, truth)
    warnings = flag_wrong_direction_stages(reports, expected={"rejection": "precision"})

    assert warnings == []


def test_interpolated_proportion_basic():
    t1 = Track(track_id=0, detections=[_det(0, (0, 0, 1, 1)), _det(1, (0, 0, 1, 1), tag=Tag.INTERPOLATED)])
    t2 = Track(track_id=1, detections=[_det(0, (0, 0, 1, 1))])

    assert interpolated_proportion([t1, t2]) == 1 / 3


def test_interpolated_proportion_empty_is_zero():
    assert interpolated_proportion([]) == 0.0
    assert interpolated_proportion([Track(track_id=0)]) == 0.0
