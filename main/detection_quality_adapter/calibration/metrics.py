"""Milestone 7: per-stage precision/recall, and the one standing check spec
S8 says needs no labelled data.

**This module cannot be run against real ground truth for this project yet
-- there isn't any.** Spec S7 is explicit: "No value here should be set by
inspection. A labelled reference set is required." This dataset's
`hand_boxes.json` is the raw, noisy detector output this whole adapter
exists to correct, not ground truth -- there's no annotation file anywhere
in `data/` recording which boxes are actually correct. The functions below
are the ready-to-run machinery: verified correct against synthetic,
hand-built ground truth in `tests/test_metrics.py`, so they're not a
prototype needing rework once a real reference set exists -- but no number
this module could report about this project's real clips today would be a
real precision/recall figure. See `sweep_thresholds.py` for what IS
honestly computable from this dataset without labels.

The one exception is `interpolated_proportion`: spec S8 names it explicitly
as needing no labelled data ("the clearest early signal the adapter has
started fabricating rather than recovering"), and it IS run for real, across
all 39 clips, in `sweep_thresholds.py`'s report.
"""

from __future__ import annotations

from dataclasses import dataclass

from adapter.types import Detection, Tag, Track

Box = tuple[float, float, float, float]


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class StageMetrics:
    """TP/FP/FN for one stage's surviving detections against ground truth,
    greedily IoU-matched per frame (closest pairs first, one match each).
    """

    stage: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0


def evaluate_against_ground_truth(
    detections_by_frame: dict[int, list[Detection]],
    ground_truth_by_frame: dict[int, list[Box]],
    stage: str,
    iou_threshold: float = 0.5,
) -> StageMetrics:
    """Needs a labelled reference set -- see this module's docstring.
    `detections_by_frame[frame]` should already be filtered to that stage's
    survivors; `rejected` detections are excluded here too, defensively.
    """
    metrics = StageMetrics(stage=stage)
    for frame in set(detections_by_frame) | set(ground_truth_by_frame):
        preds = [d for d in detections_by_frame.get(frame, []) if d.tag != Tag.REJECTED]
        truths = ground_truth_by_frame.get(frame, [])

        candidates = []
        for pi, pred in enumerate(preds):
            for ti, truth in enumerate(truths):
                iou = _iou(pred.xyxy, truth)
                if iou >= iou_threshold:
                    candidates.append((iou, pi, ti))
        candidates.sort(key=lambda c: -c[0])

        matched_pred = [False] * len(preds)
        matched_truth = [False] * len(truths)
        for _, pi, ti in candidates:
            if matched_pred[pi] or matched_truth[ti]:
                continue
            matched_pred[pi] = True
            matched_truth[ti] = True
            metrics.true_positives += 1

        metrics.false_positives += matched_pred.count(False)
        metrics.false_negatives += matched_truth.count(False)

    return metrics


def per_stage_report(
    stage_detections: dict[str, dict[int, list[Detection]]],
    ground_truth_by_frame: dict[int, list[Box]],
    iou_threshold: float = 0.5,
) -> list[StageMetrics]:
    """One `StageMetrics` per named stage, in the order `stage_detections`
    was given (Python dicts preserve insertion order) -- pass stages in
    pipeline order so `flag_wrong_direction_stages` compares each stage
    against the one immediately before it.
    """
    return [
        evaluate_against_ground_truth(dets, ground_truth_by_frame, stage, iou_threshold)
        for stage, dets in stage_detections.items()
    ]


def flag_wrong_direction_stages(reports: list[StageMetrics], expected: dict[str, str]) -> list[str]:
    """Per spec S8: false-positive rejection should raise precision,
    trajectory recovery should raise recall, duplicate merging should raise
    both. `expected` maps a stage name to which of "precision"/"recall"/
    "both" it should raise relative to the PREVIOUS stage in `reports`.
    Returns one human-readable warning per violation, empty if all stages
    that declared an expectation moved the right way (or weren't checked).
    """
    warnings = []
    for prev, curr in zip(reports, reports[1:]):
        want = expected.get(curr.stage)
        if want is None:
            continue
        if want in ("precision", "both") and curr.precision < prev.precision:
            warnings.append(
                f"{curr.stage}: precision dropped ({prev.precision:.3f} -> {curr.precision:.3f}), expected it to rise"
            )
        if want in ("recall", "both") and curr.recall < prev.recall:
            warnings.append(
                f"{curr.stage}: recall dropped ({prev.recall:.3f} -> {curr.recall:.3f}), expected it to rise"
            )
    return warnings


def interpolated_proportion(tracks: list[Track]) -> float:
    """Spec S8's labels-free standing check. No ground truth needed: a
    rising proportion of `interpolated` detections, watched over time (or
    compared across clips), is the earliest signal the adapter has started
    fabricating trajectories rather than recovering real ones. Returns 0.0
    if there are no detections at all (nothing to divide by).
    """
    all_dets = [d for t in tracks for d in t.detections]
    if not all_dets:
        return 0.0
    return sum(1 for d in all_dets if d.tag == Tag.INTERPOLATED) / len(all_dets)
