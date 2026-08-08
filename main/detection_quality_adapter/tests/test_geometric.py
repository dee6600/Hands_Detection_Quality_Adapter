"""Milestone 2: hand-built synthetic fixtures for stage 1 (geometric rejection).

One frame per rule, in the fixed order the spec requires: duplicates ->
size -> shape, plus a clean frame that survives untouched and one exercising
the full `apply_stage1` order together.
"""

from adapter.geometric import (
    apply_stage1,
    reject_duplicates,
    reject_implausible_shape,
    reject_implausible_size,
)
from adapter.types import Config, Detection, Tag


def _det(frame, xyxy, confidence, class_label=0):
    return Detection(frame=frame, xyxy=xyxy, confidence=confidence, class_label=class_label)


def test_reject_duplicates_merges_heavily_overlapping_pair():
    config = Config()
    strong = _det(0, (100.0, 100.0, 200.0, 200.0), confidence=0.9)
    weak = _det(0, (105.0, 102.0, 203.0, 198.0), confidence=0.4)  # near-identical box

    survivors = reject_duplicates([weak, strong], config)

    assert len(survivors) == 1
    assert survivors[0] is strong
    assert survivors[0].tag == Tag.MERGED


def test_reject_duplicates_leaves_non_overlapping_boxes_untouched():
    config = Config()
    a = _det(0, (0.0, 0.0, 50.0, 50.0), confidence=0.9)
    b = _det(0, (500.0, 500.0, 550.0, 550.0), confidence=0.8)

    survivors = reject_duplicates([a, b], config)

    assert survivors == [a, b]
    assert all(d.tag == Tag.REPORTED for d in survivors)


def test_reject_implausible_size_drops_too_small_and_too_large():
    config = Config(plausible_size=(20.0, 800.0))
    too_small = _det(0, (0.0, 0.0, 5.0, 5.0), confidence=0.9)
    too_large = _det(0, (0.0, 0.0, 900.0, 900.0), confidence=0.9)
    plausible = _det(0, (0.0, 0.0, 100.0, 120.0), confidence=0.9)

    survivors = reject_implausible_size([too_small, too_large, plausible], config)

    assert survivors == [plausible]
    assert too_small.tag == Tag.REJECTED
    assert too_large.tag == Tag.REJECTED
    assert plausible.tag == Tag.REPORTED


def test_reject_implausible_shape_drops_elongated_boxes():
    config = Config(plausible_shape=(0.3, 3.0))
    too_wide = _det(0, (0.0, 0.0, 400.0, 40.0), confidence=0.9)  # ratio 10.0
    too_tall = _det(0, (0.0, 0.0, 40.0, 400.0), confidence=0.9)  # ratio 0.1
    plausible = _det(0, (0.0, 0.0, 100.0, 120.0), confidence=0.9)  # ratio ~0.83

    survivors = reject_implausible_shape([too_wide, too_tall, plausible], config)

    assert survivors == [plausible]
    assert too_wide.tag == Tag.REJECTED
    assert too_tall.tag == Tag.REJECTED
    assert plausible.tag == Tag.REPORTED


def test_clean_frame_survives_all_three_rules_untouched():
    config = Config()
    a = _det(0, (100.0, 100.0, 200.0, 220.0), confidence=0.9)
    b = _det(0, (600.0, 300.0, 690.0, 400.0), confidence=0.8)

    survivors = apply_stage1([a, b], config)

    assert survivors == [a, b]
    assert all(d.tag == Tag.REPORTED for d in survivors)


def test_apply_stage1_runs_rules_in_fixed_order():
    """A frame combining all three failure modes plus one clean box: dedup
    happens first (so the duplicate pair collapses to one before size/shape
    even see it), then size, then shape.
    """
    config = Config(plausible_size=(20.0, 800.0), plausible_shape=(0.3, 3.0))

    dup_strong = _det(0, (100.0, 100.0, 200.0, 200.0), confidence=0.9)
    dup_weak = _det(0, (103.0, 101.0, 198.0, 199.0), confidence=0.5)
    too_small = _det(0, (0.0, 0.0, 5.0, 5.0), confidence=0.7)
    too_elongated = _det(0, (0.0, 0.0, 400.0, 40.0), confidence=0.7)
    clean = _det(0, (600.0, 300.0, 690.0, 400.0), confidence=0.6)

    survivors = apply_stage1([dup_strong, dup_weak, too_small, too_elongated, clean], config)

    assert survivors == [dup_strong, clean]
    assert dup_strong.tag == Tag.MERGED
    assert clean.tag == Tag.REPORTED
    assert too_small.tag == Tag.REJECTED
    assert too_elongated.tag == Tag.REJECTED
