import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'adapter'))

from exp.scafholds.nominal_calibration import (
    ZED_X_2_2MM, ZED_X_4MM, ZED_X_MINI_2_2MM, depth_from_disparity,
)


def test_focal_length_matches_datasheet_ballpark():
    # 2.2mm lens, 3um pixels -> ~733px. Cross-checked against datasheet's
    # 110deg max horizontal FOV independently in the module's derivation notes.
    assert 700 < ZED_X_2_2MM.focal_px < 770


def test_4mm_lens_focal_is_larger_than_2_2mm():
    assert ZED_X_4MM.focal_px > ZED_X_2_2MM.focal_px


def test_mini_baseline_smaller_than_full_zedx():
    assert ZED_X_MINI_2_2MM.baseline_m < ZED_X_2_2MM.baseline_m


def test_depth_from_disparity_known_value():
    # Z = f*B / d.  f=733.3px, B=0.12m, d=267px -> ~0.329m (matches the
    # real-data validation in the module docstring).
    depth = depth_from_disparity(267.0, ZED_X_2_2MM)
    assert 0.30 < depth < 0.36


def test_depth_decreases_as_disparity_increases():
    near = depth_from_disparity(300.0, ZED_X_2_2MM)
    far = depth_from_disparity(30.0, ZED_X_2_2MM)
    assert near < far


def test_zero_or_negative_disparity_is_invalid():
    assert depth_from_disparity(0.0, ZED_X_2_2MM) is None
    assert depth_from_disparity(-5.0, ZED_X_2_2MM) is None
