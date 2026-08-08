"""Milestone 6, Part 2: hand-specific parameters.

Four behavioral changes over the generic Part 1 defaults (spec S5), all
config -- no new logic needed in stages 1-4, just different numbers plus the
per-border/dedup hooks Milestones 2 and 5 added specifically to support this:

  1. At most two instances per frame, requested pool of 3-4 -- `selection.py`
     enforces the cap after association using track quality, not per-frame
     confidence.
  2. Roughly-equant shape rule, tighter than the generic default.
  3. Head-mounted camera: the static rule is unusually strong here already
     (see Milestone 4's notes) -- no extra config needed, just noting it.
  4. A second eye is available: per-box stereo depth rejects a bystander's
     hand beyond arm's reach -- see `stereo_depth.py` (migrated from the
     `exp/scafholds/` prototype).

Plus one the spec's edge-case table calls out that isn't one of the four
headline properties: hands crossing/overlapping must not be merged as a
duplicate, which needs the containment-ratio dedup gate tightened for hands
specifically (see `Config.duplicate_containment_threshold`'s docstring in
types.py for the real-data numbers behind the 0.7 chosen here).

Handedness independence (tracker never uses the left/right label) needed no
config at all -- it was already true of `association.py` and tested ahead of
schedule in Milestone 3.

Milestone 7 (`calibration/sweep_thresholds.py`) added two more overrides,
`plausible_size` and `max_dropout_frames`, both re-derived from real
distributional data across all 39 clips rather than the Milestone 1
placeholders they replace -- see each field's inline comment below for the
numbers. Neither needed labels: they're unsupervised checks against what the
detector actually emits and how tracks actually behave, not precision/recall
against ground truth (which this dataset doesn't have -- see
`calibration/metrics.py`'s module docstring).
"""

from __future__ import annotations

from adapter.types import Config


def hand_config(**overrides) -> Config:
    """A `Config` tuned for hands. Pass keyword overrides to adjust further
    (e.g. for a specific job/task type) without editing this function.
    """
    defaults = dict(
        class_max_instances=2,
        candidate_pool_size=4,
        # Milestone 7 sweep (calibration/sweep_thresholds.py) against all
        # 153,876 raw detections in the real dataset: side length p1=94px,
        # max=1110px. The generic (20, 800) lower bound is a complete no-op
        # here (0% of raw detections come anywhere near 20px) and the upper
        # bound incorrectly rejects ~0.16% of raw detections (253 boxes)
        # that are far more likely genuine close-up hands than noise, given
        # this dataset's own documented near-frame-filling characteristic
        # and the stereo-depth calibration's independent finding of hands
        # as close as 0.33m. (50, 1150) keeps a wide safety margin below
        # the smallest real box and just above the largest.
        plausible_size=(50.0, 1150.0),
        # roughly equant: real hand box aspect ratios across all 39 clips'
        # stage-1 survivors have p1=0.54, median=0.92, p99=2.10 -- (0.5, 2.0)
        # covers essentially all real hands while excluding markedly
        # elongated boxes the generic (0.3, 3.0) range would still admit.
        plausible_shape=(0.5, 2.0),
        # hands crossing/overlapping must survive as two detections, not
        # merge into one -- see geometric.py's `_containment` and
        # Config.duplicate_containment_threshold's docstring.
        duplicate_containment_threshold=0.7,
        # bottom-border exits are occlusion-driven (the wearer's own torso),
        # not a clean walk-out-of-frame: real ambiguous-dropout tracks in
        # t010/t036/ae580129_t057 often end with their box's bottom edge
        # already at or within ~150px of the frame's bottom edge but WITHOUT
        # outward (downward) velocity at the last frame -- occlusion doesn't
        # require continuing to move down, it just requires being there.
        exit_border_margin_overrides={"bottom": 150.0},
        exit_requires_outward_motion={"bottom": False},
        # beyond arm's reach = a bystander's hand, not the wearer's -- see
        # stereo_depth.py's module docstring for the calibration and why
        # 1.8m, not the more intuitive ~0.8m.
        max_reach_m=1.8,
        # Milestone 7 sweep: real internal-gap lengths within tracks (using
        # a deliberately loosened cap so the CURRENT threshold doesn't
        # pre-clip what's being measured) have p50=3, p75=13, p90=54,
        # p99=470 frames. The generic default of 10 sits just under p75 --
        # over a quarter of genuine short recoverable gaps were already too
        # long to interpolate. The long tail (p90 and up, worst case 2529
        # frames = 84s) is not trustworthy as "the same object": at that
        # timescale the tracker's speed-based gate has grown so wide
        # (radius = track_gate_speed_px_per_frame * dt) it stops meaningfully
        # constraining anything, so those almost certainly reflect
        # coincidental position matches between two different objects, not
        # real dropouts. 15 frames (0.5s) comfortably covers the reliable
        # p75 mass without reaching into that contaminated tail.
        max_dropout_frames=15,
    )
    defaults.update(overrides)
    return Config(**defaults)
