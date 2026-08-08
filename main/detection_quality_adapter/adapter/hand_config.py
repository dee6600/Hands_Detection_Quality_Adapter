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
    )
    defaults.update(overrides)
    return Config(**defaults)
