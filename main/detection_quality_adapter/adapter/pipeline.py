"""Orchestrates the adapter stages in fixed order.

Milestone 0: no-op pass-through only. Each stage is wired in as it's built
(Milestones 2-6), in this fixed order: geometric -> association -> temporal
-> interpolation -> selection.
"""


def run_pipeline(detections):
    """Pass detections through untouched, tagged 'reported'.

    `detections` is a per-frame list of raw detection dicts/objects. Milestone 1
    replaces the tagging below with the real `Detection` dataclass.
    """
    return [
        {**d, "tag": "reported"} if isinstance(d, dict) else d
        for d in detections
    ]
