# Detection Quality Adapter

Post-processing stage between a hand detector and downstream consumers.
Corrects false positives/negatives using temporal (multi-frame) logic,
without touching or retraining the detector.

See `../../planning.md` for the milestone plan and `../../hand_detection_spec.pdf`
for the source spec. Dataset lives in `../../data/` (see its README for schema).

## Setup

```
conda activate koshalabs
pip install -r requirements.txt
pytest
```

## Visualizing stage 1 (geometric rejection)

`tests/test_geometric.py` covers stage 1 on synthetic fixtures;
`tests/test_geometric_real_data.py` checks it against real clips. To see it
by eye, render an annotated video for a real clip bundle:

```
python scripts/visualize_stage1.py ../../data/0c54a47b_t010 /tmp/stage1_t010.mp4
```

Color legend: green = kept untouched, blue = kept after absorbing a
duplicate, orange dashed = dropped as a duplicate, red dashed = dropped for
implausible size/shape. Add `--max-frames N` for a quick preview.

## Visualizing stage 2 (the tracker)

`tests/test_association.py` covers the tracker on synthetic sequences;
`tests/test_association_real_data.py` checks it against real clips (run
through stage 1 first, matching the real pipeline order). To see it by eye,
render an annotated video for a real clip bundle:

```
python scripts/visualize_tracks.py ../../data/0c54a47b_t010 /tmp/tracks_t010.mp4
```

Each track gets its own persistent color (stable across the whole clip) plus
a short fading trail of recent centers, so identity continuity across a gap,
or an identity swap on a crossing, is visible directly. The console output
also prints every track's id/first-frame/last-frame/detection-count, useful
for spotting excessive fragmentation. Add `--max-frames N` to cap the
rendered video length (the tracker itself always runs on the full clip,
since this is an offline per-clip problem).

## Visualizing stage 3 (temporal rejection)

`tests/test_temporal.py` covers displacement/unsupported/static rejection on
synthetic tracks; `tests/test_temporal_real_data.py` checks it against real
clips using the clip's own VIO pose as the camera-motion signal. To see it
by eye, render an annotated video for a real clip bundle:

```
python scripts/visualize_temporal.py ../../data/0c54a47b_t010 /tmp/temporal_t010.mp4
```

Color legend: green = kept, red dashed = rejected for implausible
displacement, purple dashed = rejected as an unsupported/flicker track,
yellow dashed = rejected as static background. Add `--max-frames N` for a
quick preview.

### The static rule needed a sustained-run requirement, not just a threshold

The first version of `reject_static` flagged any single frame where a box
moved less than `static_px_threshold` while the camera was "moving" (a very
low bar: 0.05 m/s or 1°/frame, true almost continuously for a head-mounted
camera on someone working). Eyeballing real clips (`t010`, `407258cd_t036`)
showed this over-firing badly: on `t010` it rejected 261/5120 survivors
(5.1%), and inspection showed real hands getting flagged just for pausing
for a frame or two (e.g. frame 16 — a hand actively holding a device).

The deeper issue: a head-mounted camera's apparent motion is often
rotation-dominated (the wearer turning their head), and whatever the wearer
is looking at — for hand-eye tasks, usually their own hands — sits near the
center of gaze and shows the *least* apparent motion of anything in frame.
So a naive per-frame displacement threshold is almost backwards: it's most
likely to flag the hand of interest and least likely to catch peripheral
background clutter.

Fix applied: `reject_static` now requires `min_static_run_frames` (default
15) *consecutive* still-while-moving frames before rejecting anything, and
rejects the whole confirmed run including its first frame, not just the
tail. This is a real logic change, not a threshold tweak — a brief real
pause is, by construction, too short to trip it. Effect on `t010`: static
rejections dropped from 261 to ~0 (stage 3's total rejection rate dropped
from 5.7% to 0.6%, now almost entirely the flicker rule).

**This does not fully solve the problem.** On `407258cd_t036` (a
woodcarving clip), stage 3 still rejects 9.2% of survivors, nearly all
static — frame 220 shows the left hand bracing a workpiece while the right
hand carves, correctly identified as a real, deliberately-still hand, but
flagged anyway because it stays still for *seconds*, not frames. No
sustained-run window is long enough to admit a genuine multi-second brace
while still catching real background. Solving this properly means
distinguishing rotation-induced apparent motion from translation-induced
parallax and judging staticness against the *expected* motion at a box's
position, rather than a flat pixel threshold — deferred to Milestone 7,
where labeled data can validate that approach instead of more eyeballing.

Color legend: green = kept, red dashed = rejected for implausible
displacement, purple dashed = rejected as an unsupported/flicker track,
yellow dashed = rejected as static background. Add `--max-frames N` for a
quick preview.
