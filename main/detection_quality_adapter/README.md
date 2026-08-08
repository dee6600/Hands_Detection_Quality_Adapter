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

### Speed/gate calibration: the tracker gate and the displacement check needed to split

Both the tracker's match gate (stage 2) and the displacement-plausibility
rule (stage 3) originally shared one `Config.max_speed_px_per_frame` value
(150 px/frame). Checking real per-clip speed distributions across all 39
clips (using the tracker's own matched consecutive-detection pairs) surfaced
two things:

1. **Across 38 of the 39 clips, adjacent-frame hand speed is tightly and
   consistently distributed**: p99 = 82.6 px/frame, p99.5 = 97.7,
   p99.9 = 132.0. The old 150 default sat essentially at p99.9 — so loose
   that stage 3's displacement rule almost never fired (0 rejections on
   `t010`, ~1-2 on other clips), even though inspection found real
   "wobble" moments (motion blur / confidence dips mid-fast-motion — the
   spec's own edge-case table calls this out) that the rule should have
   caught. Several of these wobble segments sit inside tracks hundreds to
   over a thousand detections long, confirming the *tracker* had the right
   identity the whole time — it was specifically the plausibility check
   that was too loose to flag the bad frame.
2. **One clip, `ae580129_t057` (a maintenance worker's "open_cover" task),
   is a different population, not an outlier to fit**: its VIO camera speed
   averages 0.375 m/s vs. 0.155 (`t010`) and 0.082 (`t036`) — 2-4x higher —
   and its hand-speed distribution is shifted the same way (median
   28.5 px/frame vs. ~9 elsewhere; p99 = 202.9 vs. 82.6). This isn't
   noisier tracking, it's a faster-moving task inflating apparent
   image-space speed via camera ego-motion. A single flat px/frame
   threshold can't tell "fast hand" from "normal hand, fast camera" apart —
   see the open question in `planning.md` about scaling this threshold by
   the clip's own VIO speed instead.

**Fix applied now** (informal calibration from unsupervised distributional
data, same pattern as the static-rule fix — no labels needed for this part):
split the one shared value into two.
`Config.max_speed_px_per_frame` (now 110, just above the 38-clip p99.5) is
used only by stage 3's plausibility check. `Config.track_gate_speed_px_per_frame`
(350, comfortably above the single fastest jump ever observed in any clip,
305.8 px/frame) is used only by the stage-2 tracker's match gate, and is
deliberately generous — its only job is to not sever a genuine track at the
exact moment it's moving fastest; flagging that moment as untrustworthy is
stage 3's job, not stage 2's. Effect on `t010`: track count dropped from 65
to 44 (the old shared, tighter gate was fragmenting real tracks), while
stage 3's rejection rate rose modestly from 0.6% to 1.5% (catching more real
wobble moments without breaking track continuity to do it).

**Deferred to Milestone 7**: making `max_speed_px_per_frame` adaptive to the
clip's own camera speed at each frame (via `ClipData.pose`, the same signal
already driving the static rule), so `ae580129_t057`-like clips don't need a
systematically higher false-rejection rate than calmer clips. This needs
labeled data to validate the scaling factor rather than more eyeballing.
