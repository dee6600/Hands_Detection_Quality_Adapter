# Evaluation Log

Label-free evaluation work for the detection quality adapter paper. No ground
truth exists for this dataset (see `calibration/metrics.py`'s docstring) — every
result below is a proxy, not precision/recall. Say so in every section.

**Rules for contributors (including future Claude Code sessions):**
- Append to your own section only. Don't edit another task's section.
- Every result must state what it does and does NOT prove.
- `Detection.tag` is mutated in place by every pipeline stage — call `load_clip()`
  fresh for every configuration compared. Never reuse `Detection` objects across
  runs. Getting this wrong has silently corrupted results in this codebase before.
- Cite exact commands to reproduce every number you report.
- Link the script(s) and any committed results file(s) you produced.

---

## 1. Dropout Recovery

**Status:** done
**Script(s):** `calibration/dropout_recovery.py` (tests: `tests/test_dropout_recovery.py`)
**Results file(s):** `calibration/dropout_recovery_results.json`

**Reproduce:** `conda activate koshalabs && python calibration/dropout_recovery.py --data-dir data`

**Method:** Measures how well stages 1-4 (`run_pipeline`, hand-tuned config,
stage 5/selection deliberately excluded so it can't confound a stage-4-specific
number) reconstruct their OWN already-filtered detections after artificially
deleting stretches of them — there is no ground truth for this dataset (see
`calibration/metrics.py`'s docstring), so the detector's own post-processed
output stands in as the pseudo-reference, per this task's brief. For every one
of the 39 clips: (1) run the reference pipeline once on the untouched clip;
(2) within each resulting track, find maximal runs of consecutive frames
carrying an untouched, non-duplicate raw detection (`tag == REPORTED` only —
deliberately not `MERGED`, since a merged frame still has a suppressed
duplicate raw box sitting in the raw input that could resurface and survive
stage 1 in the removed box's place, silently defeating the mask); (3) for gap
lengths {1, 2, 3, 5, 10, 15} frames, take one centered interior block per
qualifying run (needs ≥3 real frames of context on both sides) and mark it for
deletion; (4) for each gap length, `load_clip()` completely fresh, delete
exactly those raw boxes, and re-run the full stages-1-4 pipeline on the masked
clip; (5) re-identify each placement's track via its still-untouched context
frame (exact `xyxy` match) and check, per masked frame, whether an
`interpolated` detection appeared (recovery), how close it is to the real
removed box (IoU, center distance), and whether any *unmasked* neighboring
frame got an unintended `interpolated` fill (false-fill). Stratified by gap
length and by `meta.json`'s `job` field. `load_clip()` is called fresh for the
reference run and separately for every masking configuration, per this
codebase's standing `Detection.tag`-mutation gotcha.

**Key results (pooled across all 39 clips, from `dropout_recovery_results.json`):**

| gap (frames) | n placements | recovery rate | median IoU | median center dist (px) | false-fill rate |
|---|---|---|---|---|---|
| 1  | 1,815 | 76.1% | 0.945 | 3.3   | 0.09% |
| 2  | 1,715 | 76.0% | 0.928 | 4.7   | 0.20% |
| 3  | 1,624 | 76.6% | 0.905 | 6.5   | 0.19% |
| 5  | 1,461 | 78.0% | 0.865 | 10.1  | 0.56% |
| 10 | 1,208 | 76.8% | 0.761 | 20.2  | 1.17% |
| 15 | 1,061 | **9.5%** | 0.079 | 176.0 | 0.72% |

Recovery rate holds essentially flat (76-78%) from 1 to 10 masked frames, while
localization quality degrades smoothly and predictably as the gap widens
(median IoU 0.945 → 0.761, median center error 3.3px → 20.2px) — exactly the
shape you'd expect from linear position/size interpolation over an
increasingly uncertain interval. Gap length 15 then collapses to 9.5%
recovery and near-zero IoU. This is a **real, reproducible interaction between
two stages, not noise or a bug in this evaluation script**:
`hand_config().max_dropout_frames` (15) is read both by the tracker's own
patience window in `association.py` (a track expires once
`frame_idx - last_frame > max_dropout_frames`) and by stage 4's own gap-length
cutoff in `interpolation.py` (`dt > max_dropout_frames` skips the fill). A
15-frame masked gap means `dt = 16` between the two real anchors — `16 > 15`
— so the tracker itself expires and starts a brand-new track before stage 4
is ever consulted; interpolation never even gets a chance to run on that gap,
regardless of how good its own logic is. `n_placements_track_lost` was 0 at
every gap length (the anchor-based track re-identification itself always
worked), confirming the gap=15 collapse is this off-by-one-flavored config
interaction, not a re-identification failure in this script. False-fill rate
stays low throughout (well under 1.2% even at its highest) and rises only
mildly with gap length.

**What this proves / does not prove:** Proves, without needing labels, that
stage 4's linear-interpolation fill mechanism itself recovers a large,
consistent majority (~76-78%) of short-to-medium (1-10 frame) dropouts from
the pipeline's own filtered detections, with position/size accuracy that
degrades gracefully (not catastrophically) as the gap widens, and does so
with a low rate of fabricating fills where nothing was actually missing. Also
proves a genuine, previously-undocumented limitation: sharing
`max_dropout_frames` between the tracker's patience window and stage 4's own
cutoff creates a hard collapse in recoverability right at that threshold,
which is a real, fixable-in-`adapter/`-later finding (out of scope for this
evaluation-only module) rather than an inherent limit of interpolation itself.
Does **NOT** prove anything about accuracy against real hand positions — the
"real" box a masked frame is scored against is itself just `hand_boxes.json`'s
raw detector output filtered by stages 1-3, not ground truth; a detector that
is systematically wrong in the same way at a masked frame and at its own
interpolation of that frame would still score perfectly here. Does not
prove the pipeline recovers LONG dropouts well or poorly on their own
merits — gap=15 specifically triggers the tracker/stage-4 threshold
interaction above, so this measurement can't isolate "is 15-frame
interpolation inherently harder" from "does the tracker even hand stage 4 the
opportunity."

**Caveats:** No ground truth exists for this dataset — every number above is
a self-consistency / reconstruction-of-detector-output proxy, not a
precision/recall or accuracy claim (see `calibration/metrics.py`'s docstring
and the module docstring in `dropout_recovery.py`). One placement is taken
per qualifying run per gap length (a centered block), not a dense sliding-window
sample of every possible position within a run — this keeps placements
independent and easy to reason about but under-uses very long real runs,
which could support several non-overlapping placements each; a denser sampler
is a reasonable future extension. `MERGED`-tagged frames are excluded from
maskable runs entirely (see Method), so this doesn't test recovery behavior
around dedup'd frames specifically. Masking one object's box can, rarely,
perturb a *different* object's stage-1 duplicate-merge decision at the exact
same frame if the removed box was the winning side of that merge — not
corrected for, and folded into the (already low) false-fill numbers as
ordinary measurement noise rather than filtered out.

---

## 2. Track Quality

**Status:** done
**Script(s):** `calibration/track_quality.py` (tests: `tests/test_track_quality.py`)
**Results file(s):** `calibration/track_quality_results.json`

**Reproduce:** `conda activate koshalabs && python calibration/track_quality.py --data-dir data`

**Method:** For every one of the 39 real clips, load the clip bundle TWICE
(`load_clip()` fresh each time, since `Detection.tag` is mutated in place by
every stage) and run two configurations of the same pipeline: (a)
**association only** — stage 1 (geometric) + stage 2 (tracker), nothing
else; (b) **full hand pipeline** — stages 1-5 (+ temporal rejection,
interpolation, the 2-hand selection cap), stage 6 (stereo depth) left off.
Compute, per clip and pooled across all 39: track count; mean/median track
length (frame span, first-to-last-detection, gaps included) and
detections-per-track (real+interpolated boxes only, gaps excluded); a
fragmentation proxy — pairs where one track's first detection lands within
30 frames (1s) and within 2x the larger box's diagonal of another track's
last detection, i.e. a plausible severed continuation of the same object,
not a confirmed one; gaps per track and total gap frames; and
`interpolated_proportion` (reused as-is from `calibration/metrics.py`).
Also broken down by `meta.json`'s `job` field (every clip is a different
job — see `planning.md`'s Milestone 7 notes on why aggregate-only would
hide this).

**Key results (pooled across all 39 clips, from `track_quality_results.json`):**

| metric | association only | full pipeline |
|---|---|---|
| n_tracks | 787 | 787 |
| track_length_frames (mean of clip means) | 194.48 | 194.48 |
| fragmentation_candidates / rate (mean) | 254 / 0.285 | 254 / 0.285 |
| detections_per_track (mean of clip means) | 184.01 | 188.96 |
| gaps_per_track (mean of clip means) | 2.97 | 1.39 |
| total_gap_frames (summed) | 8,240 | 4,341 |
| interpolated_proportion (mean across clips) | 0.0 | 0.0348 |

The headline finding is that the first three rows are **identical, not just
similar**, between the two arms. This isn't measurement noise — it follows
from the pipeline's own "nothing is ever deleted" design: stages 3-5 only
tag existing detections or (interpolation) append new ones strictly between
a track's existing first and last frame; no stage after the stage-2 tracker
ever merges, splits, or changes a track's start/end. So track identity,
count, length span, and this fragmentation proxy are fixed entirely by the
tracker (stage 2) — the correction stages (3-5) cannot be shown by this
comparison to reduce or increase fragmentation relative to association
alone, because both arms share the exact same track boundaries by
construction. What the correction stages visibly change is track *content*:
~53% of internal gap-frames get filled (8,240 → 4,341), gaps-per-track
roughly halves, and a real, nonzero `interpolated_proportion` (mean 3.5%,
consistent with the existing Milestone 7 finding in `planning.md`) appears
where the association-only arm has a hard 0.0 by construction (stage 4
never runs there).

**What this proves / does not prove:** Proves, without needing labels, that
this pipeline's temporal/interpolation/selection stages operate purely as a
content-level correction layer on top of whatever tracks the association
stage already produced — they recover gap frames and flag quality, but
structurally cannot be "over-bridging" fragmented tracks into fewer, longer
ones, since track count and length are provably unchanged. Does NOT prove
the association stage's own fragmentation (254 candidates, ~28.5% of
tracks) is itself correct or incorrect — the fragmentation proxy is a
spatial/temporal heuristic (arbitrary but documented thresholds: 30 frames,
2x box diagonal), not a confirmed split, and two genuinely different real
objects passing through the same place in quick succession would trigger it
identically to a real severed track. Does NOT prove interpolated frames are
accurate recoveries rather than fabrications — `interpolated_proportion`
is a proxy for the RISK of fabrication (per spec S8), not a correctness
check, and per this project's standing framing rule, must always be read
alongside the other numbers rather than treated as a quality score on its
own.

**Caveats:** No ground truth exists for this dataset (see
`calibration/metrics.py`'s docstring) — nothing here is a precision/recall
claim. The fragmentation-proxy thresholds (30 frames, 2x box diagonal) are
unlabeled-data-informed defaults, not calibrated against any known-correct
split/non-split pairs; the module's own docstring documents the reasoning
but flags them as heuristic. `track_length_frames` and
`fragmentation_candidates` only look useful as a way to characterize the
tracker in isolation, not as a comparison axis between the two pipeline
arms, given the identical-by-construction result above — worth remembering
before re-running this kind of arm comparison on a different stage split.

---

## 3. Ablation & Sensitivity

**Status:** done
**Script(s):** `calibration/ablation.py` (tests: `tests/test_ablation.py`)
**Results file(s):** `calibration/ablation_results.json`

**Method:**

Full hand pipeline (`hand_config()`, stages 1-5; stage 6/stereo depth off
throughout, per the project's own finding that it's uncalibrated outside
one clip), run across all 39 real clips: `153,876` raw detections, `157,775`
after stage 4's gap-fill fabricates new ones for the full pipeline.

Every rule is disabled purely via a `hand_config(**overrides)` value chosen
to make it a structural no-op — nothing under `adapter/` is touched:

| Rule | Override |
|---|---|
| dedup | `duplicate_iou_threshold=1.1` (IoU is bounded [0,1], never reached) |
| size filter | `plausible_size=(0.0, inf)` |
| shape filter | `plausible_shape=(0.0, inf)` |
| implausible-displacement | `max_speed_px_per_frame=inf` |
| unsupported/flicker | `min_supported_track_length=0` |
| static | `min_static_run_frames=1e9` |
| interpolation (stage 4) | `max_dropout_frames=0` — **see caveat below, not a clean isolation** |
| selection (stage 5) | `class_max_instances=1e9` |

`calibration/ablation.py`'s `classify_clip()` calls each stage's own public
sub-rule function directly (`reject_duplicates`, `reject_implausible_size`,
`reject_implausible_shape`, `reject_implausible_displacement`,
`reject_unsupported`, `reject_static`, `apply_stage4`, `apply_selection`),
snapshotting tags between them, so every one of the 157,775 detections is
attributed to exactly one of 9 buckets: `kept`, `interpolated`,
`dropped_duplicate`, `rejected_size`, `rejected_shape`,
`rejected_displacement`, `rejected_unsupported`, `rejected_static`,
`rejected_selection`. This is finer than `sweep_thresholds.py`'s existing
`rejection_reason_frequency` (which lumps all of stage 3 into one
`rejected_temporal` bucket) — needed here so each of the three stage-3
sub-rules can be ablated independently and its own bucket checked. The
total-count invariant (`sum(buckets) == raw + fabricated`) is checked in
`tests/test_ablation.py` against an independently-recomputed total, the
same style of check that caught a real accounting bug in
`sweep_thresholds.py` previously.

Sensitivity: 11 series (7 scalar thresholds; `plausible_size` and
`plausible_shape` are tuples, swept one bound at a time with the other held
at its `hand_config()` default) over 8-9 points each, centered on and
including the exact current default. Reports `kept%`/`interpolated%` over
the same raw+fabricated total, across all 39 clips per point.

Per this codebase's own documented gotcha (`Detection.tag` mutated in place
by every stage), every single data point — every ablation, every sweep
point — calls `load_clip()` fresh per clip (`run_dataset`); no `Detection`
is ever reused across two configs.

Runtime on the real dataset: ablation suite (9 full-39-clip passes) 10.7s;
sensitivity suite (11 series x 8-9 points, ~85 full-39-clip passes) 116.7s.

**Key results:**

*Ablation* (all deltas vs. the 85.3% kept / 2.5% interpolated full-pipeline
baseline; full table in the results file):

- **dedup** (`no_dedup`): kept 85.3%→83.3% (-2.0pp). `dropped_duplicate`
  drops to 0% as expected, but the freed-up duplicate boxes don't just
  become "kept" — `rejected_selection` more than triples (1.3%→4.5%,
  +3.2pp), because undeduped near-identical boxes now compete for the
  same 2-instance-per-frame cap stage 5 enforces. Confirms dedup and
  selection are doing complementary work, not redundant work.
- **size filter** (`no_size_filter`): zero effect (kept 85.3%→85.3%,
  `rejected_size` already 0.0% in the full pipeline). Independently
  confirms Milestone 7's finding that the tuned `(50, 1150)` bound is
  already a near-total no-op on this dataset's raw detections — this
  ablation is functionally a repeat of that finding from a different
  angle, not a new one.
- **shape filter** (`no_shape_filter`): kept 85.3%→87.8% (+2.5pp),
  `rejected_shape` 2.6%→0.0%. The single largest positive kept-rate
  effect of any single-rule ablation — this rule is doing real, active
  work, unlike the size filter.
- **implausible-displacement** (`no_displacement_check`): kept
  85.3%→85.8% (+0.5pp), `interpolated` drops slightly (2.5%→2.1%) since
  fewer detections get rejected-then-recovered by stage 4. Smallest
  effect of the three stage-3 sub-rules — consistent with Milestone 4's
  own finding that this threshold rarely fires by design (110px/frame
  sits just above real p99.5 speed).
- **unsupported/flicker** (`no_flicker_check`): kept unchanged at 85.3%,
  `rejected_unsupported` 0.14%→0.0% — already the smallest bucket in the
  full pipeline, essentially inert on this dataset.
- **static** (`no_static_check`): kept 85.3%→90.1% (+4.8pp, the largest
  swing of any single ablation), `rejected_static` 4.8%→0.0%. This is the
  single most consequential rule in the pipeline by volume, consistent
  with planning.md's own open question that the static rule still can't
  distinguish a background object from a hand braced motionless for
  several seconds — every one of those still-legitimate detections is
  currently being paid for as the price of catching real background.
- **interpolation** (`no_interpolation`): kept collapses to 0.0%,
  `rejected_unsupported` balloons to 94.1%. **This is not a clean ablation
  of stage 4 — see caveat below**; it's a real finding about a config
  field, not about the interpolation rule's actual contribution.
- **selection** (`no_selection_cap`): kept 85.3%→86.2% (+0.9pp),
  `rejected_selection` 1.3%→0.0%. Modest — the 2-hand cap is already close
  to non-binding most of the time on this dataset.

*Sensitivity* (kept% range across the swept points; ✓ = sits on a flat
plateau around the current default, ⚠ = knife-edge — see verdict
methodology below):

| Threshold | Default | Range swept | kept% range | Verdict |
|---|---|---|---|---|
| `max_speed_px_per_frame` | 110 | 50-500 | 81.0%-85.8% | ✓ flat (max step 2.7pp) |
| `track_gate_speed_px_per_frame` | 350 | 150-1000 | 85.2%-85.4% | ✓ essentially flat everywhere (max step 0.2pp) |
| `max_dropout_frames` | 15 | 0-100 | 0.0%-86.9% | ⚠ knife-edge, but ONLY between 0 and 1 (see caveat); flat around the actual default (15) |
| `min_static_run_frames` | 15 | 1-100 | 69.9%-89.4% | ⚠ knife-edge, steepest at the low end (1→10) |
| `static_px_threshold` | 4.0 | 1-20 | 54.9%-90.1% | ⚠ knife-edge across its whole swept range, no flat plateau found |
| `duplicate_iou_threshold` | 0.5 | 0.1-0.9 | 83.2%-85.4% | ✓ flat, with a small real step at 0.6→0.7 (1.7pp) matching the documented IoU-distribution gap in `hand_config.py` |
| `duplicate_containment_threshold` | 0.7 | 0.0-1.0 | 83.9%-85.3% | ✓ flat everywhere except the boundary value 1.0 |
| `plausible_size` (lower bound) | 50 | 0-300 | 18.2%-85.3% | ⚠ knife-edge above ~120px; flat 0-80px |
| `plausible_size` (upper bound) | 1150 | 800-2500 | 85.2%-85.3% | ✓ completely flat — confirms Milestone 7's finding this bound is a non-issue |
| `plausible_shape` (lower bound) | 0.5 | 0.1-0.8 | 58.8%-85.8% | ⚠ knife-edge above ~0.5; flat below it |
| `plausible_shape` (upper bound) | 2.0 | 1.2-5.0 | 71.1%-87.0% | ⚠ knife-edge below ~2.0; flattens out above it |

Full per-point numbers (including `interpolated%` at every point) are in
`calibration/ablation_results.json` and the script's stdout.

**What this proves / does not prove:**

- Proves, without any labels, which of the 8 rules are doing real work on
  this dataset vs. which are structurally near-inert (size filter,
  flicker check) — a real, unsupervised signal about where calibration
  effort matters, same category of evidence as `sweep_thresholds.py`'s
  rejection-reason frequencies, just finer-grained and counterfactual
  (what changes when a rule is OFF) rather than descriptive (where a rule
  fires when it's ON).
- Proves several thresholds (`track_gate_speed_px_per_frame`,
  `duplicate_iou_threshold`, `duplicate_containment_threshold`,
  `plausible_size`'s upper bound) sit on genuinely flat plateaus — the
  current defaults could move substantially with negligible effect on
  kept%, which is a defensible (if informal) robustness claim for those
  four.
- Identifies `static_px_threshold`, `min_static_run_frames`,
  `plausible_size`'s lower bound, and both `plausible_shape` bounds as
  knife-edge — small changes there swing kept% by double-digit percentage
  points. These are exactly the thresholds where a precision/recall-optimal
  value (needing labels, per Milestone 7's still-open status) would matter
  most; this ablation narrows down WHERE to spend that future labeling
  effort, without being able to say what the optimal value is.
- Does **not** establish correctness or precision/recall for any
  configuration — "kept%"/"interpolated%" are proxies for pipeline
  behavior, not confirmed true/false positive rates (no ground truth
  exists for this dataset — see `metrics.py`'s docstring). A rule firing
  more or less doesn't by itself say whether it's now more or less
  correct.

**Caveats:**

- **`max_dropout_frames` cannot be cleanly ablated for stage 4 alone.**
  `association.py` (stage 2, the tracker) reads the *same* config field as
  its own patience window for expiring a stale track
  (`frame_idx - track.last_frame > config.max_dropout_frames`), independent
  of `interpolation.py`'s use of it for the gap-fill cap. At
  `max_dropout_frames=0`, a track expires the instant a single frame
  passes with no matching detection — given this dataset's real per-frame
  detection dropout, that fragments nearly every track down to 1-2
  detections, which stage 3's flicker rule (`min_supported_track_length`)
  then rejects almost entirely (`rejected_unsupported` balloons to 94.1%,
  kept collapses to 0.0%). This was found BY the ablation, not designed
  around: `hand_config(**overrides)` genuinely has no lever that disables
  stage 4's fill in isolation without also touching stage 2, because both
  read the one field. Reported as a real finding about this field's
  cross-stage coupling, not as evidence about what stage 4 itself
  contributes to the pipeline. The sensitivity sweep on the same field
  (which includes 0) shows the identical cliff directly, and shows it's
  entirely local to the 0→1 boundary — everything from 1 upward behaves
  smoothly, and the actual default (15) sits on the flat part of that
  curve.
- The "knife-edge vs. flat" verdicts are computed from a fixed rule (a
  single-step kept% swing over 5 percentage points anywhere in the swept
  range = knife-edge) — a threshold can be flagged knife-edge because of
  one boundary artifact (e.g. `duplicate_containment_threshold` only moves
  at exactly 1.0) rather than genuine sensitivity across its whole
  plausible range. Read the per-point table, not just the verdict label,
  before citing a threshold as fragile.
- Sweep ranges are hand-chosen (centered on the current default, informed
  by the real percentiles already documented in `hand_config.py`/
  `types.py`), not exhaustive — a threshold could still behave
  non-monotonically or have a second cliff outside the tested range.
- All numbers are aggregated across all 39 clips. Per planning.md's own
  standing note, every clip is a different job/task; a threshold flagged
  "flat" in aggregate could still be knife-edge for one specific job's
  population and simply outvoted by the other 38 clips in the sum. Not
  broken out per-job here.

---

## Open items for the paper (fill in once all three land)

- [ ] Merge all three eval branches back into `paper`
- [ ] Consolidate duplicated clip-iteration helper code across the three scripts
- [ ] Pull final numbers into the paper's Results section
- [ ] Update Limitations section to reflect what label-free eval can't establish
