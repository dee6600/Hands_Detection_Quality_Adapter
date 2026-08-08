# Detection Quality Adapter — Project Plan

Source spec: `hand_detection_spec.pdf` (Detection Quality Adapter — Part 1 generic, Part 2 hand-specialized)

Goal: build a post-processing stage that sits between a hand detector and downstream
consumers, correcting false positives and false negatives using temporal (multi-frame)
logic, without touching or retraining the detector itself.

This plan is written so each milestone can be handed to Claude as a self-contained
prompt. Work top to bottom — later stages depend on data structures and outputs from
earlier ones. Milestones 1–5 need NO dataset; they can be fully built and unit-tested
on synthetic data. Milestone 6 is hand-specific. Milestone 7 needs real labeled data.

## Dataset (arrived — 39 clips; `load_clip` verified against all 39)

Inspected in depth so far: `0c54a47b_t010`, `407258cd_t036`, `ae580129_t057`,
`6cd0b236_t000`, `130d64c7_t054`, `7fe10737_t009`.

Each clip bundle: `video_left.mp4`, `video_right.mp4` (ZED stereo, 1920x1200 @30fps
CFR), `frame_ts.json` (frame index → unix ns), `hand_boxes.json` (raw WiLoR detector
output, xyxy pixels, class 0/1 = detector's left/right guess, confidence, **no cap,
no dedup, no forward-fill**), `vio_pose.json` (full-rate 6DoF pose: x/y/z/roll/pitch/
yaw/speed, clip-relative seconds), `meta.json` (task label, provenance).

Confirmed from the actual `t010` clip (2700 frames / 90s):
- 2 boxes: 79.6% of frames-with-detections · 1 box: 12.7% · 3+: 7.7% (max 5) ·
  0 boxes: 5 frames — roughly matches the README's aggregate stats across all 39 clips
  (72.3% / 18.2% / 5.25% 3+, max 6, 4.3% zero-detection).
- No `imu.json` in this bundle — confirmed absent, `meta.json` records it explicitly.
- **No stereo calibration/intrinsics file is included.** `video_right.mp4` gives a
  second view but there's no baseline/intrinsics/extrinsics to turn it into per-box
  depth yet — this blocks the Milestone 6 stereo-depth rule until resolved (see Open
  Questions).
- Camera motion is directly available and better than the optical-flow stub originally
  planned: `vio_pose.json` gives real per-frame 6DoF pose at native rate, joined to
  `hand_boxes.json`/`frame_ts.json` by frame index. This replaces the Milestone 4
  camera-motion stub outright — no need to estimate it from pixels.
- Everything is delivered as whole clips, not a live stream → this is an **offline,
  non-causal, per-clip batch** problem. The tracker/interpolator can look both forward
  and backward within a clip.

---

## Working strategy (added at the Milestone 4 checkpoint, applies going forward)

Confirmed by three real fixes so far (Milestone 6's `max_reach_m`, Milestone 4's
static-rule rewrite, and the Milestone 3/4 speed-gate split below): **don't wait for
Milestone 7 to sanity-check a placeholder threshold against real data.** Most of
Milestone 7 is genuinely blocked on labeled ground truth (precision/recall need
labels) — but a threshold that's just wrong on its face can usually be caught earlier,
without labels, using the real clips already in hand:

1. **Check every placeholder against ALL 39 clips, not one or two**, before trusting
   it. A single clip can be a different population, not an outlier — see the
   `ae580129_t057` speed finding below, where one clip's camera moves 2-4x faster
   than the rest for task reasons (a maintenance job vs. hairstyling/carpentry), and
   averaging over it or ignoring it would both have been wrong.
2. **When a threshold looks off, get the distribution, not just a fix.** Percentiles
   across the real dataset (no labels needed) usually reveal whether the current
   default is in the right neighborhood, and whether the problem is "wrong number"
   or "wrong shape of rule" (the static-rule fix needed a sustained-run requirement,
   not just a smaller pixel threshold — a shape problem a single number couldn't fix).
3. **Pull the actual outlier frames/boxes before trusting a statistic.** Aggregate
   percentiles told us *that* something was off; looking at specific frames (e.g.
   t010 frame 16, t036 frame 220, the `ae580129_t057` track-41 sequence) told us
   *why*, and changed the fix each time.
4. **Document the mechanism and the residual limitation, not just the new default.**
   Every fix so far solves the common case and leaves a harder edge case open
   (bracing hands for the static rule, camera-motion-relative speed for the
   displacement rule) — write both down (see `detection_quality_adapter/README.md`)
   so the gap is a tracked open question, not a silent gap.
5. **Still defer to Milestone 7** anything that genuinely needs labels to validate —
   precision/recall tradeoffs, or a fix whose correctness depends on comparing
   against ground truth rather than just "is this number in a sane range."

---

## Milestone 0 — Project scaffolding ✅ DONE

**Goal:** repo structure, environment, no logic yet.

- [x] Create project structure:
  ```
  detection_quality_adapter/
    adapter/
      __init__.py
      types.py          # Detection, Track, Config dataclasses
      geometric.py       # stage 1: duplicate/size/shape rejection
      association.py     # stage 2: tracker
      temporal.py         # stage 3: displacement/flicker/static rejection
      interpolation.py    # stage 4: gap fill + exit detection
      selection.py         # instance-cap selection after association
      hand_config.py       # Part 2: hand-specific parameters
      pipeline.py           # orchestrates stages in fixed order
      ingest.py              # NEW: loads a clip bundle into Detection/pose objects
    tests/
      test_geometric.py
      test_association.py
      test_temporal.py
      test_interpolation.py
      test_pipeline_synthetic.py
    calibration/
      sweep_thresholds.py   # Milestone 7, needs real data
      metrics.py             # per-stage precision/recall
    data/                     # empty until dataset arrives
    requirements.txt
    README.md
  ```
- [x] Set up virtual env, `numpy`, `pytest`, optionally `scipy` (Hungarian matching) and
  `filterpy` (Kalman filter) as candidates for Milestone 3.
- [x] Decide language/runtime version and commit an empty pipeline that just passes
  detections through untouched, with a passing smoke test.

**Prompt to give Claude:** "Scaffold the repo structure above, empty modules, one
smoke test that imports everything and runs a no-op pipeline call."

---

## Milestone 1 — Data model & contracts ✅ DONE

**Goal:** lock down the shapes everything else builds on.

- [x] `Detection`: box (x, y, w, h or corners), confidence, frame_id/timestamp,
  class label, source flag (`reported` at input).
- [x] `Track`: id, ordered list of Detections, state (active / ended / exiting),
  predicted next position (for association + exit test).
- [x] Output tag enum: `reported | merged | rejected | interpolated`.
- [x] `Config`: per-class expectations — plausible size range, plausible shape
  (aspect ratio range), candidate pool size, class max instances, max speed
  (px/frame or px/sec), exit-border weighting.
- [x] Write this as plain dataclasses/structs first — no behavior, just shape —
  so every later stage can be unit tested against hand-built fixtures.

**Prompt to give Claude:** "Implement `types.py` with these dataclasses. Add a
`Config` with sensible placeholder defaults for a generic object, not hands yet."

**Exit criteria:** you can hand-construct a list of `Detection`s in a test file
and pass them through a no-op pipeline that returns them unchanged but tagged
`reported`.

**Note (added at the Milestone 4 checkpoint):** `Config` has grown well past
the fields listed above as later milestones needed more granular thresholds
(`duplicate_iou_threshold`, `min_supported_track_length`, `static_px_threshold`,
`camera_moving_speed_mps`, `camera_moving_angular_deg_per_frame`,
`min_static_run_frames`, and a split `max_speed_px_per_frame` /
`track_gate_speed_px_per_frame` — see Milestone 4's notes). Check
`adapter/types.py` directly for the current field list rather than trusting this
document's Milestone 1 description, which only reflects the original placeholder set.

---

## Milestone 1.5 — Data ingestion (real bundle → internal types) ✅ DONE

**Goal:** one loader that turns a clip folder into the objects Milestones 2–6
operate on. This is new now that the dataset has actually arrived — do it
right after Milestone 1, in parallel with (not instead of) the synthetic-data
work in Milestones 2–5.

- [x] `load_clip(clip_dir) -> ClipData`, where `ClipData` bundles:
  - `detections: list[list[Detection]]` — one list per frame index, built from
    `hand_boxes.json` (note: frames with zero detections are **omitted** from
    the file, not present as empty lists — the loader must reconstruct the
    full-length `[0..frame_count)` sequence with empty lists filled in).
  - `frame_ts: dict[int, int]` — from `frame_ts.json`, unix ns per frame.
  - `pose: list[PoseSample]` — from `vio_pose.json`'s columnar arrays
    (`t, x, y, z, roll, pitch, yaw, speed`), zipped into per-frame records.
    Confirm `vio_pose.json`'s `n` matches `frame_ts.json`'s `frame_count` (off
    by one is common with this kind of bundle — check `t010`: `frame_ts` has
    2700 frames, `vio_pose` has 2701 samples; reconcile the indexing before
    building anything downstream on top of it).
  - `meta: dict` — pass through `meta.json` as-is (task label, provenance).
- [x] Map `hand_boxes.json`'s `class 0/1` field into `Detection`'s handedness
  field but **do not** treat it as a stable identity — every downstream stage
  must already ignore it per the spec, this is just carrying it through for
  debugging/visualization.
- [x] Write one integration test that loads the real `t010` bundle end-to-end
  and asserts basic sanity: frame count matches `meta.json`, no crashes on the
  handful of zero-detection frames, pose array aligns with frame timestamps.

**Prompt to give Claude:** "Implement `ingest.py`'s `load_clip` against the
real file schemas in the README (paste the README + a `meta.json` sample).
Handle the `frame_ts` vs `vio_pose` length mismatch explicitly rather than
assuming they're the same length. Write one integration test against the
uploaded `t010` bundle."

**Exit criteria:** `load_clip` on the real bundle produces a `ClipData` object
that Milestone 2's geometric-rejection stage can already run against, frame by
frame, without modification.

**Done:** the `frame_ts`/`vio_pose` off-by-one turned out to be consistent
(`n == frame_count + 1`) across every clip checked, not `t010`-specific — see
`adapter/ingest.py`'s `_load_pose` docstring. Tests in `tests/test_ingest.py`
now run against 4 clips (not just `t010`), chosen to also cover a dense
zero-gap clip and two clips with real dropout gaps.

---

## Milestone 2 — Geometric rejection (Stage 1, per-frame only) ✅ DONE

**Goal:** the three per-frame rules, in fixed order, no tracking required yet.

Order matters (spec §3):
1. **Duplicate detections** — heavily overlapping boxes on one object → merge,
   keep the stronger, tag `merged`.
2. **Implausible size** — reject boxes far outside plausible size at working
   distance.
3. **Implausible shape** — reject boxes markedly more elongated than the
   object should be.

- [x] IoU-based duplicate merge function, testable with two overlapping synthetic
  boxes → confirm one `merged` output survives.
- [x] Size filter against `Config.plausible_size`.
- [x] Shape filter against `Config.plausible_shape` (aspect ratio bounds).
- [x] Unit tests: one synthetic frame per rule, plus one frame that passes clean.

**Prompt to give Claude:** "Implement `geometric.py` with `reject_duplicates`,
`reject_implausible_size`, `reject_implausible_shape`, applied in that order.
Write pytest cases for each using hand-built boxes — no dataset needed."

**Exit criteria:** stage 1 runs standalone on a list of per-frame candidate
boxes and returns a filtered/merged/tagged list. Fully testable now.

**Done:** implemented as greedy NMS (confidence-sorted, IoU-gated) for
duplicates, then size/shape filters. Real-data test suite
(`tests/test_geometric_real_data.py`) checks structural invariants (no
crashes, no residual duplicates, count never increases) across 3 real clips,
plus a test locking in a confirmed real duplicate pair (`t010` frame 5, two
overlapping right-hand boxes). `scripts/visualize_stage1.py` renders a real
clip color-coded by what stage 1 did to each box — see
`detection_quality_adapter/README.md` for the color legend and example.

---

## Milestone 3 — Association (Stage 2, the tracker) ✅ DONE

**Goal:** turn per-frame detections into per-object tracks. This is the piece
everything downstream depends on — worth the most care.

- [x] Decide matching strategy: start simple — greedy nearest-neighbor using
  predicted position (linear extrapolation from last known velocity) — before
  reaching for a Kalman filter or Hungarian algorithm. Upgrade only if the
  simple version isn't good enough on synthetic sequences.
- [x] A detection close to a predicted position extends that track.
- [x] A detection far from any active track starts a new track.
- [x] Track object needs: history, current predicted position/velocity,
  "active" vs "no detection this frame but still within patience window" state.
- [x] Unit tests: synthetic multi-frame sequences —
  - one object moving smoothly → one continuous track
  - one object with a 2-frame gap → track continues, not two tracks
  - two objects crossing paths → tracks don't swap identities
  - a new object entering mid-sequence → new track, not merged into existing

**Prompt to give Claude:** "Implement `association.py`. Start with greedy
nearest-neighbor + linear motion prediction. Write synthetic multi-frame test
sequences covering: continuous motion, short gap, crossing paths, new object
entry. Flag in comments where a Kalman filter or Hungarian matching would
improve robustness, but don't build them yet."

**Exit criteria:** a list of per-frame detection lists in → a list of Tracks
out, correctly separating and continuing objects across synthetic sequences.

**Done:** greedy NN + linear velocity extrapolation, gated by
`Config.track_gate_speed_px_per_frame` scaled by frames-since-last-seen (a
coasting track gets a wider gate the longer it's been coasting). Real-data
tests (`tests/test_association_real_data.py`) caught a real ordering bug: a
stale track's patience was originally checked *after* matching instead of
before, so an ever-widening gate could resurrect a track that should have
already ended — fixed by expiring stale tracks at the top of each frame's
processing. `scripts/visualize_tracks.py` renders a real clip with a
persistent color per track plus a fading trail, for eyeballing identity
continuity/fragmentation directly. See the Milestone 4 entry below for the
gate-vs-plausibility speed split this milestone's `max_speed`-based gate
eventually needed.

---

## Milestone 4 — Temporal rejection (Stage 3, needs tracks) ✅ DONE

**Goal:** the three rules that only make sense once tracks exist (spec §3,
rules 4–6).

- [x] **Implausible displacement** — box jumps further between consecutive
  frames than `Config.max_speed` allows → reject.
- [x] **Unsupported detection** — appears for 1–2 frames with no track before
  or after → reject as flicker.
- [x] **Static detection** — doesn't move while the camera does → reject as
  background structure. Camera motion is now a real signal, not a stub: use
  `vio_pose.json` (via `ingest.py`'s `ClipData.pose`) — frame-to-frame delta in
  position and/or yaw/pitch/roll gives "is the camera moving" directly, no
  optical flow needed. Define a motion threshold (e.g. speed or angular rate
  above some epsilon) rather than a boolean stub.
- [x] Unit tests: synthetic tracks — one flicker-only track, one track
  exceeding max speed, one static track against a "moving camera" stub, one
  clean track that should survive all three.

**Prompt to give Claude:** "Implement `temporal.py`'s three rejection rules
operating on Track objects from Milestone 3. Stub camera motion as a boolean
per frame for now. Write synthetic test tracks for each rejection rule plus
one clean pass-through case."

**Exit criteria:** tracks in → tracks with implausible/unsupported/static
detections tagged `rejected`, rest untouched.

**Done, in three passes** (full mechanism, numbers, and screenshots in
`detection_quality_adapter/README.md` — summarized here):

1. **Initial implementation.** All three rules against real VIO pose data
   (no stub). Real-data tests confirmed it runs cleanly and rejects a
   non-trivial, non-total fraction of survivors.
2. **Static rule rewrite** (user caught this by eyeballing
   `407258cd_t036`): the original per-frame threshold flagged real hands for
   pausing even one frame, because a head-mounted camera is "moving" by a
   low-bar definition almost continuously, and rotation-dominated camera
   motion means the hand the wearer is *looking at* is often the thing in
   frame with the *least* apparent motion — nearly backwards from what the
   rule wants. Fixed by requiring `min_static_run_frames` (15) *consecutive*
   still-while-moving frames, not one. Cut `t010`'s static false-rejections
   from 261 to ~0. **Still open**: a hand braced motionless for several
   *seconds* (e.g. `t036` frame 220, steadying a workpiece) still gets
   flagged — no sustained-run window is both long enough to admit that and
   short enough to catch real background. Real fix needs to separate
   rotation-induced apparent motion from translation-induced parallax
   instead of using a flat pixel threshold — deferred to Milestone 7.
3. **Speed gate/plausibility split** (this checkpoint, prompted by the user
   noticing detections still jumping too far): per-clip speed percentiles
   across all 39 clips showed the shared `max_speed_px_per_frame=150` sat at
   ~p99.9 of 38 clips' real adjacent-frame hand speed (p99=82.6, p99.9=132)
   — so loose that stage 3's displacement rule almost never fired, even
   though real "motion-blur wobble" moments inside long, clearly-continuous
   tracks should have been caught. One clip (`ae580129_t057`, a maintenance
   task) turned out to be a genuinely different population (camera moves
   2-4x faster on average) rather than an outlier to average away. Split
   into two config fields: `max_speed_px_per_frame` (110, tightened, used
   only by the displacement-plausibility check) and
   `track_gate_speed_px_per_frame` (350, generous, used only by the
   Milestone 3 tracker's match gate) — conflating them meant tightening one
   to catch real wobble moments was simultaneously loosening or fragmenting
   the tracker's own identity continuity. Effect on `t010`: track count
   dropped 65→44 (less gate-driven fragmentation) while stage 3's rejection
   rate rose 0.6%→1.5% (catching more real wobble without breaking tracks
   to do it). **Still open**: making `max_speed_px_per_frame` scale with the
   clip's own VIO camera speed at each frame, so a naturally-faster task
   like `ae580129_t057` doesn't inherit a systematically higher
   false-rejection rate — needs labeled data to validate the scaling factor,
   deferred to Milestone 7.

---

## Milestone 5 — Interpolation + exit detection (Stage 4)

**Goal:** fill genuine gaps, but never invent an object that left the frame
(spec §4).

- [ ] Exit test: near a frame border + motion directed outward → track ends,
  no interpolation.
- [ ] Otherwise, if a track resumes near its predicted position after a gap
  within the max-dropout-length parameter → interpolate the missing frames,
  tag `interpolated`.
- [ ] A break that doesn't resume, and isn't an exit either, is left untouched
  (no fabricated detections, no forced closure).
- [ ] Unit tests: synthetic tracks —
  - object exits at border with outward motion → no interpolation
  - object occluded mid-frame, resumes near prediction → interpolated
  - object gap exceeds max dropout length → left untouched
  - object gap resumes far from prediction → new track, not bridged

**Design notes from the spec cross-check (added at the checkpoint, read
before implementing):**

- **A "gap" for interpolation purposes must include `rejected` frames, not
  just literally-missing ones.** The spec's edge case table treats "motion
  blur during rapid movement — confidence falls across several consecutive
  frames" as an *interpolate* case, not a *reject* case. But Milestone 4's
  displacement rule (tightened at the checkpoint specifically to catch
  motion-blur "wobble" — see its notes above) now tags some of those exact
  frames `rejected`, since the box position momentarily looks implausible
  even though the underlying trajectory is real. These aren't in conflict
  *if* Milestone 5 treats a track's `rejected`-tagged detections the same as
  an absent frame when looking for gaps to fill — i.e. don't trust a
  rejected detection's position, but do let the surrounding trajectory
  predict through it. If Milestone 5 only fills gaps where a frame is
  missing entirely, the Milestone 4 tightening becomes a net loss: it
  correctly flags the bad frame but nothing then recovers it, which is
  exactly the false-negative the spec's edge case wants recovered.
- **The bottom-border exit test likely needs a different trigger condition
  than the side-border one, not just a different weight.** The spec gives
  "hand exits at a side border" and "hand passes below the camera" the same
  required handling (terminate, don't interpolate) but different
  observations: the side case is "motion directed outward," the bottom case
  is "occluded by the torso." A hand disappearing behind the wearer's own
  body isn't guaranteed to show a clean outward velocity right before it
  vanishes — it can be moving sideways, or even slightly upward, as an arm
  bends and drops out of the lower frame. Milestone 6's plan already says
  the bottom border should be *weighted* more heavily, but weighting alone
  won't help if the exit test's core logic still hard-requires outward
  motion. Consider a separate, more permissive trigger for the bottom
  border specifically (e.g. "last detection near the bottom edge" may be
  sufficient on its own, without also requiring outward velocity).

**Prompt to give Claude:** "Implement `interpolation.py`: exit-border test
first, then gap-fill logic gated by max dropout length and predicted-position
proximity. Write the four synthetic test cases listed above."

**Exit criteria:** full generic Part 1 pipeline (stages 1–4) runs end to end
on a synthetic multi-frame sequence and produces correctly tagged output.
This is a good checkpoint to wire `pipeline.py` together and add an
integration test.

---

## Milestone 6 — Hand specialization (Part 2)

**Goal:** layer hand-specific parameters and rules on top of the generic
adapter — mostly config, four behavioral changes.

- [ ] `hand_config.py`: class max = 2, candidate pool = 3–4, shape rule tuned
  for roughly-equant boxes, exit-border weighting favors the bottom border.
  **Also needs a hand-specific duplicate-merge override** — see the new
  bullet below on hands crossing/overlapping; the generic IoU-only dedup
  from Milestone 2 is a real risk here, not just a theoretical one.
- [ ] **Selection after association** (`selection.py`): with a low,
  constantly-reached cap, implement "rank remaining candidates by track
  quality, select top 2" rather than per-frame confidence.
- [x] **Static-detection rule is unusually strong here, and needed real code
  changes, not just confirmation.** The spec's assumption — "a hand is never
  stationary in the image for a sustained period" because the camera is
  always moving — turned out not to hold universally: `407258cd_t036` (a
  carpentry clip) shows a hand braced motionless for several *seconds* while
  steadying a workpiece (frame 220), which is exactly a sustained stationary
  period despite continuous camera motion. Milestone 4's sustained-run fix
  (`min_static_run_frames`) handles brief pauses but not this multi-second
  brace — see Milestone 4's "Done" notes and the open question below. Worth
  remembering this is a place where the spec's own stated assumption doesn't
  universally hold, not just an implementation gap.
- [x] **Handedness**: tracker must never use the left/right label to decide
  association — done ahead of schedule during Milestone 3
  (`test_handedness_label_never_influences_association` in
  `tests/test_association.py`), since it was cheap to add right after the
  crossing-paths test and didn't need Milestone 6's other pieces first.
- [ ] **NEW (found at the spec cross-check): hands crossing/overlapping vs.
  true duplicates need a hand-specific dedup rule, not just the generic
  IoU threshold.** The spec's edge case table is explicit: "Hands cross or
  overlap ... Retain both. Must not be treated as a duplicate" — but
  Milestone 2's `reject_duplicates` runs per-frame, before any tracking
  exists, so it has no way to know whether two overlapping boxes are one
  object detected twice or two real hands that happen to overlap; it merges
  on IoU alone. Checked real data for this: scanning all 39 clips for
  2-detection frames with IoU in the current merge range (≥0.5) didn't turn
  up a clear "two real distinct hands wrongly merged" example — the
  moderate-IoU real cases found all looked like genuine duplicate echoes
  (one strong box, one weaker, nested/offset, same class label), not two
  independent hands. So this isn't visibly biting yet in the tested clips,
  but the risk is real and spec-named, and absence of an observed case
  isn't the same as absence of the failure mode (hands actually crossing is
  probably just rarer than generic 2-box frames in this data). Two possibly
  useful, evidence-backed refinements for hand_config.py: (a) a tighter
  hand-specific `duplicate_iou_threshold` than the generic default, and/or
  (b) a containment-ratio check (intersection / smaller-box-area) in
  addition to IoU, since the real duplicate pattern observed was one box
  mostly *inside* the other with a confidence gap — genuinely distinct
  crossing hands are more likely to be similar-sized and laterally offset,
  which plain IoU doesn't distinguish but containment ratio might. Needs an
  explicit synthetic test per the spec's edge case (two similar-sized,
  offset, non-nested boxes → both retained) before trusting either fix.
- [x] **Stereo depth check** — prototype DONE at the nominal-calibration
  tier, but **NOT YET migrated into the real `adapter/` package** — this
  milestone can't be marked fully done until that happens. Currently lives
  in `exp/scafholds/nominal_calibration.py` + `exp/scafholds/stereo_depth.py`
  (a standalone prototype, not part of `main/detection_quality_adapter/`),
  tested against synthetic cases and validated on the real `t010` clip:
  - Calibration: ZED X, 2.2mm lens (0.3m min depth), 12cm baseline — chosen
    because 4mm (1.5m min depth) can't explain the near-frame-filling hand
    boxes actually seen in the data. Cross-checked: a real high-confidence
    box (frame 230) gave disparity ~267px → ~0.33m depth, right at the 2.2mm
    lens's minimum focus distance — exactly what you'd expect for a box that
    large. See the module docstring for the full derivation.
  - Discovered the two eyes are **not perfectly vertically aligned** (~2.4px
    mean offset, up to ~9px, from ORB+RANSAC feature matching) — likely
    unrectified output and/or slight temporal skew between eyes on this
    moving rig. The depth-per-box matcher uses a vertical search tolerance
    rather than assuming a strict horizontal epipolar line.
  - Per-box depth via template matching (patch around box center in the left
    frame, searched in a vertical band of the right frame) — median match
    score 0.95 across ~150 real high-confidence hand detections in `t010`.
  - `max_reach_m` default set to **1.8m**, not the more intuitive ~0.8m —
    empirically, 72% of this clip's own-hand detections exceed 1.0m because
    it's a hairstyling task with arms extended toward a seated client. Still
    a placeholder pending Milestone 7 calibration, and probably needs to vary
    by job/task type.
  - 13/13 tests pass (synthetic + one real-clip integration check). See
    `exp/scafholds/test_stereo_depth*.py`, and run the demo with
    `python exp/scafholds/demo_stereo_depth.py --hand-boxes ... --left ... --right ...`.
  - **Migration TODO**: port `nominal_calibration.py`/`stereo_depth.py` (and
    the "beyond arm's reach" rejection rule) into
    `main/detection_quality_adapter/adapter/`, wired into `hand_config.py`/
    `pipeline.py` like the other stages, with tests moved into
    `detection_quality_adapter/tests/`. Until this happens, the real
    pipeline can't actually run the stereo-depth rule end to end — it only
    exists as a validated standalone prototype.
- [ ] Implement the edge-case table from spec §6 as explicit test cases:
  duplicate on one hand, side exit, bottom exit (occluded by torso), brief
  occlusion, hands crossing, long-absence re-entry, motion-blur confidence
  dip, other person's hands, gloved hand (explicitly marked "behavior
  unmeasured — needs labelled examples", don't guess at logic here).

**Prompt to give Claude:** "Implement `hand_config.py` and `selection.py` per
the notes above. Add the stereo-depth rejection rule with a stubbed depth
source. Write one test per row of the spec's edge-case table (§6), using
synthetic data — mark the gloved-hand case as `xfail`/TODO since the spec
says it needs real data."

**Exit criteria:** full hand-configured pipeline passes all edge-case tests
on synthetic data. This is the last milestone before real data is required.

---

## Milestone 7 — Calibration & validation (needs the dataset)

**Goal:** replace every guessed threshold with one derived from a labeled
reference set (spec §7–8). Do this only once the dataset is downloaded.

- [ ] Build the reference set per spec's guidance: weighted toward hard cases
  — both hands, one hand, no hands, hands at border, borderline detections —
  not "representative" footage. Confirm it includes a meaningful fraction of
  zero-hand frames.
- [ ] `metrics.py`: precision/recall computed **per stage**, not aggregate —
  false-positive rejection should raise precision, trajectory recovery should
  raise recall, duplicate merging should raise both. Flag any stage moving the
  wrong direction.
- [ ] `sweep_thresholds.py`: derive from the reference set —
  - frequency of each false-positive class → prioritize calibration effort
  - distribution of dropout lengths → sets max interpolation gap
  - rate at which a true hand is outranked by a spurious box → sets
    candidate pool width
- [ ] Standing check: track proportion of `interpolated` detections over time
  as an early-warning signal (no labels needed) that the adapter has started
  fabricating rather than recovering.
- [ ] Replace stubbed camera-motion and stereo-depth signals with real ones
  once the corresponding data streams are available.

**Prompt to give Claude:** "Implement `metrics.py` for per-stage
precision/recall and `sweep_thresholds.py` once the dataset is in `data/`.
Wire real camera-motion and stereo-depth signals in place of the Milestone
4/6 stubs."

---

## Suggested working order / session plan

1. ~~Session 1: Milestones 0–1 (scaffold + types)~~ ✅ done
2. ~~Session 2: Milestone 1.5 (real data ingestion against `t010`)~~ ✅ done —
   `load_clip` has since been run against all 39 clips without incident
   (during the Milestone 4 checkpoint's speed analysis), not just the 4
   clips pytest covers
3. ~~Session 3: Milestone 2 (geometric rejection)~~ ✅ done — synthetic tests
   plus real-data checks and a visualization script
4. ~~Session 4: Milestone 3 (association/tracker)~~ ✅ done — greedy NN +
   linear prediction; real-data testing caught a real patience-ordering bug
5. ~~Session 5: Milestone 4 (temporal rejection)~~ ✅ done, wired to real VIO
   pose from the start, plus a mid-course checkpoint session (below)
6. **Checkpoint session (this one): refine planning/memory + recalibrate
   against real data now that ~half the milestones are built.** Not a new
   milestone — a deliberate pause to (a) keep this document and the
   assistant's memory in sync with what's actually built, since both had
   drifted behind the real state of the repo, and (b) chase down a
   correctness concern (detections jumping further than they should) with
   real multi-clip data instead of guessing. Produced the Milestone 4 speed
   gate/plausibility split above and the "Working strategy" section up top.
   Worth repeating this kind of pause every few milestones rather than only
   at the end.
7. Session 7: Milestone 5 (interpolation/exit) + wire `pipeline.py` end to end
8. Session 8: Milestone 6 (hand specialization + edge-case tests) — stereo
   depth rule is DONE against nominal calibration (see above); remaining
   Milestone 6 work is `hand_config.py`, `selection.py`, and the edge-case
   table tests, which still need Milestones 0-5's types/pipeline in place
   first (selection needs association output, etc.)
9. Once labels exist for a reference set: Milestone 7 (calibration/validation)
   — though see "Working strategy" above: don't wait until this session to
   sanity-check every placeholder against real data, only the parts that
   truly need labels (precision/recall). `plausible_size`, `plausible_shape`,
   and `max_dropout_frames` haven't had the same real-data pass that speed
   and static-motion got yet — good candidates for the next checkpoint,
   before Milestone 6, not necessarily after.

Each session should end with passing tests before moving on. Synthetic tests
remain the primary correctness check through Milestone 6; real-clip checks
are a secondary pass, not a substitute, until labels exist — but per the
Working strategy section, that secondary pass should happen continuously
across all 39 clips, not just once at the end.

## Open questions

Resolved by the dataset arrival:
- ~~Detector output format~~ → xyxy pixels, 1920x1200, conf ≥ 0.10, class 0/1
  = detector's left/right guess (unreliable, ignore for tracking identity),
  no cap/dedup/forward-fill. Confirmed in `hand_boxes.json`.
- ~~Camera-motion source~~ → real 6DoF VIO pose per frame (`vio_pose.json`),
  not IMU, not optical flow. No per-clip IMU exists in this bundle set at all.
- ~~Batch vs streaming~~ → offline, per-clip, non-causal. Full clip files are
  delivered up front.
- ~~`frame_ts`/`vio_pose` off-by-one, `t010`-specific or universal?~~ →
  universal: `load_clip` has now run cleanly against all 39 clips with the
  `n == frame_count + 1` assumption enforced (fails loudly otherwise, per
  `ingest.py`), not just the 4 clips pytest exercises. Confirmed during the
  Milestone 4 checkpoint's all-clips speed analysis.

Found during Milestones 2-4 and the Milestone 4 checkpoint (see their "Done"
notes above for full detail):
- **Static-rejection rule can't yet tell "background" from "hand held still
  on purpose" over multi-second timescales — and this is a place the spec's
  own stated assumption doesn't universally hold, not just an implementation
  gap.** The spec says (§5): "Head-mounted camera: the camera moves
  constantly, so the static-detection rule is unusually discriminating. A
  hand is never stationary in the image for a sustained period." `t036`
  frame 220 is a direct counter-example: a hand braced motionless for
  several *seconds* while steadying a workpiece, well past "not sustained."
  The Milestone 4 sustained-run fix solved brief real pauses but not this.
  Properly separating rotation-induced apparent motion from real parallax —
  judging staticness against the *expected* motion at a box's position
  rather than a flat pixel threshold — needs labeled data to validate; a
  Milestone 7 candidate. Worth building the labeled reference set (§7) with
  this specific failure mode in mind — deliberately include some
  bracing/steadying frames, not just the border/borderline cases the spec
  already calls out.
- **A single flat speed threshold can't distinguish "fast hand" from "normal
  hand, fast camera."** `ae580129_t057`'s camera moves 2-4x faster on
  average than other clips (VIO speed), which shows up as systematically
  faster apparent hand motion with no change in real hand behavior. Scaling
  `max_speed_px_per_frame` by the clip's own VIO camera speed at each frame
  is the likely right fix; needs labeled data to validate the scaling
  factor, so deferred to Milestone 7 — but worth remembering it's a *task*
  difference (this clip is a maintenance job, not hairstyling/carpentry),
  not noise, so Milestone 7's reference set should probably report metrics
  per job/task type rather than only in aggregate.
- **`plausible_size`, `plausible_shape`, and `max_dropout_frames` haven't had
  a real-data sanity pass yet**, unlike speed and static motion. Given how
  much the speed threshold moved once actually checked (150 → 110, plus
  splitting it in two), these are worth the same treatment before assuming
  they're in a reasonable range — see "Working strategy" above.

Found during a full spec re-read at this checkpoint (spec text supplied in
full for the first time — prior sessions worked from summaries in this
document; see the Milestone 5/6 sections above for the full detail behind
each):
- **Milestone 2's duplicate-merge rule risks wrongly merging two real,
  overlapping hands** — the spec explicitly requires "hands cross or
  overlap ... retain both, must not be treated as a duplicate," but
  `reject_duplicates` runs pre-tracking and can't distinguish that from a
  true duplicate echo using IoU alone. Checked all 39 clips for a live
  example; didn't find a clear one yet (real duplicate-echo patterns in this
  data look different enough — nested/offset with a confidence gap — that
  plain IoU hasn't visibly misfired), but the failure mode is real and
  spec-named, not hypothetical. See Milestone 6's new bullet on this.
- **Interpolation (Milestone 5) needs to treat a track's `rejected`
  detections as gap-equivalent, not just literally-missing frames**, or the
  Milestone 4 speed-threshold tightening becomes a net loss on exactly the
  case the spec wants recovered ("motion blur ... interpolate where the
  trajectory remains continuous"). Not yet built, so not yet a live bug —
  flagged so Milestone 5 gets built with this from the start. See
  Milestone 5's design notes above.
- **The bottom-border exit test probably needs its own trigger condition**,
  not just a heavier weight on the bottom border as currently planned — see
  Milestone 5's design notes above.
- **Milestone 6's stereo-depth "done" status was overstated**: the
  implementation is a validated prototype in `exp/scafholds/`, not yet
  migrated into `main/detection_quality_adapter/adapter/`. Fixed in this
  document; the actual migration is still outstanding.

Still open:
- **Stereo calibration is nominal, not exact — resolved enough to build on,
  worth tightening later.** Implemented and validated (see Milestone 6 above)
  against a public ZED X datasheet + real-clip cross-checks: 2.2mm lens,
  12cm baseline, 733px focal length. Good enough for the coarse arm's-reach
  threshold, not for precision depth. If real per-unit calibration ever
  becomes available (it's normally embedded in the raw `.svo` file the ZED
  SDK produces, or a `.conf` file by serial number — and since `vio_pose.json`
  was clearly generated with the SDK's real calibration already, someone
  upstream has it), swap it into `nominal_calibration.py` and re-validate the
  `max_reach_m` threshold, since it was derived under the nominal numbers.
- The unrectified/misaligned stereo pair (vertical offset ~2-9px) is worth
  understanding better — is it truly unrectified raw ZED output, or a
  temporal sync gap between `video_left.mp4`/`video_right.mp4`? The two have
  different explanations and different fixes (rectify with real calibration,
  vs. find/apply a frame-offset correction). Not blocking — the vertical
  search tolerance in `stereo_depth.py` handles it either way — but worth a
  follow-up if depth precision needs to improve beyond the current coarse
  threshold.
- Formal, labeled precision/recall calibration (Milestone 7) is still fully
  blocked on the reference set for every threshold — informal real-data
  passes (speed, static motion) can catch "this number is obviously wrong"
  and "this rule has the wrong shape," but not "this is the precision/recall-
  optimal value," which genuinely needs ground truth.
- Should the adapter run per-clip independently (39 separate runs) or is there
  ever continuity across clips within a session worth preserving? Currently
  assuming per-clip independence, matching how the bundles are delivered.
