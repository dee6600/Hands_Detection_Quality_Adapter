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

## Milestone 5 — Interpolation + exit detection (Stage 4) ✅ DONE

**Goal:** fill genuine gaps, but never invent an object that left the frame
(spec §4).

- [x] Exit test: near a frame border + motion directed outward → track ends,
  no interpolation.
- [x] Otherwise, if a track resumes near its predicted position after a gap
  within the max-dropout-length parameter → interpolate the missing frames,
  tag `interpolated`.
- [x] A break that doesn't resume, and isn't an exit either, is left untouched
  (no fabricated detections, no forced closure).
- [x] Unit tests: synthetic tracks —
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

**Done:** both design notes above got implemented, not just noted —
`rejected` detections are excluded from a track's "trustworthy anchor"
sequence (so a stage-3 wobble flag gets recovered by interpolation instead
of just lost) and `Config.exit_border_margin_overrides`/
`exit_requires_outward_motion` let a border skip the outward-motion
requirement with its own margin (mechanism built now, generic defaults;
Milestone 6 sets the actual hand-specific bottom-border override). `pipeline.py`
is wired end to end (`run_pipeline(clip.detections, clip.pose)` running all
four stages), with a new `tests/test_pipeline_synthetic.py` satisfying the
exit criteria directly, and the two Milestone 0/1 placeholder pipeline tests
(`test_smoke.py`, `test_types.py`) updated from asserting the old no-op
passthrough to exercising the real pipeline.

Real-data testing caught two bugs, both instructive:
- **A test-writing bug, caught before ever running against real data**: the
  original gap-fill "prediction" check computed the predicted position from
  the gap's own two endpoints, then checked that prediction against one of
  those same endpoints — tautologically always zero distance, so any
  resumption would have passed regardless of plausibility. Found by
  hand-deriving the expected numbers for the "resumes far from prediction"
  test before running it, not by trusting a green test run. Fixed by
  predicting from the anchor's own incoming velocity (from whatever came
  before it), independent of where the gap resumes.
- **A real bug, only visible on real data**: the exit test compared box
  *center* distance to the frame border. Indistinguishable from edge
  distance on the small, fixed-size synthetic test boxes — so every
  synthetic test passed — but on real clips it produced **zero** confirmed
  exits across all three clips checked, despite hands obviously leaving
  frame in every one. This dataset's boxes are often large/near-frame-filling
  (noted since the dataset first arrived): a real box with `y2=1200`, exactly
  on the 1200px-tall frame's bottom edge, had its center 107px short of the
  default 20px margin. Fixed by measuring from the box's nearest edge
  instead. After the fix: 8/6/10 tracks correctly classified `exiting` on
  `t010`/`t036`/`ae580129_t057` (0 on `6cd0b236_t000`, correctly — a dense
  clip with only 2 tracks, both running to the clip's own last frame with no
  dropouts to explain). See `detection_quality_adapter/README.md` for full
  numbers and a locked-in regression test.

---

## Milestone 6 — Hand specialization (Part 2) ✅ DONE

**Goal:** layer hand-specific parameters and rules on top of the generic
adapter — mostly config, four behavioral changes.

- [x] `hand_config.py`: class max = 2, candidate pool = 3–4, shape rule tuned
  for roughly-equant boxes, exit-border weighting favors the bottom border.
  **Also needed a hand-specific duplicate-merge override** — see the bullet
  below on hands crossing/overlapping; the generic IoU-only dedup from
  Milestone 2 was a real risk here, not just a theoretical one.
- [x] **Selection after association** (`selection.py`): with a low,
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
- [x] **Hands crossing/overlapping vs. true duplicates needed a hand-specific
  dedup rule, not just the generic IoU threshold.** The spec's edge case
  table is explicit: "Hands cross or overlap ... Retain both. Must not be
  treated as a duplicate" — but Milestone 2's `reject_duplicates` runs
  per-frame, before any tracking exists, so it has no way to know whether
  two overlapping boxes are one object detected twice or two real hands
  that happen to overlap; it merges on IoU alone. Real-data check across
  all 39 clips found the discriminator: every real moderate-IoU duplicate
  pair had a **containment ratio** (intersection / smaller-box-area) of at
  least 0.73 — nested, one strong box mostly inside a weaker one — while a
  synthetic same-size 50%-overlap crossing (two similarly-sized real hands)
  sits at 0.5. Added `Config.duplicate_containment_threshold` (default 0.0
  = always passes, no behavior change for the generic pipeline);
  `hand_config()` sets it to 0.7. `tests/test_hand_config.py` includes a
  pair chosen right in the gap between the two gates (IoU 0.52, containment
  0.68) to prove the fix's precision, not just that low IoU alone saves it.
- [x] **Stereo depth check** — migrated out of `exp/scafholds/` into the
  real package (it was previously a validated prototype, not wired into
  anything real — see the now-resolved open question below). Adapted to
  `Detection`/`Config`/`Tag` instead of ad-hoc dicts:
  - Calibration: ZED X, 2.2mm lens (0.3m min depth), 12cm baseline — chosen
    because 4mm (1.5m min depth) can't explain the near-frame-filling hand
    boxes actually seen in the data. Cross-checked: a real high-confidence
    box (frame 230) gave disparity ~267px → ~0.33m depth, right at the 2.2mm
    lens's minimum focus distance — exactly what you'd expect for a box that
    large. See `nominal_calibration.py`'s module docstring for the full
    derivation.
  - Discovered the two eyes are **not perfectly vertically aligned** (~2.4px
    mean offset, up to ~9px, from ORB+RANSAC feature matching) — likely
    unrectified output and/or slight temporal skew between eyes on this
    moving rig. The depth-per-box matcher uses a vertical search tolerance
    rather than assuming a strict horizontal epipolar line.
  - `max_reach_m` default set to **1.8m**, not the more intuitive ~0.8m —
    empirically, 72% of `t010`'s own-hand detections exceed 1.0m because
    it's a hairstyling task with arms extended toward a seated client. Still
    a placeholder pending Milestone 7 calibration, and probably needs to
    vary by job/task type.
  - New `apply_stereo_depth_stage(tracks, video_left_path, video_right_path,
    config)` runs it across a whole clip, grouped by frame so each frame's
    video pair decodes once. `pipeline.run_hand_pipeline` calls it as an
    optional stage 6 when both video paths are given (a no-op otherwise, or
    if `config.max_reach_m` is `None`).
  - The old sandbox-path-dependent real-clip test
    (`/mnt/user-data/uploads`, which doesn't exist in this repo) is fixed to
    use the real `data/` directory, same pattern as every other real-data
    test here. `exp/scafholds/`'s now-redundant files were deleted after
    confirming nothing outside that directory imported from them.
  - **Real bug found while testing this milestone (not stereo depth
    itself)**: the batch visualization script's `--max-frames` only capped
    the rendered video, not what was fed into the pipeline -- so turning on
    stage 6 for a "quick preview" still ran video-seek + template-match
    stereo depth across the *entire* ~2700-frame clip. Took over 15 minutes
    before this was caught and fixed (cap the actual pipeline input, not
    just the render) — see `detection_quality_adapter/README.md`.
- [x] Implement the edge-case table from spec §6 as explicit test cases —
  `tests/test_hand_config.py`, one test per row: duplicate on one hand
  (merges), side exit (terminates, no interpolation), bottom exit (torso
  occlusion, terminates via the outward-motion override), brief occlusion
  (interpolated), hands crossing (retained as two, with the sanity-check
  pair proving the fix isn't just "low IoU"), long-absence re-entry (new
  track, not bridged), motion-blur confidence dip (interpolated, not lost —
  see Milestone 5's `rejected`-as-gap-material design). Two rows live
  elsewhere: "another person's hands" is `tests/test_stereo_depth*.py`
  (needs real video, not just detections); "gloved/partially-occluded hand"
  is an explicit `xfail(strict=True)` — per the spec's own instruction not
  to guess at logic there, with `strict=True` so it loudly breaks the suite
  if something accidentally starts passing before real labeled examples
  justify it.

**Prompt to give Claude:** "Implement `hand_config.py` and `selection.py` per
the notes above. Add the stereo-depth rejection rule with a stubbed depth
source. Write one test per row of the spec's edge-case table (§6), using
synthetic data — mark the gloved-hand case as `xfail`/TODO since the spec
says it needs real data."

**Exit criteria:** full hand-configured pipeline passes all edge-case tests
on synthetic data. This is the last milestone before real data is required.

**Done:** `pipeline.run_hand_pipeline(clip.detections, clip.pose,
video_left_path=..., video_right_path=...)` runs the full 5-6-stage
hand-specialized pipeline. `scripts/visualize_hand_pipeline.py` (batch by
default, same interface as `visualize_interpolation.py`) renders real clips
with a finer color breakdown than the generic visualization — specifically
so Milestone 6's two *new* rejection reasons (selection, stereo depth) are
checkable by eye, not just folded into a flat "rejected" bucket. Checked
visually on `t010` with `--stereo-depth`: the wearer's own hand stays
"kept," a detection out near the seated client (clearly beyond arm's reach)
gets flagged "rejected_stereo_depth" — the depth check is discriminating on
the right thing. 143 tests pass, 1 `xfail`ed as designed.

---

## Milestone 7 — Calibration & validation (needs the dataset) ⚠️ PARTIALLY DONE

**Goal:** replace every guessed threshold with one derived from a labeled
reference set (spec §7–8). Do this only once the dataset is downloaded.

**Honest status, stated up front: there is still no labelled reference set
for this dataset.** The dataset arrived (Milestone 1.5), but it's exactly
`hand_boxes.json` — the raw, noisy detector output this whole adapter exists
to correct — plus `meta.json`'s task-level metadata (label, job, scene,
provenance). Neither is ground truth for which boxes are correct on which
frames. Spec S7 is explicit that "no value here should be set by
inspection," so the precision/recall-optimal half of this milestone is
still genuinely blocked, not skipped by choice. What follows is what's
honestly achievable without labels, done properly rather than faked:

- [ ] Build the reference set per spec's guidance — **still blocked**, no
  labels exist. Not attempted; would need real human annotation.
- [x] `metrics.py`: precision/recall computed **per stage**, not aggregate —
  the machinery is built and correct (`StageMetrics`, greedy IoU matching,
  `flag_wrong_direction_stages` catching a stage that moves the wrong way),
  verified against hand-built synthetic ground truth in `tests/test_metrics.py`.
  **Not run against real ground truth for this project — there isn't any.**
  Every number this module could report about the real 39 clips today would
  not be a real precision/recall figure; the module's own docstring says so
  up front so this can't get misquoted later.
- [x] `sweep_thresholds.py`: derived what's honestly derivable **without**
  labels, across all 39 real clips:
  - frequency of each **rejection reason** (not confirmed false-positive
    class — that needs labels — but a real, unsupervised proxy for where
    each rule fires): kept 85.3%, rejected_temporal 5.1%, dropped_duplicate
    3.1%, rejected_size_shape 2.6%, interpolated 2.5%, rejected_selection
    1.3%, across all 153,876 raw detections (+ 2,964 stage-4 fabrications).
  - distribution of dropout lengths → directly used to retune
    `max_dropout_frames` (see `hand_config.py`'s notes below) — this is the
    one spec S7 explicitly says a real (non-labelled) distribution can set.
  - candidate-pool-width tuning ("rate at which a true hand is outranked by
    a spurious box") — **not attempted**, genuinely needs labels: there's no
    way to know which candidate was "true" without them.
- [x] Standing check: `interpolated_proportion` (spec S8: "needs no
  labelled data") — implemented in `metrics.py`, run for real across all 39
  clips in `sweep_thresholds.py`'s report: mean 3.5%, max 17.2%
  (`beb348be_t000`, checked by hand — longest interpolated runs there are
  6-12 frames, consistent with genuine brief-occlusion recovery, not
  fabrication).
- [x] Stubbed camera-motion and stereo-depth signals — **already resolved in
  earlier milestones**, not deferred to here: camera motion was real VIO
  pose from Milestone 4 onward (never actually stubbed, since the dataset
  had it from the start), and stereo depth was migrated into the real
  package in Milestone 6.

**Two real accounting bugs found and fixed while building
`sweep_thresholds.py` itself** (both in the diagnostic tool, not the
pipeline): `rejection_reason_frequency` first only walked `tracks`, silently
dropping every stage-1 casualty (dedup losers, size/shape rejects never
reach `track_detections`) — undercounted the real dataset by 4.1% with no
error. The fix then walked raw per-frame lists instead, which silently
dropped stage-4's brand-new fabricated `interpolated` detections in the
other direction (they never existed as raw input). Caught by an independent
total-count assertion in `tests/test_sweep_thresholds.py`
(`test_rejection_reason_frequency_total_matches_raw_plus_fabricated`) before
either version shipped. See `calibration/sweep_thresholds.py`'s docstring
for the full account — a good example of why "the numbers added up" isn't
sufficient proof of correctness on its own; the first buggy version's total
also looked internally consistent.

**Tuning applied to `hand_config()` from this real-data analysis** (see its
own inline comments for the numbers): `plausible_size` (20,800) → (50,1150)
— the old lower bound was a complete no-op (0% of 153,876 raw detections
anywhere near 20px) and the old upper bound incorrectly rejected ~0.16% of
raw detections that are far more likely genuine close-up hands than noise,
given this dataset's own documented near-frame-filling characteristic.
`max_dropout_frames` 10 → 15 — the old value sat just under the real
gap-length distribution's p75 (13 frames), so over a quarter of genuine
short recoverable gaps were already too long to interpolate; the long tail
beyond that (p90=54, worst case 2529 frames) isn't trustworthy as "the same
object" since the position gate has grown too wide by then to mean anything,
so 15 stays well clear of it. Both are still informal, distribution-based
calibration, not precision/recall-optimal — flagged as such, not oversold.

**Final run**: the tuned hand pipeline (stages 1-5) ran against all 39
clips (`scripts/visualize_hand_pipeline.py --all`), 996.8s, one annotated
video per clip saved to `detection_quality_adapter/scripts/output/
final_all_clips/` (gitignored, ~8.3GB, local only). Aggregate: 148,714 final
detections across 787 tracks, 90.5% kept, 5.5% rejected at stage 3, 2.6%
interpolated, 1.4% rejected at stage 5, 36.1% of tracks confirmed exiting —
matches `sweep_thresholds.py`'s independently-computed numbers exactly.
Stage 6 (stereo depth) omitted from this run: at the per-clip rate observed
in Milestone 6, the full dataset would take several hours rather than
~15-20 minutes; available via `--stereo-depth` on a smaller selection. One
limitation found reviewing the run: this script's stats/video never show a
`rejected_geometric` case (same root cause as the `sweep_thresholds.py`
accounting bug above — stage-1 casualties never reach `track_detections`,
so a track-only walk never sees them) — doesn't affect the pipeline's actual
output, only this one script's ability to show *why* a stage-1 rejection
happened; `visualize_stage1.py` already covers that view correctly. Not
fixed-and-re-rendered for this pass (would cost another ~17 min + 8GB for a
visualization completeness gap, not a behavior change) — flagged here
instead so it's a tracked, known thing rather than a silent gap.

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
7. ~~Session 7: Milestone 5 (interpolation/exit) + wire `pipeline.py` end to
   end~~ ✅ done, informed by a full spec re-read in between (see the
   Milestone 5/6 design notes and Open Questions added then) — real-data
   testing caught a real bug (exit test used box center, not edge) and a
   test-writing bug (a tautological prediction check), both documented in
   Milestone 5's "Done" notes above and the README.
8. ~~Session 8: Milestone 6 (hand specialization + edge-case tests)~~ ✅
   done — `hand_config.py`, `selection.py`, the containment-ratio dedup fix,
   and the stereo-depth migration out of `exp/scafholds/`, plus one edge-case
   test per spec §6 row (gloved-hand deliberately `xfail`ed).
   `scripts/visualize_hand_pipeline.py` added, same batch-by-default
   interface as `visualize_interpolation.py`.
9. ~~Session 9: Milestone 7 (calibration/validation)~~ ⚠️ partially done —
   `metrics.py` and `sweep_thresholds.py` built, and everything honestly
   achievable without labels was: rejection-reason frequency, dropout-length
   distribution, the interpolated-proportion standing check, all across all
   39 real clips, plus retuning `plausible_size` and `max_dropout_frames`
   from that data. The precision/recall-optimal half (a labelled reference
   set, candidate-pool-width tuning) is still genuinely blocked — see
   Milestone 7's "Honest status" note above. `plausible_size`/
   `plausible_shape`/`max_dropout_frames` (flagged as unexamined after
   Milestone 4's checkpoint) are now all done; `plausible_shape` had already
   been covered incidentally during Milestone 6's hand_config tuning.

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
- ~~`plausible_size`, `plausible_shape`, and `max_dropout_frames` haven't had
  a real-data sanity pass yet~~ → done at Milestone 7: `plausible_size`
  (20,800)→(50,1150) and `max_dropout_frames` 10→15 in `hand_config()`, both
  from real distributional data across all 39 clips; `plausible_shape` had
  already been covered during Milestone 6. See Milestone 7's notes above
  for the numbers.

Found during a full spec re-read at this checkpoint (spec text supplied in
full for the first time — prior sessions worked from summaries in this
document; see the Milestone 5/6 sections above for the full detail behind
each):
- ~~Milestone 2's duplicate-merge rule risks wrongly merging two real,
  overlapping hands~~ → done: `Config.duplicate_containment_threshold`
  (0.7 for hands) gates dedup on containment ratio in addition to IoU — see
  Milestone 6's notes above for the real-data numbers behind 0.7.
- ~~Interpolation (Milestone 5) needs to treat a track's `rejected`
  detections as gap-equivalent~~ → done: built into Milestone 5 from the
  start (see its "Done" notes above), not retrofitted after the fact.
- ~~The bottom-border exit test probably needs its own trigger condition~~ →
  done: `Config.exit_border_margin_overrides`/`exit_requires_outward_motion`,
  set by `hand_config()` for the bottom border specifically.
- ~~Milestone 6's stereo-depth "done" status was overstated~~ → done: fully
  migrated into `main/detection_quality_adapter/adapter/`, wired into
  `pipeline.run_hand_pipeline` as an optional stage 6, old `exp/scafholds/`
  files deleted.

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
