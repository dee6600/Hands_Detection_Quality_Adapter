# Detection Quality Adapter — Project Plan

Source spec: `hand_detection_spec.pdf` (Detection Quality Adapter — Part 1 generic, Part 2 hand-specialized)

Goal: build a post-processing stage that sits between a hand detector and downstream
consumers, correcting false positives and false negatives using temporal (multi-frame)
logic, without touching or retraining the detector itself.

This plan is written so each milestone can be handed to Claude as a self-contained
prompt. Work top to bottom — later stages depend on data structures and outputs from
earlier ones. Milestones 1–5 need NO dataset; they can be fully built and unit-tested
on synthetic data. Milestone 6 is hand-specific. Milestone 7 needs real labeled data.

## Dataset (arrived — 39 clips, one bundle inspected: `0c54a47b_t010`)

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

## Milestone 0 — Project scaffolding

**Goal:** repo structure, environment, no logic yet.

- [ ] Create project structure:
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
- [ ] Set up virtual env, `numpy`, `pytest`, optionally `scipy` (Hungarian matching) and
  `filterpy` (Kalman filter) as candidates for Milestone 3.
- [ ] Decide language/runtime version and commit an empty pipeline that just passes
  detections through untouched, with a passing smoke test.

**Prompt to give Claude:** "Scaffold the repo structure above, empty modules, one
smoke test that imports everything and runs a no-op pipeline call."

---

## Milestone 1 — Data model & contracts

**Goal:** lock down the shapes everything else builds on.

- [ ] `Detection`: box (x, y, w, h or corners), confidence, frame_id/timestamp,
  class label, source flag (`reported` at input).
- [ ] `Track`: id, ordered list of Detections, state (active / ended / exiting),
  predicted next position (for association + exit test).
- [ ] Output tag enum: `reported | merged | rejected | interpolated`.
- [ ] `Config`: per-class expectations — plausible size range, plausible shape
  (aspect ratio range), candidate pool size, class max instances, max speed
  (px/frame or px/sec), exit-border weighting.
- [ ] Write this as plain dataclasses/structs first — no behavior, just shape —
  so every later stage can be unit tested against hand-built fixtures.

**Prompt to give Claude:** "Implement `types.py` with these dataclasses. Add a
`Config` with sensible placeholder defaults for a generic object, not hands yet."

**Exit criteria:** you can hand-construct a list of `Detection`s in a test file
and pass them through a no-op pipeline that returns them unchanged but tagged
`reported`.

---

## Milestone 1.5 — Data ingestion (real bundle → internal types)

**Goal:** one loader that turns a clip folder into the objects Milestones 2–6
operate on. This is new now that the dataset has actually arrived — do it
right after Milestone 1, in parallel with (not instead of) the synthetic-data
work in Milestones 2–5.

- [ ] `load_clip(clip_dir) -> ClipData`, where `ClipData` bundles:
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
- [ ] Map `hand_boxes.json`'s `class 0/1` field into `Detection`'s handedness
  field but **do not** treat it as a stable identity — every downstream stage
  must already ignore it per the spec, this is just carrying it through for
  debugging/visualization.
- [ ] Write one integration test that loads the real `t010` bundle end-to-end
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

---

## Milestone 2 — Geometric rejection (Stage 1, per-frame only)

**Goal:** the three per-frame rules, in fixed order, no tracking required yet.

Order matters (spec §3):
1. **Duplicate detections** — heavily overlapping boxes on one object → merge,
   keep the stronger, tag `merged`.
2. **Implausible size** — reject boxes far outside plausible size at working
   distance.
3. **Implausible shape** — reject boxes markedly more elongated than the
   object should be.

- [ ] IoU-based duplicate merge function, testable with two overlapping synthetic
  boxes → confirm one `merged` output survives.
- [ ] Size filter against `Config.plausible_size`.
- [ ] Shape filter against `Config.plausible_shape` (aspect ratio bounds).
- [ ] Unit tests: one synthetic frame per rule, plus one frame that passes clean.

**Prompt to give Claude:** "Implement `geometric.py` with `reject_duplicates`,
`reject_implausible_size`, `reject_implausible_shape`, applied in that order.
Write pytest cases for each using hand-built boxes — no dataset needed."

**Exit criteria:** stage 1 runs standalone on a list of per-frame candidate
boxes and returns a filtered/merged/tagged list. Fully testable now.

---

## Milestone 3 — Association (Stage 2, the tracker)

**Goal:** turn per-frame detections into per-object tracks. This is the piece
everything downstream depends on — worth the most care.

- [ ] Decide matching strategy: start simple — greedy nearest-neighbor using
  predicted position (linear extrapolation from last known velocity) — before
  reaching for a Kalman filter or Hungarian algorithm. Upgrade only if the
  simple version isn't good enough on synthetic sequences.
- [ ] A detection close to a predicted position extends that track.
- [ ] A detection far from any active track starts a new track.
- [ ] Track object needs: history, current predicted position/velocity,
  "active" vs "no detection this frame but still within patience window" state.
- [ ] Unit tests: synthetic multi-frame sequences —
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

---

## Milestone 4 — Temporal rejection (Stage 3, needs tracks)

**Goal:** the three rules that only make sense once tracks exist (spec §3,
rules 4–6).

- [ ] **Implausible displacement** — box jumps further between consecutive
  frames than `Config.max_speed` allows → reject.
- [ ] **Unsupported detection** — appears for 1–2 frames with no track before
  or after → reject as flicker.
- [ ] **Static detection** — doesn't move while the camera does → reject as
  background structure. Camera motion is now a real signal, not a stub: use
  `vio_pose.json` (via `ingest.py`'s `ClipData.pose`) — frame-to-frame delta in
  position and/or yaw/pitch/roll gives "is the camera moving" directly, no
  optical flow needed. Define a motion threshold (e.g. speed or angular rate
  above some epsilon) rather than a boolean stub.
- [ ] Unit tests: synthetic tracks — one flicker-only track, one track
  exceeding max speed, one static track against a "moving camera" stub, one
  clean track that should survive all three.

**Prompt to give Claude:** "Implement `temporal.py`'s three rejection rules
operating on Track objects from Milestone 3. Stub camera motion as a boolean
per frame for now. Write synthetic test tracks for each rejection rule plus
one clean pass-through case."

**Exit criteria:** tracks in → tracks with implausible/unsupported/static
detections tagged `rejected`, rest untouched.

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
- [ ] **Selection after association** (`selection.py`): with a low,
  constantly-reached cap, implement "rank remaining candidates by track
  quality, select top 2" rather than per-frame confidence.
- [ ] **Static-detection rule** is unusually strong here (head-mounted camera
  is always moving) — no code change, just confirm the stub camera-motion
  signal from Milestone 4 gets replaced with a real one before this matters.
- [ ] **Handedness**: tracker must never use the left/right label to decide
  association — confirm Milestone 3's tracker only uses position/motion
  (it should already, but add an explicit test with hands crossing and
  swapped labels).
- [x] **Stereo depth check** — DONE (nominal calibration tier). Implemented
  in `adapter/nominal_calibration.py` + `adapter/stereo_depth.py`, tested
  against synthetic cases and validated on the real `t010` clip:
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
    `detection_quality_adapter/tests/` and run the demo with
    `python scripts/demo_stereo_depth.py --hand-boxes ... --left ... --right ...`.
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

1. Session 1: Milestones 0–1 (scaffold + types)
2. Session 2: Milestone 1.5 (real data ingestion against `t010`) — do this
   early now that data is in hand, so every later milestone can be sanity
   checked against a real clip alongside its synthetic tests
3. Session 3: Milestone 2 (geometric rejection) — synthetic tests first, then
   a quick manual look at stage-1 output on `t010`'s 3+ box frames
4. Session 4: Milestone 3 (association/tracker) — the hardest, give it room;
   `t010`'s handedness-swap-on-crossing risk is a good real test case once
   synthetic cases pass
5. Session 5: Milestone 4 (temporal rejection) — wire real VIO-based camera
   motion in immediately, no stub needed
6. Session 6: Milestone 5 (interpolation/exit) + wire `pipeline.py` end to end
7. Session 7: Milestone 6 (hand specialization + edge-case tests) — stereo
   depth rule is DONE against nominal calibration (see above); remaining
   Milestone 6 work is `hand_config.py`, `selection.py`, and the edge-case
   table tests, which still need Milestones 0-5's types/pipeline in place
   first (selection needs association output, etc.)
8. Once labels exist for a reference set: Milestone 7 (calibration/validation)

Each session should end with passing tests before moving on. Synthetic tests
remain the primary correctness check through Milestone 6; the real `t010`
clip is a secondary sanity pass, not a substitute, until labels exist.

## Open questions

Resolved by the dataset arrival:
- ~~Detector output format~~ → xyxy pixels, 1920x1200, conf ≥ 0.10, class 0/1
  = detector's left/right guess (unreliable, ignore for tracking identity),
  no cap/dedup/forward-fill. Confirmed in `hand_boxes.json`.
- ~~Camera-motion source~~ → real 6DoF VIO pose per frame (`vio_pose.json`),
  not IMU, not optical flow. No per-clip IMU exists in this bundle set at all.
- ~~Batch vs streaming~~ → offline, per-clip, non-causal. Full clip files are
  delivered up front.

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
- `frame_ts.json` (`frame_count`) and `vio_pose.json` (`n`) differ by one
  sample in the inspected clip (2700 vs 2701) — is this a consistent off-by-one
  across all 39 clips (e.g. pose sampled at both endpoints of the window) or
  clip-specific? Worth checking a second bundle once available, so
  `ingest.py` handles it generally rather than special-casing `t010`.
- Max-speed, plausible-size/shape, and dropout-length thresholds (Milestone 7)
  still need the labeled reference set — raw detections alone don't have
  ground truth, so calibration can't start until labels exist, even though
  the raw clips themselves are now in hand.
- Should the adapter run per-clip independently (39 separate runs) or is there
  ever continuity across clips within a session worth preserving? Currently
  assuming per-clip independence, matching how the bundles are delivered.
