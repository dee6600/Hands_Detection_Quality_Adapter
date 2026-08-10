# Detection Quality Adapter — Interview Reference

*A post-processing layer that turns a noisy hand detector into a trustworthy signal — without ever touching the detector itself.*

**Dataset:** 39 egocentric work clips, 85,786 frames, 153,876 raw detections
**Stack:** Python, OpenCV, NumPy
**Tests:** 160 passing, 1 deliberate xfail

## Contents

1. [The problem, plainly](#1-the-problem-plainly)
2. [Architecture: the six stages](#2-architecture-the-six-stages)
3. [Key design decisions](#3-key-design-decisions)
4. [The data](#4-the-data)
5. [Results & metrics](#5-results--metrics)
6. [Limitations](#6-limitations)
7. [What went well](#7-what-went-well)
8. [Core interview questions](#8-core-interview-questions)
9. [Tricky questions](#9-tricky-questions)
10. [Glossary](#10-glossary)

---

## 1. The problem, plainly

A hand detector looking at video makes two kinds of mistakes. It reports a hand where there isn't one — a **false positive** (a duplicate box, a bystander's hand, a shadow it mistook for skin). Or it misses a hand that's clearly there — a **false negative** (a frame where the hand briefly blurs, or the model just has a bad frame).

Neither mistake is visible from a single frame. A duplicate box looks fine in isolation. A missed hand looks like the hand simply isn't there. The only way to catch either is to look across *many* frames — a real hand moves along a continuous, physically plausible path; noise and gaps don't.

The obvious fix — retrain the detector — is exactly what this project *doesn't* do. Retraining is slow, expensive, and doesn't fix the structural issue: a detector scoring boxes one frame at a time has no way to know that a duplicate should lose to the real hand, because in that one frame, it doesn't know which is which. So instead: let the detector over-report (emit every plausible box, no cap, no dedup), and put a stateful, multi-frame layer *after* it that has the context a single frame never will.

> **Why this matters.** This reframes the whole project as a **tracking and consistency** problem, not a computer-vision problem. Nothing here does object detection. Everything here asks: *given these noisy boxes over time, what's the most plausible explanation?*

---

## 2. Architecture: the six stages

Detections flow through six stages, always in this order. Order is load-bearing: stages 3–6 need *tracks*, which don't exist until stage 2 has run, and stage 2 needs geometrically-sane boxes, which don't exist until stage 1 has run.

```
Stage 1          Stage 2          Stage 3          Stage 4              Stage 5           Stage 6
Geometric   →    Association  →   Temporal    →    Interpolation   →    Selection    →    Stereo depth
rejection        (tracker)        rejection        / exit               (hand-only)       (hand-only, opt-in)
```

| Stage | Rule | Basis |
|---|---|---|
| 1 | Duplicate detections | Two heavily-overlapping boxes on one object → merge, keep the higher-confidence one |
| 1 | Implausible size | Box far larger/smaller than a hand can be at working distance |
| 1 | Implausible shape | Box markedly more elongated than a hand should be |
| 2 | Association | Greedy nearest-neighbor match against each track's velocity-predicted position |
| 3 | Implausible displacement | Box jumps further between frames than a hand can travel |
| 3 | Unsupported (flicker) | Appears for 1–2 frames with no track before or after |
| 3 | Static detection | Doesn't move while the camera does → probably background |
| 4 | Exit test, then gap-fill | Near a border with outward motion → let it end. Otherwise, resumes near prediction → interpolate |
| 5 | Instance-cap selection | More than 2 live candidates on a frame → keep the 2 best-supported *tracks*, not the 2 highest-confidence *boxes* |
| 6 | Beyond arm's reach | Per-box stereo depth → reject if further than the wearer could reach |

Every detection carries a `tag`: `reported`, `merged`, `rejected`, or `interpolated`. Nothing is ever deleted — a rejected box stays in the record, just marked untrustworthy. This matters for two reasons: it makes every stage's decision auditable after the fact, and it lets a *later* stage change its mind about an *earlier* tag (interpolation can overwrite a rejected box's position if the surrounding trajectory supports it).

---

## 3. Key design decisions

### Tracking algorithm — greedy nearest-neighbor, not Kalman or Hungarian

The tracker predicts each track's next position by linear extrapolation from its last known velocity, then matches the closest detection within a distance gate. This is deliberately the simplest thing that could work, with the upgrade path (Kalman filter for noisy velocity, Hungarian algorithm for globally-optimal assignment instead of greedy) flagged in code comments but not built, because the simple version passed every synthetic and real-data test thrown at it.

**Why:** premature sophistication is a cost, not a feature. Build the upgrade when a real failure demands it, not speculatively.

### Identity signal — position and motion only, never the detector's left/right label

The detector guesses "left hand" / "right hand" per box, but that label is unreliable exactly when it matters most: when two hands cross. The tracker is built to ignore it completely for identity decisions, and this is explicitly tested with an adversarial case — two hands crossing with their labels swapped right at the crossing point.

**Why:** an unreliable signal is worse than no signal if you don't know it's unreliable.

### Two gate radii, not one

Early on, one `max_speed` value governed both the tracker's match gate *and* stage 3's displacement-plausibility check. Real data showed why that's wrong: a genuinely fast, real hand movement needs a *generous* gate so the tracker doesn't fragment the track over it — but that same fast movement should still get flagged as visually untrustworthy once it's part of a track. One number couldn't do both jobs at once.

**Why:** conflating "should these two things be considered the same object" with "is this specific frame trustworthy" is a common tracking-system bug. They're different questions on different timescales.

### Hand specialization — a generic pipeline plus a thin, config-only layer

Part 1 of the system (stages 1–4) knows nothing about hands specifically — it's written for a generic tracked object with configurable size/shape/speed expectations. `hand_config()` layers hand-specific numbers and two extra stages (selection, stereo depth) on top, without any stage 1–4 code needing to know a hand from any other object.

**Why:** this is what makes the system testable in isolation and reusable.

### Calibration philosophy — real-data checks over guesses, labels over both, but don't wait for labels to catch the obvious

Every placeholder threshold got checked against the real dataset's own distributions before being trusted, using percentiles and structural invariants rather than inspection. A threshold that turned out to be a complete no-op or actively wrong got fixed immediately, without waiting for labeled ground truth. Anything that genuinely needed labels — true precision/recall, candidate-pool tuning — was left alone and documented as blocked.

**Why:** "we don't have labels" is not the same as "we can't check anything."

---

## 4. The data

39 clips, each a different real work task, recorded on a head-mounted stereo camera (1920×1200, 30fps, both eyes). Every clip bundle carries five things: the stereo video pair, raw hand boxes from a fine-tuned detector (uncapped, no dedup, no smoothing — deliberately noisy), full-rate 6DoF camera pose from onboard visual-inertial odometry, per-frame timestamps, and task metadata.

> **A fact that shaped a lot of decisions.** Every one of the 39 clips is a *different job* — hairstylist, carpenter, farmworker, jeweler, mason, mechanic, and so on, no repeats. That single fact is why several "one global number" thresholds turned out to be too simple: a farmworker's typical working distance (median 0.77m) and a hairstylist's (1.5–1.8m, arms extended toward a seated client) are genuinely different populations, not noise around one true value.

| Quantity | Value |
|---|---|
| Clips | 39 |
| Total frames | 85,786 |
| Total raw detections | 153,876 |
| Frames with exactly 2 boxes | 72.3% |
| Frames with 1 box | 18.2% |
| Frames with 0 boxes | 4.3% |
| Frames with 3+ boxes (bystanders, duplicates) | 5.25% |
| Ground-truth labels available | none |

That last row matters more than it looks — see [Results](#5-results--metrics) and [Limitations](#6-limitations).

---

## 5. Results & metrics

The spec's own validation section asks for two things: **precision and recall computed per stage** (not in aggregate, because each stage should move a specific direction), and a labels-free **standing check** on the proportion of interpolated detections. Here's what was actually achievable, honestly, without ground truth.

### What real precision/recall would need

Precision and recall are both defined against ground truth — "was this box actually a hand." This dataset has none. The matching machinery to compute them (greedy IoU matching against a reference set, flagged if any stage moves the wrong direction) is fully built and unit-tested against synthetic ground truth, ready to run the moment a labeled reference set exists. No number it could produce today would be a real precision/recall figure.

### What was computed instead, for real, across all 39 clips

| Metric | Value | What it stands in for |
|---|---|---|
| Final detections (stages 1–5) | 148,714 | — |
| Kept | 90.5% | Survived every rule |
| Rejected — temporal (stage 3) | 5.5% | Proxy for false-positive frequency |
| Interpolated (stage 4) | 2.6% | Recovered false negatives |
| Rejected — selection (stage 5) | 1.4% | Cap enforcement, ranked by track quality |
| Dropped as duplicate (stage 1)* | 3.1% | Proxy for one false-positive class |
| Rejected — size/shape (stage 1)* | 2.6% | Proxy for another false-positive class |
| Tracks confirmed exiting | 36.1% | Border-exit correctly not fabricated |
| Interpolated proportion, mean / max across clips | 3.5% / 17.2% | The spec's own labels-free standing check |

*measured against all 153,876 raw detections, not just the 148,714 that survive to a track*

These are **rejection-reason frequencies**, not confirmed false-positive rates — they say where each rule fires, not whether firing was correct. That distinction was kept explicit everywhere it's reported.

### What was tuned from this, without labels

| Parameter | Before | After | Evidence |
|---|---|---|---|
| `plausible_size` | (20, 800)px | (50, 1150)px | Real side-length p1=94px, max=1110px — old floor was a no-op, old ceiling rejected real close-up hands |
| `plausible_shape` | (0.3, 3.0) | (0.5, 2.0) | Real aspect-ratio p1=0.54, p99=2.10 |
| `max_dropout_frames` | 10 | 15 | Real gap-length p75=13 frames |
| `max_speed` (plausibility) | 150 px/frame | 110 px/frame | Real adjacent-frame speed p99.5=97.7 |
| `track_gate_speed` (matching) | shared with above | 350 px/frame | Split out; above the fastest real jump ever observed (305.8) |
| `duplicate_containment_threshold` | n/a | 0.7 (hands) | Real duplicate pairs cluster ≥0.73; crossing hands sit ~0.5 |

---

## 6. Limitations

- **No ground truth exists for this dataset.** The single biggest constraint. Every threshold above is evidence-based, not precision/recall-optimal — a genuinely different, weaker guarantee than what the spec's validation section asks for.
- **Stereo depth doesn't generalize past one clip.** The camera calibration behind the "beyond arm's reach" rule was derived from and validated against exactly one clip (`t010`). Run across all 39, it confidently misjudged a real hand's depth by roughly 4× on a different clip (`42bba884_t012`) — traced to the dataset spanning multiple recording batches, likely different physical camera rigs, with no per-clip calibration metadata to correct for it. This stage is opt-in and excluded from the default/final output for exactly this reason.
- **Static-background detection fails on multi-second braces.** A hand deliberately held still for a few frames is correctly protected. A hand braced motionless for several *seconds* still reads as background, because the underlying signal — "did the box move less than the camera" — can't structurally tell a long real pause from true static clutter without also modeling camera rotation vs. translation.
- **Some thresholds are one global number for 39 different jobs.** Arm's reach, camera speed, and hand size all vary by task in ways confirmed in the data. A job-aware version was considered and deliberately not built, because deriving a job's "normal" range from its own detections risks being circular without labels to check it against.
- **The tracker is greedy, not globally optimal.** Nearest-neighbor matching can mis-assign in dense, ambiguous scenes. No real or synthetic test found this actually happening, but it's a known theoretical gap.

---

## 7. What went well

- **Testing caught real bugs before they shipped, repeatedly.** A tracker patience bug (a stale track could be resurrected via an ever-widening search gate), an exit-detection bug (measuring from box center instead of edge, which produced *zero* detected exits on real near-frame-filling boxes), and a diagnostic tool that silently mis-counted 4% of the dataset in one direction and then the other — all caught by tests or hand-verified math before being trusted.
- **Nothing is ever silently discarded.** The reported/merged/rejected/interpolated tag model means every decision is inspectable after the fact, and reversible where a later stage has better information.
- **Every stage is independently testable.** Synthetic fixtures cover every rule with no dataset required, plus a real-data pass on top for every stage — 160 tests total, run in about 10 seconds.
- **Visual verification, not just assertions.** A family of batch visualization scripts render real clips with every box labeled by exactly what happened to it and why — which is precisely what caught the stereo-depth calibration problem.
- **Honest about what "done" means.** Every threshold's provenance (guessed placeholder vs. real-data-checked vs. label-validated) is documented and kept current.

---

## 8. Core interview questions

**Q: Why not just improve the detector instead of building all this?**
Because the failure modes aren't fixable per-frame. A duplicate box and a real box can look identical in isolation — the only signal that tells them apart is which one belongs to a continuous trajectory over time, and a single-frame detector structurally can't see that. Retraining also can't fix "the detector correctly saw a bystander's hand" — that's not a detection error at all, it's a business-logic decision (is this hand in reach) that has nothing to do with detection quality.

**Q: Why does the stage order matter so much?**
Stages 3 onward all reason about *tracks*, which don't exist until stage 2 (association) has run. And stage 2 needs geometrically sane input — if you tried to track before removing duplicates, you'd be trying to assign two overlapping boxes on one hand to two different tracks, which is nonsensical.

**Q: Walk me through what happens when a hand goes behind an object for a few frames.**
The detector reports nothing for those frames. The tracker keeps the track alive for up to `max_dropout_frames`, "coasting" on its last known velocity. When a detection resumes, stage 4 checks two things: is the resumption near where the coasting prediction said it should be, and was the last real sighting near a frame border with outward motion (meaning the hand actually left, not just got occluded)? If it resumes near the prediction and wasn't an exit, the missing frames get filled in with linearly-interpolated boxes tagged `interpolated`.

**Q: How do you stop interpolation from just inventing hands that left the frame?**
The exit test runs *before* the gap-fill decision, every time, and it's a hard veto — if the last sighting was near a border moving outward, the gap is never filled, no matter how well a later detection might line up with the prediction.

**Q: Why greedy nearest-neighbor instead of a Kalman filter or Hungarian matching?**
Because it's the simplest thing that could work, and it did — every synthetic test and every real-clip check passed without needing either upgrade. A Kalman filter buys a smoothed velocity estimate under sensor noise; this system's noise shows up as missing/duplicate boxes, which stages 1 and 3 already handle. Hungarian matching buys globally optimal assignment instead of greedy — useful when greedy would pick a locally-good but globally-wrong pairing, which never showed up in testing.

**Q: The detector guesses left/right hand per box. Why doesn't the tracker use that?**
Because it's exactly unreliable when it matters most — when two hands cross, the detector's left/right guess frequently flips. Using it would mean the tracker inherits that instability. This is tested directly: two hands cross with their labels deliberately swapped exactly at the crossing frame, and the tracker still keeps both identities straight because it never looks at the label at all.

**Q: What's the difference between the generic pipeline and the hand-specialized one?**
The generic pipeline (stages 1–4) is written for any trackable object — no hand-specific logic anywhere. `hand_config()` supplies hand-specific numbers and the pipeline adds two more stages on top: selection and stereo depth. Nothing in stages 1–4 changes to support this — it's pure configuration plus two additive stages.

**Q: You had no labeled data. How did you calibrate anything?**
By distinguishing what genuinely needs labels from what doesn't. True precision/recall needs ground truth, full stop — left honestly blocked. But whether a threshold is even in a sane range relative to what the detector actually emits doesn't need labels — it needs the real distribution. A size floor that 0% of 150,000+ real boxes ever get close to is a bug regardless of which boxes are "correct."

**Q: Why is the static-detection rule unusually important for a head-mounted camera?**
Because the camera never stops moving, so anything that stays visually fixed in the frame while the camera moves is almost certainly scene structure rather than a hand. In practice it turned out to be the trickiest rule in the whole system, precisely because "the camera is moving" is true almost continuously, and a naive version fired on real hands pausing for a single frame.

---

## 9. Tricky questions

**Q: Your static rule still misses multi-second braces. Why not just make the "sustained run" window longer?**
Because that trades one failure for its opposite. The shipped fix requires many consecutive still-while-moving frames before rejecting anything — long enough to stop punishing a brief real pause, but short enough to still catch true background. Stretching the window further would also let genuinely static clutter survive for that same window before being caught. The real fix isn't a bigger number — it's distinguishing rotation-induced apparent motion from real parallax, which needs a geometric model this system doesn't have yet.

**Q: You derived `max_dropout_frames` from a distribution with a tail out to 2,529 frames but picked 15. Isn't that arbitrary?**
The tail is the tell, not the target. At 2,529 frames of separation, the tracker's own position gate has grown so wide it's no longer meaningfully restrictive — that's the signature of coincidental position matches between two *different* objects, not one object's genuine dropout. The reliable part of the distribution is the front — p75 ≈ 13 frames, where three-quarters of real, short, recoverable gaps live. 15 sits just past that, deliberately not chasing the contaminated tail.

**Q: You found a stereo-depth bug that confidently misjudged a real hand's distance. Walk me through how you caught it, and why testing didn't catch it first.**
The synthetic tests couldn't have caught it — they test the matching *mechanism*, not whether a real camera's calibration constants are right for a real clip. The bug surfaced when a full-dataset run was reviewed by eye: a specific clip's rejection rate (44.6%) was implausibly high, so I pulled the actual frame, saw the wearer's own hands sitting on a nearby desk, and traced the wrong depth back to a stereo match that was confident (0.9+ correlation) but physically implausible. Cross-referencing clip metadata confirmed the dataset spans multiple recording batches, likely different physical rigs. The lesson: a synthetic test proves the algorithm is implemented correctly; it says nothing about whether its real-world assumptions hold.

**Q: Selection ranks candidates by track length. Doesn't that unfairly penalize a real hand that only briefly enters frame?**
It would, if selection and flicker-rejection were the same rule — but they're deliberately not. A track has to survive stage 3's unsupported-track check first, which only removes tracks of 1–2 detections with no support before or after. Anything that clears that bar is already a real, if short, trajectory. Selection only matters when *more* real candidates are competing for the cap than allowed, and in that contention, favoring the better-established track over a just-arrived one is the correct call.

**Q: If two clips were recorded on physically different cameras, how would that break your stereo math, and how would you catch it automatically instead of by hand?**
The depth formula is `Z = focal_length × baseline / disparity`. Both are physical constants of one specific camera unit; a different lens or eye separation turns a correctly-measured disparity into a wrong depth — exactly what happened here. Catching it automatically without labels is genuinely hard: the best available proxy would be a consistency check — does a clip's whole depth distribution read as systematically implausible, flagging that clip's calibration as suspect rather than trusting it silently. That's an anomaly detector, not a fix.

**Q: Your calibration is drawn from 39 clips. How do you know it generalizes, and what would change your mind?**
I don't know that it fully generalizes, and I'd say so directly. What I do have is stronger than a guess: every threshold was checked against the full real distribution across all 39 clips, not fit to one or two — precisely because the original stereo-depth calibration was a single-clip check that didn't generalize even one clip over. What would change my mind: a labeled reference set showing a threshold is systematically wrong, or a new batch of clips whose distributions don't match the ones already seen.

**Q: Why build a custom tracker instead of using an off-the-shelf one like ByteTrack or DeepSORT?**
Those trackers are built for a different problem — many similar-looking objects across a wide frame, usually with an appearance embedding for re-identification. This problem has at most two objects, of a known class, with position and velocity as the only trustworthy signal (confidence isn't; the handedness label isn't). A general-purpose tracker brings machinery this problem doesn't need. The custom tracker is roughly 150 lines and every assumption is visible and tested — it doesn't handle the general case, but it doesn't need to.

**Q: How would a bystander deliberately close to the wearer break your arm's-reach rule?**
It would — the rule is purely geometric (distance), so a bystander's hand genuinely within the wearer's arm's reach is indistinguishable from the wearer's own hand by this signal alone. The spec is explicit that position and entry direction alone can't separate them either, which is why depth is used at all — but depth only encodes "how far," not "whose." A more robust version would need an orthogonal signal (hand appearance consistency, or motion correlated with the wearer's own arm kinematics from pose data) that this project doesn't attempt to build.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| Detection | One raw box the detector reported on one frame, with a confidence score |
| Track | An ordered sequence of detections believed to be the same physical object over time |
| Tag | What happened to a detection: `reported`, `merged`, `rejected`, or `interpolated`. Never deleted, always tagged |
| IoU | Intersection-over-Union — how much two boxes overlap, from 0 (none) to 1 (identical) |
| Disparity | How many pixels the same point shifts between the left and right stereo views — closer objects shift more |
| VIO pose | Visual-inertial odometry — the camera's own estimated 6DoF position/orientation per frame, used as the "is the camera moving" signal |
| Gate / gating | A maximum-distance cutoff used to decide whether a detection is even a plausible candidate match for a track |
| Standing check | A monitor needing no ground truth — here, the proportion of interpolated detections, which rising over time would signal fabrication rather than recovery |

---

*Numbers reflect the project's real, final calibration run (stages 1–5, all 39 clips) — stage 6 is documented as opt-in and excluded from the default result for the reasons in §6.*
