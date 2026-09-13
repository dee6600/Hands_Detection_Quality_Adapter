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

**Status:** not started
**Script(s):**
**Results file(s):**

**Method:**

**Key results:**

**What this proves / does not prove:**

**Caveats:**

---

## 2. Track Quality

**Status:** not started
**Script(s):**
**Results file(s):**

**Method:**

**Key results:**

**What this proves / does not prove:**

**Caveats:**

---

## 3. Ablation & Sensitivity

**Status:** not started
**Script(s):**
**Results file(s):**

**Method:**

**Key results:**

**What this proves / does not prove:**

**Caveats:**

---

## Open items for the paper (fill in once all three land)

- [ ] Merge all three eval branches back into `paper`
- [ ] Consolidate duplicated clip-iteration helper code across the three scripts
- [ ] Pull final numbers into the paper's Results section
- [ ] Update Limitations section to reflect what label-free eval can't establish
