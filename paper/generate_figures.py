#!/usr/bin/env python3
"""Generates the paper's figures from the committed calibration result files.

Run from the repo root: `python paper/generate_figures.py`
Reads calibration/{ablation,dropout_recovery}_results.json (already committed,
reproducible via `python calibration/<script>.py --data-dir data`).
Writes vector PDFs to paper/figures/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def _load(name):
    with open(os.path.join(ROOT, "calibration", name)) as f:
        return json.load(f)


def fig1_rejection_breakdown():
    ablation = _load("ablation_results.json")
    c = ablation["ablation"]["full_pipeline"]["counts"]
    total = ablation["ablation"]["full_pipeline"]["total"]

    cats = {
        "Kept": c["kept"],
        "Rejected\n(temporal)": c["rejected_displacement"] + c["rejected_unsupported"] + c["rejected_static"],
        "Dropped\n(duplicate)": c["dropped_duplicate"],
        "Rejected\n(size/shape)": c["rejected_size"] + c["rejected_shape"],
        "Interpolated": c["interpolated"],
        "Rejected\n(selection)": c["rejected_selection"],
    }
    labels = list(cats.keys())
    vals = [100 * cats[k] / total for k in labels]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    colors = ["#2c3e50", "#c0392b", "#c0392b", "#c0392b", "#7f8c8d", "#c0392b"]
    y = range(len(labels))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of {total:,} detections (raw + stage-4 fabrications)", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_rejection_breakdown.pdf"))
    plt.close(fig)


def fig2_dropout_recovery():
    d = _load("dropout_recovery_results.json")
    by_gap = d["overall"]["by_gap"]
    gaps = sorted((int(g) for g in by_gap), key=int)
    recovery = [100 * by_gap[str(g)]["recovery_rate"] for g in gaps]
    iou50 = [by_gap[str(g)]["iou"]["50"] for g in gaps]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))

    ax = axes[0]
    bars = ax.bar([str(g) for g in gaps], recovery, color="#2c3e50")
    bars[-1].set_color("#c0392b")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_xlabel("Masked gap length (frames)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("tracker/stage-4\nthreshold coupling\n(see text)", xy=(len(gaps) - 1, recovery[-1]),
                xytext=(len(gaps) - 3.6, 45), fontsize=6.5, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7))

    ax = axes[1]
    ax.plot([str(g) for g in gaps], iou50, marker="o", color="#2c3e50", ms=4)
    ax.set_ylabel("Median IoU vs. masked box")
    ax.set_xlabel("Masked gap length (frames)")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_dropout_recovery.pdf"))
    plt.close(fig)


def fig3_ablation():
    ablation = _load("ablation_results.json")["ablation"]
    baseline_kept = ablation["full_pipeline"]["pct"]["kept"] * 100

    rules = [
        ("Static detection", "no_static_check"),
        ("Shape filter", "no_shape_filter"),
        ("Dedup", "no_dedup"),
        ("Selection cap", "no_selection_cap"),
        ("Displacement", "no_displacement_check"),
        ("Flicker/unsupported", "no_flicker_check"),
        ("Size filter", "no_size_filter"),
    ]
    deltas = [(name, ablation[key]["pct"]["kept"] * 100 - baseline_kept) for name, key in rules]
    deltas.sort(key=lambda t: abs(t[1]))

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    names = [d[0] for d in deltas]
    vals = [d[1] for d in deltas]
    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals]
    y = range(len(names))
    ax.barh(y, vals, color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(min(vals) - 1.6, max(vals) + 1.6)
    for i, v in enumerate(vals):
        ax.text(v + (0.2 if v >= 0 else -0.2), i, f"{v:+.1f}pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_xlabel("Change in kept% with rule disabled\n(vs. 85.3% full-pipeline baseline)", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.32)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_ablation.pdf"))
    plt.close(fig)


def fig4_sensitivity():
    sens = _load("ablation_results.json")["sensitivity"]

    panels = [
        ("static_px_threshold", "static_px_threshold (px)", "knife-edge"),
        ("plausible_shape.hi", "plausible_shape upper bound", "knife-edge"),
        ("duplicate_iou_threshold", "duplicate_iou_threshold", "flat"),
        ("track_gate_speed_px_per_frame", "track_gate_speed_px_per_frame (px/frame)", "flat"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.9), sharey=True)
    for ax, (key, label, tag) in zip(axes, panels):
        points = sens[key]["points"]
        xs = [p["value"] for p in points]
        ys = [p["outcome"]["pct"]["kept"] * 100 for p in points]
        default_x = next(p["value"] for p in points if p["is_default"])
        color = "#c0392b" if tag == "knife-edge" else "#27ae60"
        ax.plot(xs, ys, marker="o", ms=3, color=color, lw=1.2)
        ax.axvline(default_x, color="black", lw=0.6, ls="--")
        ax.set_xlabel(label, fontsize=6.3)
        ax.set_title(tag, fontsize=7, color=color)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("kept %")
    axes[0].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_sensitivity.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_rejection_breakdown()
    fig2_dropout_recovery()
    fig3_ablation()
    fig4_sensitivity()
    print("figures written to", FIG_DIR)
