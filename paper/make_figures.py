#!/usr/bin/env python3
"""Draw paper Figs. 2-4 (vector PDF) from the CSVs written by reproduce_paper.py.

    python paper/reproduce_paper.py --data-dir data
    python paper/make_figures.py                     # -> paper/figures/*.pdf

Fig. 1 (pipeline diagram) is drawn in TikZ inside main.tex.
"""
import argparse
import csv
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL_W = 3.45  # IEEEtran column width, inches
FOCAL, GREY, LIGHT, INTERP = "#1f4e79", "#8c8c8c", "#e6e6e6", "#7fa7cc"
META = {"Creator": None, "Producer": None, "Author": None, "Title": None, "CreationDate": None}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "hatch.color": "white", "hatch.linewidth": 0.8,
})


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct1(v):
    return f"{Decimal(str(round(v, 3))).quantize(Decimal('0.1'), ROUND_HALF_UP)}%"


def hbar(ax, labels, vals, colors, hatches):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, [100] * len(y), color=LIGHT, height=0.62, zorder=0)
    for yi, v, c, h in zip(y, vals, colors, hatches):
        ax.barh(yi, v, color=c, height=0.62, hatch=h, edgecolor="white" if h else c, linewidth=0, zorder=1)
        ax.text(101.5, yi, pct1(v), va="center", ha="left")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, ha="right", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def save(fig, name):
        fig.savefig(os.path.join(a.out, name + ".pdf"), metadata=META, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # Fig. 2: raw detections per frame
    d2 = rows(os.path.join(a.results, "fig2_boxes_per_frame.csv"))
    lab = {"2": "2 boxes", "1": "1 box", "3+": "3 or more boxes", "0": "0 boxes"}
    fig, ax = plt.subplots(figsize=(COL_W * 0.78, 1.25))
    hbar(ax, [lab[r["boxes_per_frame"]] for r in d2], [float(r["pct_of_frames"]) for r in d2],
         [FOCAL, GREY, GREY, GREY], [None] * 4)
    ax.set_xlabel("Share of frames (%)")
    save(fig, "fig2_boxes_per_frame")

    # Fig. 3: outcome of every detection
    d3 = rows(os.path.join(a.results, "fig3_outcomes.csv"))
    lab = {"kept": "Kept", "rejected_temporal": "Rejected: temporal (Stage 3)",
           "dropped_duplicate": "Merged: duplicate (Stage 1)", "rejected_size_shape": "Rejected: size/shape (Stage 1)",
           "interpolated": "Interpolated (Stage 4)", "rejected_selection": "Rejected: selection (Stage 5)"}
    fig, ax = plt.subplots(figsize=(COL_W * 0.62, 1.7))
    hbar(ax, [lab[r["outcome"]] for r in d3], [float(r["pct_of_total"]) for r in d3],
         [FOCAL if r["outcome"] == "kept" else INTERP if r["outcome"] == "interpolated" else GREY for r in d3],
         ["////" if r["outcome"] == "interpolated" else None for r in d3])
    ax.set_xlabel("Share of all detections (%)")
    save(fig, "fig3_outcomes")

    # Fig. 4: synthetic-dropout self-consistency test
    d4 = rows(os.path.join(a.results, "fig4_dropout_recovery.csv"))
    g = np.array([int(r["gap_frames"]) for r in d4])
    rec = np.array([float(r["recovery_pct"]) for r in d4])
    ff = np.array([float(r["false_fill_pct"]) for r in d4])
    q25, q50, q75 = (np.array([float(r[k]) for r in d4]) for k in ("iou_p25", "iou_median", "iou_p75"))
    ok = g <= 10
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W + 0.15, 1.75), gridspec_kw={"wspace": 0.55})
    a1.plot(g[ok], rec[ok], "-o", color=FOCAL, ms=3.5, lw=1.2)
    a1.plot(g[~ok], rec[~ok], "o", mfc="white", mec=FOCAL, ms=4, mew=1.0)
    a1.plot(g, ff, "-s", color=GREY, ms=3, lw=1.0)
    a1.annotate("tracker\nexpires", xy=(g[~ok][0], rec[~ok][0]), xytext=(9.2, 30), ha="center", fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    a1.set(xlabel="Masked gap (frames)", ylabel="Rate (%)", ylim=(-4, 100), xticks=[1, 5, 10, 15])
    a1.text(1.0, 86, "recovered", color=FOCAL, fontsize=7)
    a1.text(1.0, 8, "false fill", color=GREY, fontsize=7)
    letter(a1, "a")
    a2.fill_between(g[ok], q25[ok], q75[ok], color=FOCAL, alpha=0.18, lw=0)
    a2.plot(g[ok], q50[ok], "-o", color=FOCAL, ms=3.5, lw=1.2)
    a2.set(xlabel="Masked gap (frames)", ylabel="IoU with removed box", ylim=(0.4, 1.0), xticks=[1, 3, 5, 10])
    a2.text(1.2, 0.47, "median, IQR band", fontsize=7, color="0.25")
    letter(a2, "b")
    save(fig, "fig4_dropout_recovery")
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
