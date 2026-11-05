#!/usr/bin/env python3
"""Draw Figure 1 from the limiting weights and their logistic approximation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/efs-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compute_weights import GAMMA_LABELS, K, RANKS, compute

ROOT = Path(__file__).resolve().parents[2]
XLIM = (0.0, 20.95)
YLIM = (0.0, 1.05)
COLORS = ("#756BB1", "#D55E00", "#0072B2")


def plot(curves, output, preview):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": .6,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "pdf.fonttype": 42, "path.simplify": False,
    })
    fig, ax = plt.subplots(figsize=(3.0, 2.05))
    fig.subplots_adjust(left=.155, right=.985, bottom=.20, top=.965)
    for index, color in enumerate(COLORS):
        for variant in ("exact", "approximation"):
            ax.plot(range(1, RANKS + 1), curves[index, variant], color=color, linewidth=1.2,
                    linestyle="-" if variant == "exact" else (0, (3.2, 2.0)),
                    solid_capstyle="round", dash_capstyle="butt", zorder=2)
    ax.axvline(K, color="#777777", linewidth=.8, linestyle=(0, (3, 2)), zorder=0)
    ax.annotate(rf"$k={K}$", xy=(K, 1.0), xytext=(-5, -1),
                textcoords="offset points", ha="right", va="top", fontsize=8,
                color="#666666")
    ax.set(xlim=XLIM, ylim=YLIM, xlabel=r"$j$", ylabel=r"$\widetilde{w}_j^{k,\gamma}$")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.6, pad=2)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 3
    handles = [Line2D([], [], color=color, linewidth=1.2,
                      label=rf"$\gamma={label}$")
               for color, label in zip(COLORS, GAMMA_LABELS)]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              borderaxespad=.2, handlelength=1.6, handletextpad=.5,
              labelspacing=.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Title": "EFS weights and logistic approximation",
                                 "Subject": "Theorem 3.1 and its logistic approximation",
                                 "CreationDate": None, "ModDate": None})
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "neurips/figures/figure1.pdf")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    curves, _ = compute()
    plot(curves, args.output, args.preview)


if __name__ == "__main__":
    main()
