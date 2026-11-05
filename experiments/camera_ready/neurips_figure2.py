#!/usr/bin/env python3
"""Restyle the saved Figure 2 estimates for the NeurIPS manuscript only."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/efs-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = Path(__file__).parent / "results/neurips_1000/correlated_marginal_summary.csv"
REPLICATES = 1000
METHODS = ("FS", "EFS")
COLORS = ("#0072B2", "#D55E00")
MARKERS = ("o", "s")
SNRS = (0.05, 0.25, 1.0)


def load_estimates(path, replicates=REPLICATES):
    with path.open() as handle:
        records = list(csv.DictReader(handle))
    estimates = {}
    for snr in SNRS:
        for method in METHODS:
            for metric in ("training_mse", "df_covariance"):
                rows = sorted((row for row in records if float(row["snr"]) == snr
                               and row["method"] == method and row["metric"] == metric),
                              key=lambda row: int(row["k"]))
                assert [int(row["k"]) for row in rows] == list(range(1, 21))
                if any(int(row["replicates"]) != replicates for row in rows):
                    raise ValueError(f"Expected {replicates} datasets per SNR in {path}")
                values = np.array([[float(row[key]) for key in ("mean", "ci_low", "ci_high")]
                                   for row in rows])
                assert np.all(values[:, 1] <= values[:, 0])
                assert np.all(values[:, 0] <= values[:, 2])
                estimates[snr, method, metric] = values
    return estimates


def plot(estimates, output, preview=None):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "axes.edgecolor": "#444444", "xtick.color": "#444444",
        "ytick.color": "#444444", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.285, top=0.85, wspace=0.27)
    for index, (ax, snr) in enumerate(zip(axes, SNRS)):
        bounds = []
        for method, color, marker in zip(METHODS, COLORS, MARKERS):
            x = estimates[snr, method, "training_mse"]
            y = estimates[snr, method, "df_covariance"]
            # Plot every saved mean and both pointwise interval coordinates.
            ax.errorbar(x[:, 0], y[:, 0],
                        xerr=np.array([x[:, 0]-x[:, 1], x[:, 2]-x[:, 0]]),
                        yerr=np.array([y[:, 0]-y[:, 1], y[:, 2]-y[:, 0]]),
                        fmt="none", ecolor=color, elinewidth=0.65, capsize=0.85,
                        capthick=0.6, alpha=0.75, zorder=4.5)
            # Keep every curve vertex and interval, but mark only annotated steps.
            ax.plot(x[:, 0], y[:, 0], color=color, linewidth=1.05,
                    marker=marker, markevery=[0, 19], markersize=2.3,
                    markerfacecolor="white", markeredgewidth=0.6, zorder=3)
            ax.plot(x[9, 0], y[9, 0], marker=marker, markersize=3.5,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.45, linestyle="none", zorder=4)
            bounds.append((x[:, 1].min(), x[:, 2].max(), y[:, 1].min(), y[:, 2].max()))
        bounds = np.array(bounds)
        xmin, xmax = bounds[:, 0].min(), bounds[:, 1].max()
        ymin, ymax = bounds[:, 2].min(), bounds[:, 3].max()
        ax.set_xlim(xmin - 0.07*(xmax-xmin), xmax + 0.07*(xmax-xmin))
        ax.set_ylim(max(0, ymin-0.12*(ymax-ymin)), ymax+0.15*(ymax-ymin))
        x = estimates[snr, "FS", "training_mse"][:, 0]
        y = estimates[snr, "FS", "df_covariance"][:, 0]
        # Put labels in panel-specific whitespace, connected to their FS points.
        # Axes-fraction locations keep the text separate from both paths and CIs.
        label_positions = (
            {1: (0.84, 0.36), 10: (0.65, 0.74), 20: (0.37, 0.97)},
            {1: (0.83, 0.35), 10: (0.55, 0.64), 20: (0.34, 0.97)},
            {1: (0.82, 0.48), 10: (0.38, 0.08), 20: (0.34, 0.96)},
        )[index]
        for k, position in label_positions.items():
            point_clearance = 5.5 if index == 0 and k == 20 else 3
            ax.annotate(rf"$k={k}$", xy=(x[k-1], y[k-1]), xytext=position,
                        textcoords="axes fraction", ha="center", va="center",
                        fontsize=7.2, color="#333333",
                        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5,
                                    "shrinkA": 2, "shrinkB": point_clearance},
                        zorder=5)
        ax.set_xlabel("Training MSE", labelpad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, pad=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#DADADA", linewidth=0.35, alpha=0.45)
        ax.set_axisbelow(True)
        center = (ax.get_position().x0 + ax.get_position().x1)/2
        snr_text = ("0.05", "0.25", "1.0")[index]
        fig.text(center, 0.035, f"({chr(97+index)}) SNR = {snr_text}",
                 ha="center", va="bottom", fontsize=9)
    fig.text(0.012, 0.57, "Degrees of freedom", rotation=90, va="center", fontsize=9)
    handles = [Line2D([0], [0], color=color, marker=marker, markersize=3.5,
                      markerfacecolor="white", markeredgewidth=0.65, linewidth=1.1,
                      label=method) for method, color, marker in zip(METHODS, COLORS, MARKERS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.015),
               ncol=2, frameon=False, handlelength=1.8, handletextpad=0.5,
               columnspacing=1.6, borderaxespad=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    if preview:
        fig.savefig(preview, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--output", type=Path, default=ROOT / "neurips/figures/figure2.pdf")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    plot(load_estimates(args.summary, args.replicates), args.output, args.preview)


if __name__ == "__main__":
    main()
