#!/usr/bin/env python3
"""Restyle Figure 1 using the original PDF's recovered polyline vertices.

Run with no arguments to plot the checked-in CSV.  To repeat the extraction:
    python neurips_figure1.py --extract-source images/wj_vs_gamma_logistic.pdf

The extraction changes neither curve values nor interpolation.  The original
plotting source is unavailable, so the PDF vertices, not a recomputed theoretical
formula, are the source of truth.  See the accompanying provenance JSON for a
pre-existing difference between the dashed curves and the original caption's
formula.  The NeurIPS caption has been corrected to describe the plotted curves.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/efs-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS / "figure1_original_curves.csv"
PROVENANCE_PATH = RESULTS / "figure1_original_curves_provenance.json"
VALIDATION_PATH = RESULTS / "figure1_restyle_validation.json"
SOURCE_SHA256 = "5d6c7bb92a53eec24fd78419bbc5fe51b85bfcb82b852ee1467fd200eb47b8d3"
SOURCE_RECT = (65.29375, 46.825, 421.36, 297.16)
XLIM = (0.0, 20.95)
YLIM = (0.0, 1.05)
GAMMAS = (0.05, 0.15, 1.0 / 3.0)
GAMMA_LABELS = ("0.05", "0.15", "0.333")
OLD_COLORS = ((0.36470588, 0.29019608, 0.59607843),
              (0.99215686, 0.55294118, 0.23529412),
              (0.45490196, 0.6627451, 0.81176471))
COLORS = ("#756BB1", "#D55E00", "#0072B2")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polylines(page):
    """Return stroked, untransformed line paths with graphics-state metadata."""
    state = {"color": None, "dash": (), "clip_rect": None,
             "matrix": (1., 0., 0., 1., 0., 0.)}
    stack, vertices = [], []
    for index, (arguments, operator) in enumerate(page.get_contents().operations):
        if operator == b"q":
            stack.append(copy.deepcopy(state))
        elif operator == b"Q":
            state = stack.pop()
        elif operator == b"cm":
            # Every scientific curve in the source and output is untransformed.
            # Record any nonidentity transform so it cannot be silently ignored.
            state["matrix"] = tuple(map(float, arguments))
        elif operator == b"RG":
            state["color"] = tuple(map(float, arguments))
        elif operator == b"d":
            state["dash"] = tuple(map(float, arguments[0]))
        elif operator == b"re":
            state["clip_rect"] = tuple(map(float, arguments))
        elif operator == b"m":
            vertices = [tuple(map(float, arguments))]
        elif operator == b"l":
            vertices.append(tuple(map(float, arguments)))
        elif operator == b"S":
            if len(vertices) == 20 and state["color"] is not None:
                assert state["matrix"] == (1., 0., 0., 1., 0., 0.)
                yield index, copy.deepcopy(state), np.asarray(vertices)
            vertices = []
        elif operator in (b"n", b"f", b"f*", b"B", b"B*"):
            vertices = []


def inverse_axes(vertices, rectangle):
    left, bottom, width, height = rectangle
    return (vertices - (left, bottom)) * (
        (XLIM[1] - XLIM[0]) / width, (YLIM[1] - YLIM[0]) / height)


def extract(source):
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("The source PDF differs from the audited original.")
    reader = PdfReader(source)
    assert len(reader.pages) == 1
    rows, curves = [], []
    for operation, state, vertices in polylines(reader.pages[0]):
        matches = [i for i, color in enumerate(OLD_COLORS)
                   if np.allclose(state["color"], color, rtol=0, atol=5e-9)]
        if not matches:
            continue
        assert state["clip_rect"] == SOURCE_RECT
        gamma_index = matches[0]
        variant = "approximation" if state["dash"] else "exact"
        data = inverse_axes(vertices, SOURCE_RECT)
        assert np.max(np.abs(data[:, 0] - np.arange(1, 21))) < 3e-8
        assert np.all(np.diff(data[:, 1]) < 0)
        for point, (xy, pdf_xy) in enumerate(zip(data, vertices), start=1):
            rows.append(dict(gamma_index=gamma_index, gamma=GAMMAS[gamma_index],
                             display_gamma=GAMMA_LABELS[gamma_index],
                             variant=variant, point_index=point,
                             j=xy[0], weight=xy[1],
                             original_pdf_x=pdf_xy[0], original_pdf_y=pdf_xy[1]))
        formula_diagnostics = {}
        if variant == "approximation":
            alpha = -math.log1p(-GAMMAS[gamma_index])
            h = math.log(math.expm1(10 * alpha)) / alpha
            for name, midpoint in (("h_plus_one_half", h + .5),
                                   ("k_plus_one_half", 10.5)):
                values = 1 / (1 + np.exp(alpha * (data[:, 0] - midpoint)))
                formula_diagnostics[name + "_maximum_absolute_difference"] = float(
                    np.max(np.abs(data[:, 1] - values)))
        roundtrip = data * (SOURCE_RECT[2] / 20.95, SOURCE_RECT[3] / 1.05)
        roundtrip += SOURCE_RECT[:2]
        curves.append(dict(gamma_index=gamma_index, gamma=GAMMAS[gamma_index],
                           display_gamma=GAMMA_LABELS[gamma_index], variant=variant,
                           vertex_count=len(vertices), stroke_operation=operation,
                           original_rgb=state["color"], original_dash=state["dash"],
                           maximum_roundtrip_error_pdf_points=float(
                               np.max(np.abs(roundtrip - vertices))),
                           formula_diagnostics=formula_diagnostics))
    assert len(curves) == 6
    assert {(c["gamma_index"], c["variant"]) for c in curves} == {
        (i, v) for i in range(3) for v in ("exact", "approximation")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(
        source="images/wj_vs_gamma_logistic.pdf", source_sha256=SOURCE_SHA256,
        csv_sha256=sha256(CSV_PATH), source_page=1,
        source_page_size_pdf_points=list(map(float, reader.pages[0].mediabox)),
        source_axes_rectangle_pdf_points=SOURCE_RECT, x_limits=XLIM, y_limits=YLIM,
        extraction="Inverse affine map of all 120 original PDF polyline vertices. "
                   "No fitting, smoothing, interpolation changes, or formula substitution.",
        axis_calibration="Plot rectangle and labeled ticks give x limits [0,20.95] "
                         "and y limits [0,1.05]. Original PDF coordinates are retained "
                         "alongside each recovered data coordinate.",
        scientific_note="The dashed curves agree with midpoint h(alpha,10)+1/2, "
                        "not 10+1/2 as referenced by the original caption. The "
                        "NeurIPS caption has been corrected to describe the plotted "
                        "curves. The third gamma is 1/3, displayed as 0.333. "
                        "All original curve coordinates are preserved.",
        curves=curves)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n")


def load_curves():
    provenance = json.loads(PROVENANCE_PATH.read_text())
    assert sha256(CSV_PATH) == provenance["csv_sha256"]
    with CSV_PATH.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 120
    curves = {}
    for index in range(3):
        for variant in ("exact", "approximation"):
            points = [r for r in rows if int(r["gamma_index"]) == index
                      and r["variant"] == variant]
            assert [int(r["point_index"]) for r in points] == list(range(1, 21))
            curves[index, variant] = np.array([
                (float(r["j"]), float(r["weight"])) for r in points])
    return curves


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
    for index in range(3):
        for variant in ("exact", "approximation"):
            xy = curves[index, variant]
            ax.plot(xy[:, 0], xy[:, 1], color=COLORS[index], linewidth=1.2,
                    linestyle="-" if variant == "exact" else (0, (3.2, 2.0)),
                    solid_capstyle="round", dash_capstyle="butt", zorder=2)
    ax.axvline(10, color="#777777", linewidth=.8, linestyle=(0, (3, 2)), zorder=0)
    ax.annotate(r"$k=10$", xy=(10, 1.0), xytext=(-5, -1),
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
                                 "Subject": "Restyled original vector curves",
                                 "CreationDate": None, "ModDate": None})
    fig.savefig(preview, dpi=220)
    plt.close(fig)


def validate_output(curves, output):
    reader = PdfReader(output)
    assert len(reader.pages) == 1
    assert np.allclose(list(map(float, reader.pages[0].mediabox)), [0, 0, 216, 147.6])
    new_colors = [tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
                  for color in COLORS]
    checks = []
    for _, state, vertices in polylines(reader.pages[0]):
        matches = [i for i, color in enumerate(new_colors)
                   if np.allclose(state["color"], color, rtol=0, atol=5e-9)]
        if not matches:
            continue
        index = matches[0]
        variant = "approximation" if state["dash"] else "exact"
        recovered = inverse_axes(vertices, state["clip_rect"])
        maximum = np.max(np.abs(recovered - curves[index, variant]), axis=0)
        assert maximum[0] < 1e-7 and maximum[1] < 1e-8
        checks.append(dict(gamma_index=index, variant=variant, vertex_count=20,
                           maximum_j_difference=float(maximum[0]),
                           maximum_weight_difference=float(maximum[1])))
    assert len(checks) == 6
    report = dict(source_sha256=SOURCE_SHA256, csv_sha256=sha256(CSV_PATH),
                  output=os.path.relpath(output, ROOT), output_sha256=sha256(output),
                  source_vertices_preserved=120, output_size_inches=[3.0, 2.05],
                  x_limits=XLIM, y_limits=YLIM,
                  note="Differences reflect decimal rounding in PDF path serialization.",
                  curves=checks)
    VALIDATION_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "neurips/figures/figure1.pdf")
    parser.add_argument("--preview", type=Path, default=Path("/tmp/efs-figure1-restyled.png"))
    args = parser.parse_args()
    if args.extract_source:
        extract(args.extract_source)
    curves = load_curves()
    plot(curves, args.output, args.preview)
    validate_output(curves, args.output)


if __name__ == "__main__":
    main()
