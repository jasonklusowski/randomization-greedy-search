#!/usr/bin/env python3
"""Summarize saved correlated-design replicates and plot pointwise 95% t intervals."""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/efs-matplotlib")
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from correlated import CONFIG, METRICS, RESULTS

METHODS = ("FS", "EFS")
COLORS = ("#D55E00", "#E9AC44")


def interval(values):
    n = len(values)
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)
    se = sd / np.sqrt(n)
    radius = t.ppf(0.975, n - 1) * se
    return mean, sd, se, mean - radius, mean + radius


def write_csv(path, records):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(output, expected, *, source_for=None, config=None):
    """Summarize checkpoints, optionally using a supplied path and config per replicate."""
    if config is None:
        config = CONFIG
    kmax = config["kmax"]
    raw = []
    summary = []
    paired = []
    data = {}
    diagnostics = {}
    all_seeds = set()
    for scenario, snr in enumerate(config["snrs"]):
        arrays, selected_ms, selected_ks, elapsed = [], [], [], []
        for rep in range(expected):
            if source_for is None:
                path = output / "replicates" / f"correlated_snr{snr:g}_rep{rep:04d}.npz"
                source_config = config
            else:
                path, source_config = source_for(rep, scenario)
                # Historical SNR grids may differ at other scenario indices.
                if (source_config["snrs"][scenario] != snr
                        or {**source_config, "snrs": config["snrs"]} != config):
                    raise AssertionError(f"Incompatible source configuration: {path}")
            with np.load(path) as saved:
                if json.loads(str(saved["config_json"])) != source_config:
                    raise AssertionError(f"Configuration mismatch: {path}")
                identity = (int(saved["replicate"]), int(saved["scenario"]), float(saved["snr"]))
                if identity != (rep, scenario, snr):
                    raise AssertionError(f"Replicate identity mismatch: {path}")
                values = saved["metrics"]
                coefficients = saved["coefficients"]
                assert values.shape == (2, kmax, len(METRICS))
                assert np.all(np.isfinite(values))
                assert np.all(values[:, :, :2] >= 0)
                sigma2 = float(saved["sigma2"])
                delta = values[1] - values[0]
                np.testing.assert_allclose(delta[:, 2], delta[:, 3], rtol=0, atol=1e-10)
                np.testing.assert_allclose(delta[:, 1] - delta[:, 0],
                                           (2 * sigma2 / config["n"]) * delta[:, 3],
                                           rtol=0, atol=1e-9)
                np.testing.assert_array_equal(np.count_nonzero(coefficients[0], axis=1),
                                              np.arange(kmax + 1))
                selected_m, selected_k = int(saved["selected_m"]), int(saved["selected_k"])
                cv = saved["cv_losses"]
                best_m_by_k = np.argmin(cv, axis=0)
                check_k = int(np.argmin(cv[best_m_by_k, np.arange(kmax)])) + 1
                assert check_k == selected_k
                assert config["m_grid"][best_m_by_k[check_k - 1]] == selected_m
                data_seed = int(saved["data_seed"])
                seed_sequence = np.random.SeedSequence([config["base_seed"], rep, scenario, 0])
                expected_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
                if data_seed != expected_seed:
                    raise AssertionError(f"Data seed mismatch: {path}")
                assert data_seed not in all_seeds
                all_seeds.add(data_seed)
                if selected_m == config["p"]:
                    np.testing.assert_array_equal(values[0], values[1])
                arrays.append(values)
                selected_ms.append(selected_m)
                selected_ks.append(selected_k)
                elapsed.append(float(saved["elapsed_seconds"]))
                for method, method_name in enumerate(METHODS):
                    for ki in range(kmax):
                        record = dict(replicate=rep, snr=snr, method=method_name, k=ki + 1,
                                      selected_m=selected_m, selected_k=selected_k,
                                      sigma2=sigma2, data_seed=data_seed)
                        record.update({name: values[method, ki, mi]
                                       for mi, name in enumerate(METRICS)})
                        raw.append(record)
        values = np.stack(arrays)
        data[snr] = values
        marginal = interval(values)
        differences = interval(values[:, 1] - values[:, 0])
        for method, method_name in enumerate(METHODS):
            for ki in range(kmax):
                for mi, metric in enumerate(METRICS):
                    record = dict(snr=snr, method=method_name, k=ki + 1,
                                  metric=metric, replicates=expected)
                    record.update({key: float(array[method, ki, mi]) for key, array in
                                   zip(("mean", "sd", "mcse", "ci_low", "ci_high"), marginal)})
                    summary.append(record)
        for ki in range(kmax):
            for mi, metric in enumerate(METRICS):
                record = dict(snr=snr, k=ki + 1, metric=metric, replicates=expected,
                              contrast="EFS minus FS")
                record.update({key: float(array[ki, mi]) for key, array in
                               zip(("mean", "sd", "mcse", "ci_low", "ci_high"), differences)})
                paired.append(record)
        df_estimator_offset = values[:, 0, :, 2] - values[:, 0, :, 3]
        diagnostics[str(snr)] = dict(
            selected_m_counts=dict(sorted(Counter(selected_ms).items())),
            selected_k_counts=dict(sorted(Counter(selected_ks).items())),
            per_dataset_seconds_mean=float(np.mean(elapsed)),
            per_dataset_seconds_range=[float(np.min(elapsed)), float(np.max(elapsed))],
            optimism_minus_covariance_mean=float(df_estimator_offset.mean()),
            optimism_minus_covariance_mcse=float(df_estimator_offset[:, 0].std(ddof=1)/np.sqrt(expected)),
            paired_df_estimators_max_discrepancy=float(np.max(np.abs(
                (values[:, 1, :, 2] - values[:, 0, :, 2])
                - (values[:, 1, :, 3] - values[:, 0, :, 3])))),
            risk_improvement_mean_k1_10_20=[float(differences[0][k - 1, 1]) for k in (1, 10, 20)],
            training_improvement_mean_k1_10_20=[float(differences[0][k - 1, 0]) for k in (1, 10, 20)],
            df_improvement_mean_k1_10_20=[float(differences[0][k - 1, 3]) for k in (1, 10, 20)])
    write_csv(output / "correlated_replicates.csv", raw)
    write_csv(output / "correlated_marginal_summary.csv", summary)
    write_csv(output / "correlated_paired_summary.csv", paired)
    (output / "correlated_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return data, diagnostics


def plot_marginal(output, snr, values):
    mean, sd, se, low, high = interval(values)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    fig, ax = plt.subplots(figsize=(2.05, 1.9), layout="constrained")
    for method, name in enumerate(METHODS):
        x, y = mean[method, :, 0], mean[method, :, 3]
        xerr = (high[method, :, 0] - low[method, :, 0])/2
        yerr = (high[method, :, 3] - low[method, :, 3])/2
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=COLORS[method],
                    elinewidth=0.55, capsize=1.1, capthick=0.55, alpha=0.55, zorder=1)
        ax.plot(x, y, color=COLORS[method], marker="o" if method == 0 else "*",
                markersize=2.8 if method == 0 else 4.2, linewidth=1.05,
                markeredgecolor="#252525", markeredgewidth=0.3, label=name, zorder=2)
        # Endpoints identify direction without overcrowding the small main panels.
        if method == 0:
            for ki, offset in ((0, (-2, -9)), (19, (2, 4))):
                ax.annotate(f"k={ki + 1}", (x[ki], y[ki]), xytext=offset,
                            textcoords="offset points", fontsize=6.7,
                            ha="right" if ki == 0 else "left", color=COLORS[method])
    ax.set_xlabel("Training MSE", labelpad=2)
    ax.set_ylabel("Degrees of freedom", labelpad=2)
    ax.grid(alpha=0.16, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0,
              handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
    ax.tick_params(length=2, pad=2)
    ax.margins(x=0.15, y=0.12)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    old_snr = {0.053: "0.05", 0.25: "0.25", 1.0: "1.00"}[snr]
    stem = f"correlated_mse_vs_df_by_k_snr_{old_snr}_banded_exact_p100"
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--replicates", type=int, default=200)
    args = parser.parse_args()
    data, diagnostics = summarize(args.output, args.replicates)
    for snr, values in data.items():
        plot_marginal(args.output, snr, values)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
