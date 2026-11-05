#!/usr/bin/env python3
"""Compute the pointwise Monte Carlo intervals for Figure 2 and Appendix A."""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.stats import t

from correlated import CONFIG, METRICS

METHODS = ("FS", "EFS")


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


def summarize(output, expected):
    kmax = CONFIG["kmax"]
    summary = []
    paired = []
    all_seeds = set()
    for scenario, snr in enumerate(CONFIG["snrs"]):
        arrays = []
        for rep in range(expected):
            path = output / "replicates" / f"correlated_snr{snr:g}_rep{rep:04d}.npz"
            with np.load(path) as saved:
                if json.loads(str(saved["config_json"])) != CONFIG:
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
                                           (2 * sigma2 / CONFIG["n"]) * delta[:, 3],
                                           rtol=0, atol=1e-9)
                np.testing.assert_array_equal(np.count_nonzero(coefficients[0], axis=1),
                                              np.arange(kmax + 1))
                selected_m, selected_k = int(saved["selected_m"]), int(saved["selected_k"])
                cv = saved["cv_losses"]
                best_m_by_k = np.argmin(cv, axis=0)
                check_k = int(np.argmin(cv[best_m_by_k, np.arange(kmax)])) + 1
                assert check_k == selected_k
                assert CONFIG["m_grid"][best_m_by_k[check_k - 1]] == selected_m
                data_seed = int(saved["data_seed"])
                seed_sequence = np.random.SeedSequence([CONFIG["base_seed"], rep, scenario, 0])
                expected_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
                if data_seed != expected_seed:
                    raise AssertionError(f"Data seed mismatch: {path}")
                assert data_seed not in all_seeds
                all_seeds.add(data_seed)
                if selected_m == CONFIG["p"]:
                    np.testing.assert_array_equal(values[0], values[1])
                arrays.append(values)
        values = np.stack(arrays)
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
    write_csv(output / "correlated_marginal_summary.csv", summary)
    write_csv(output / "correlated_paired_summary.csv", paired)
