#!/usr/bin/env python3
"""Rerun only SNR 0.05 with the original 200 seeds and summarize Figure 2.

Run: python experiments/camera_ready/neurips_snr005.py --workers 3
The original SNR 0.25 and 1.0 records are read in place, without modification.
"""
from __future__ import annotations

import correlated as core  # Sets all numerical-library thread counts to one.
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "results"
OUTPUT = ORIGINAL / "neurips_snr005"
ORIGINAL_CONFIG = deepcopy(core.CONFIG)
CONFIG = deepcopy(ORIGINAL_CONFIG)
CONFIG["snrs"] = [0.05, 0.25, 1.0]
REPLICATES = 200


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_worker():
    core.CONFIG = deepcopy(CONFIG)
    core.load_kernel()


def source_hashes():
    return {name: sha256(HERE / name) for name in
            ("neurips_snr005.py", "correlated.py", "gram_fs.cpp", "correlated_plot.py")}


def summarize(output):
    # Use exactly the interval implementation used for the original Figure 2.
    from correlated_plot import interval, write_csv
    raw, marginal, paired = [], [], []
    diagnostics, validation = {}, {}
    seeds = set()
    max_identity_error = 0.0
    for scenario, snr in enumerate(CONFIG["snrs"]):
        directory = output if scenario == 0 else ORIGINAL
        expected_config = CONFIG if scenario == 0 else ORIGINAL_CONFIG
        arrays, selected_ms, selected_ks = [], [], []
        for rep in range(REPLICATES):
            path = directory / "replicates" / f"correlated_snr{snr:g}_rep{rep:04d}.npz"
            with np.load(path) as saved:
                saved_config = json.loads(str(saved["config_json"]))
                assert saved_config == expected_config
                assert {k: v for k, v in saved_config.items() if k != "snrs"} == {
                    k: v for k, v in CONFIG.items() if k != "snrs"}
                assert float(saved["snr"]) == snr
                assert int(saved["scenario"]) == scenario
                assert int(saved["replicate"]) == rep
                values = saved["metrics"]
                assert values.shape == (2, CONFIG["kmax"], len(core.METRICS))
                assert np.all(np.isfinite(values)) and np.all(values[:, :, :2] >= 0)
                seed = int(saved["data_seed"])
                assert seed == core.seed_for(rep, scenario, 0) and seed not in seeds
                seeds.add(seed)
                sigma2 = float(saved["sigma2"])
                selected_m, selected_k = int(saved["selected_m"]), int(saved["selected_k"])
                cv = saved["cv_losses"]
                best_m = np.argmin(cv, axis=0)
                best_k = int(np.argmin(cv[best_m, np.arange(CONFIG["kmax"])])) + 1
                assert selected_k == best_k
                assert selected_m == CONFIG["m_grid"][best_m[best_k - 1]]
                np.testing.assert_array_equal(
                    np.count_nonzero(saved["coefficients"][0], axis=1),
                    np.arange(CONFIG["kmax"] + 1))
                if selected_m == CONFIG["p"]:
                    np.testing.assert_array_equal(values[0], values[1])
                delta = values[1] - values[0]
                np.testing.assert_allclose(delta[:, 2], delta[:, 3], rtol=0, atol=1e-10)
                error = np.max(np.abs(delta[:, 1] - delta[:, 0]
                                     - 2 * sigma2 / CONFIG["n"] * delta[:, 3]))
                max_identity_error = max(max_identity_error, float(error))
                if scenario == 0:
                    old_path = ORIGINAL / "replicates" / f"correlated_snr0.053_rep{rep:04d}.npz"
                    with np.load(old_path) as old:
                        assert int(old["data_seed"]) == seed
                        np.testing.assert_allclose(sigma2 / float(old["sigma2"]),
                                                   0.053 / 0.05, rtol=1e-14)
                arrays.append(values)
                selected_ms.append(selected_m)
                selected_ks.append(selected_k)
                for method, name in enumerate(("FS", "EFS")):
                    for ki in range(CONFIG["kmax"]):
                        record = dict(replicate=rep, snr=snr, method=name, k=ki + 1,
                                      selected_m=selected_m, selected_k=selected_k,
                                      sigma2=sigma2, data_seed=seed)
                        record.update({name: values[method, ki, mi]
                                       for mi, name in enumerate(core.METRICS)})
                        raw.append(record)
        values = np.stack(arrays)
        marginal_stats = interval(values)
        paired_stats = interval(values[:, 1] - values[:, 0])
        for method, name in enumerate(("FS", "EFS")):
            for ki in range(CONFIG["kmax"]):
                for mi, metric in enumerate(core.METRICS):
                    row = dict(snr=snr, method=name, k=ki + 1, metric=metric,
                               replicates=REPLICATES)
                    row.update({key: float(a[method, ki, mi]) for key, a in zip(
                        ("mean", "sd", "mcse", "ci_low", "ci_high"), marginal_stats)})
                    marginal.append(row)
        for ki in range(CONFIG["kmax"]):
            for mi, metric in enumerate(core.METRICS):
                row = dict(snr=snr, k=ki + 1, metric=metric, replicates=REPLICATES,
                           contrast="EFS minus FS")
                row.update({key: float(a[ki, mi]) for key, a in zip(
                    ("mean", "sd", "mcse", "ci_low", "ci_high"), paired_stats)})
                paired.append(row)
        diagnostics[str(snr)] = dict(
            selected_m_counts=dict(sorted(Counter(selected_ms).items())),
            selected_k_counts=dict(sorted(Counter(selected_ks).items())),
            source="new SNR 0.05 fits" if scenario == 0 else "unchanged original fits")
    assert len(seeds) == 600 and max_identity_error < 1e-9
    # Reused scenarios must reproduce the original summaries exactly.
    import csv
    for name, table in (("correlated_marginal_summary.csv", marginal),
                        ("correlated_paired_summary.csv", paired)):
        with (ORIGINAL / name).open() as handle:
            old_rows = [r for r in csv.DictReader(handle) if float(r["snr"]) != 0.053]
        new_rows = [r for r in table if r["snr"] != 0.05]
        assert len(old_rows) == len(new_rows)
        for old, new in zip(old_rows, new_rows):
            for key in ("mean", "sd", "mcse", "ci_low", "ci_high"):
                assert float(old[key]) == new[key]
    write_csv(output / "correlated_replicates.csv", raw)
    write_csv(output / "correlated_marginal_summary.csv", marginal)
    write_csv(output / "correlated_paired_summary.csv", paired)
    claims = {}
    for snr in CONFIG["snrs"]:
        claims[str(snr)] = {}
        for metric in ("training_mse", "df_covariance", "prediction_risk"):
            rows = [r for r in paired if r["snr"] == snr and r["metric"] == metric]
            claims[str(snr)][metric] = dict(
                negative_pointwise_interval_k=[r["k"] for r in rows if r["ci_high"] < 0],
                positive_pointwise_interval_k=[r["k"] for r in rows if r["ci_low"] > 0],
                interval_contains_zero_k=[r["k"] for r in rows if r["ci_low"] <= 0 <= r["ci_high"]],
                mean_and_interval_k1_2_10_20=[{key: r[key] for key in
                    ("k", "mean", "ci_low", "ci_high")} for r in rows if r["k"] in (1, 2, 10, 20)])
    validation = dict(status="passed", source_sha256=source_hashes(),
                      unique_data_seeds=len(seeds), new_low_snr_seed_matches=200,
                      new_replicate_records=200, reused_original_records=400,
                      configuration_difference="only snrs[0]: 0.053 to 0.05",
                      paired_risk_training_df_identity_max_error=max_identity_error,
                      reused_summary_rows="exactly unchanged", cv_selections="passed",
                      marginal_rows=len(marginal), paired_rows=len(paired), raw_rows=len(raw))
    for filename, obj in (("correlated_diagnostics.json", diagnostics),
                          ("validation.json", validation), ("claims.json", claims)):
        (output / filename).write_text(json.dumps(obj, indent=2) + "\n")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3, 4))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_worker()
    if args.summarize_only:
        print(json.dumps(summarize(args.output)), flush=True)
        return
    original_paths = sorted((ORIGINAL / "replicates").glob("correlated_*.npz"))
    assert len(original_paths) == 600
    original_hashes = {str(p.relative_to(HERE)): sha256(p) for p in original_paths}
    directory = args.output / "replicates"
    preexisting = len(list(directory.glob("correlated_snr0.05_rep*.npz")))
    manifest = dict(status="running", config=CONFIG, original_config=ORIGINAL_CONFIG,
                    replicates_per_snr=REPLICATES, workers=args.workers, blas_threads=1,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=source_hashes(), python=sys.version,
                    numpy=np.__version__, platform=platform.platform(),
                    new_snr=0.05, unchanged_snrs=[0.25, 1.0],
                    planned_low_snr_fits=200, reused_original_fits=400,
                    preexisting_low_snr_checkpoints=preexisting,
                    seed_policy="Original base_seed and all replicate/scenario/fit indices unchanged. "
                                "Scenario 0 uses identical designs, standard-normal noise draws, folds, "
                                "and fit RNG seeds. Only the noise scale changes.",
                    original_replicate_sha256=original_hashes,
                    interval="pointwise marginal/paired Student-t 95%, df=199")
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    completed, generated = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as pool:
        futures = [pool.submit(core.run_replicate, rep, 0, directory)
                   for rep in range(REPLICATES)]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            generated += not result.get("cached", False)
            if completed % 10 == 0 or completed == REPLICATES:
                print(json.dumps(dict(completed=completed, planned=REPLICATES,
                                      seconds=time.monotonic() - started)), flush=True)
    fit_seconds = time.monotonic() - started
    validation = summarize(args.output)
    assert all(sha256(HERE / p) == digest for p, digest in original_hashes.items())
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    generated_low_snr_fits=generated, total_low_snr_fits=completed,
                    fit_elapsed_seconds=fit_seconds, elapsed_seconds=time.monotonic() - started,
                    original_replicates_unchanged=True, validation=validation["status"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "generated_low_snr_fits",
                     "reused_original_fits", "fit_elapsed_seconds", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
