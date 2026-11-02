#!/usr/bin/env python3
"""Reproduce the correlated-design comparison with paired Monte Carlo uncertainty.

Fresh implementation of the paper's independent-base-learner EFS estimator.
Both methods fit an unpenalized intercept. CV jointly selects (k,m), then the
whole k path is refitted at that one selected m, as in the archived EFSCV.
The archived optional resampling of feature-set frequencies is not used.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONFIG = dict(n=1000, p=100, rho=0.5, sparsity=10, snrs=[0.053, 0.25, 1.0],
              ensemble_size=500, folds=10, kmax=20, base_seed=20261027,
              m_grid=[int(2 + 98 * (1.5**i - 1)/(1.5**9 - 1)) for i in range(10)],
              intercept=True, resample_feature_frequencies=False,
              cv_convention="joint k,m selection; full k path at selected m",
              estimator_version=1)
METRICS = ("training_mse", "prediction_risk", "df_optimism", "df_covariance")
_LIB = None


def load_kernel():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = HERE / "gram_fs.cpp"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    binary = Path(tempfile.gettempdir()) / f"efs_gram_{digest}.so"
    if not binary.exists():
        temporary = binary.with_suffix(f".{os.getpid()}.so")
        subprocess.run(["clang++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(source), "-o", str(temporary)], check=True)
        os.replace(temporary, binary)
    _LIB = ctypes.CDLL(str(binary))
    ptr = ctypes.POINTER(ctypes.c_double)
    iptr = ctypes.POINTER(ctypes.c_int)
    _LIB.efs_gram.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                            ptr, iptr, iptr]
    _LIB.efs_gram.restype = ctypes.c_int
    return _LIB


def seed_for(*indices):
    return int(np.random.SeedSequence([CONFIG["base_seed"], *indices])
               .generate_state(1, dtype=np.uint64)[0])


def fit_gram(G, g, kmax, m, B, seed, trace=False):
    G = np.ascontiguousarray(G, dtype=np.float64)
    g = np.ascontiguousarray(g, dtype=np.float64)
    p = len(g)
    coefficients = np.zeros((kmax + 1, p))
    actual_B = 1 if m == p else B
    selected = np.full((actual_B, kmax), -1, dtype=np.int32) if trace else None
    candidates = np.full((actual_B, kmax, p), -1, dtype=np.int32) if trace else None
    ptr = ctypes.POINTER(ctypes.c_double)
    iptr = ctypes.POINTER(ctypes.c_int)
    status = load_kernel().efs_gram(
        G.ctypes.data_as(ptr), g.ctypes.data_as(ptr), p, kmax, m, B, seed,
        coefficients.ctypes.data_as(ptr),
        selected.ctypes.data_as(iptr) if trace else None,
        candidates.ctypes.data_as(iptr) if trace else None)
    if status:
        raise RuntimeError(f"Gram FS kernel failed with status {status}")
    return (coefficients, selected, candidates) if trace else coefficients


def centered_stats(X, y):
    xm, ym = X.mean(axis=0), y.mean()
    xc, yc = X - xm, y - ym
    return xc.T @ xc, xc.T @ yc, xm, ym


def validate_kernel():
    """Check selected features and refits against direct least squares."""
    rng = np.random.default_rng(771)
    X = rng.normal(size=(80, 12))
    X[:, 1] += 0.7 * X[:, 0]
    X += rng.normal(size=12)
    y = 2.4 + X[:, :3] @ np.array([1., -0.5, 0.4]) + rng.normal(size=80)
    G, g, xm, ym = centered_stats(X, y)
    xc, yc = X - xm, y - ym
    maximum_error = 0.0
    for m in (1, 3, 7, 12):
        coefs, selected, candidates = fit_gram(G, g, 6, m, 7, 101 + m, True)
        direct = np.zeros_like(coefs)
        for b in range(len(selected)):
            active = []
            for k in range(6):
                choices = candidates[b, k][candidates[b, k] >= 0]
                rss = []
                for j in choices:
                    columns = active + [int(j)]
                    beta = np.linalg.lstsq(xc[:, columns], yc, rcond=None)[0]
                    rss.append(np.sum((yc - xc[:, columns] @ beta)**2))
                expected = int(choices[np.argmin(rss)])
                if expected != selected[b, k]:
                    raise AssertionError((m, b, k, expected, selected[b, k]))
                active.append(expected)
                beta = np.linalg.lstsq(xc[:, active], yc, rcond=None)[0]
                direct[k + 1, active] += beta / len(selected)
        error = float(np.max(np.abs(direct - coefs)))
        maximum_error = max(maximum_error, error)
        np.testing.assert_allclose(coefs, direct, atol=2e-11, rtol=2e-11)
        intercepts = ym - coefs @ xm
        np.testing.assert_allclose((X @ coefs.T + intercepts).mean(axis=0), ym,
                                   atol=1e-11)
    return {"maximum_coefficient_error": maximum_error,
            "checked_m": [1, 3, 7, 12], "checked_independent_paths": 22}


def run_replicate(rep, scenario, directory):
    config_json = json.dumps(CONFIG, sort_keys=True)
    snr = CONFIG["snrs"][scenario]
    path = Path(directory) / f"correlated_snr{snr:g}_rep{rep:04d}.npz"
    if path.exists():
        with np.load(path) as saved:
            if str(saved["config_json"]) != config_json:
                raise RuntimeError(f"Existing result has a different configuration: {path}")
        return {"replicate": rep, "snr": snr, "cached": True}
    started = time.monotonic()
    n, p, rho, kmax, B = (CONFIG[x] for x in ("n", "p", "rho", "kmax", "ensemble_size"))
    data_seed = seed_for(rep, scenario, 0)
    rng = np.random.default_rng(data_seed)
    X = rng.normal(size=(n, p))
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * X[:, j]
    beta = np.zeros(p)
    beta[:CONFIG["sparsity"]] = 1.0
    Sigma = rho**np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    sigma2 = float(beta @ Sigma @ beta / snr)
    truth = X @ beta
    noise = rng.normal(scale=np.sqrt(sigma2), size=n)
    y = truth + noise
    fold_rng = np.random.default_rng(seed_for(rep, scenario, 1))
    folds = np.array_split(fold_rng.permutation(n), CONFIG["folds"])
    cv_losses = np.zeros((len(CONFIG["m_grid"]), kmax))
    all_indices = np.arange(n)
    for fold, validation in enumerate(folds):
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        G, g, xm, ym = centered_stats(X[training], y[training])
        xv, yv = X[validation] - xm, y[validation] - ym
        validation_G = xv.T @ xv / len(validation)
        validation_g = xv.T @ yv / len(validation)
        validation_y2 = np.mean(yv**2)
        for mi, m in enumerate(CONFIG["m_grid"]):
            coefficients = fit_gram(G, g, kmax, m, B,
                                    seed_for(rep, scenario, 2, fold, mi))[1:]
            losses = (validation_y2 - 2 * coefficients @ validation_g
                      + np.einsum("ki,ij,kj->k", coefficients, validation_G, coefficients))
            cv_losses[mi] += losses / CONFIG["folds"]
    # Archive convention: smallest m wins ties within k, then smallest k.
    best_m_by_k = np.argmin(cv_losses, axis=0)
    selected_k = int(np.argmin(cv_losses[best_m_by_k, np.arange(kmax)])) + 1
    selected_m_index = int(best_m_by_k[selected_k - 1])
    selected_m = CONFIG["m_grid"][selected_m_index]
    G, g, xm, ym = centered_stats(X, y)
    fs = fit_gram(G, g, kmax, p, 1, seed_for(rep, scenario, 3))
    efs = fit_gram(G, g, kmax, selected_m, B, seed_for(rep, scenario, 4))
    coefficients = np.stack([fs, efs])
    intercepts = ym - coefficients @ xm
    metrics = np.empty((2, kmax, len(METRICS)))
    for method in range(2):
        predictions = X @ coefficients[method, 1:].T + intercepts[method, 1:]
        train_mse = np.mean((y[:, None] - predictions)**2, axis=0)
        risk = np.mean((truth[:, None] - predictions)**2, axis=0)
        df_opt = n * (risk - train_mse + sigma2) / (2 * sigma2)
        df_cov = np.sum(noise[:, None] * (predictions - truth[:, None]), axis=0) / sigma2
        metrics[method] = np.column_stack([train_mse, risk, df_opt, df_cov])
    if selected_m == p:
        np.testing.assert_array_equal(efs, fs)
    np.testing.assert_allclose(metrics[1, :, 2] - metrics[0, :, 2],
                               metrics[1, :, 3] - metrics[0, :, 3], atol=1e-10)
    elapsed = time.monotonic() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.npz")
    np.savez_compressed(temporary, metrics=metrics, coefficients=coefficients,
                        intercepts=intercepts, cv_losses=cv_losses,
                        selected_k=selected_k, selected_m=selected_m, sigma2=sigma2,
                        replicate=rep, scenario=scenario, snr=snr,
                        data_seed=np.uint64(data_seed), elapsed_seconds=elapsed,
                        config_json=config_json)
    os.replace(temporary, path)
    return {"replicate": rep, "snr": snr, "selected_k": selected_k,
            "selected_m": selected_m, "seconds": round(elapsed, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--scenario", type=int, choices=range(3), default=None)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    load_kernel()
    if args.validate:
        print(json.dumps(validate_kernel()), flush=True)
        return
    directory = args.output / ("benchmark" if args.benchmark else "replicates")
    scenarios = [args.scenario] if args.scenario is not None else list(range(3))
    if args.benchmark:
        print(json.dumps(validate_kernel()), flush=True)
        print(json.dumps(run_replicate(0, scenarios[0], directory)), flush=True)
        return
    if not 1 <= args.workers <= 4:
        raise ValueError("Use between one and four workers")
    args.output.mkdir(parents=True, exist_ok=True)
    validation = validate_kernel()
    (args.output / "correlated_validation.json").write_text(json.dumps(validation, indent=2))
    manifest = dict(config=CONFIG, replicates_per_snr=args.replicates,
                    workers=args.workers, scenarios=scenarios, status="running",
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, numpy=np.__version__, platform=platform.platform(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    kernel_sha256=hashlib.sha256((HERE / "gram_fs.cpp").read_bytes()).hexdigest(),
                    blas_threads=1, metrics=list(METRICS),
                    plotted_df="df_covariance", interval="pointwise paired/marginal t 95%")
    manifest_path = args.output / "correlated_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    jobs = [(rep, scenario, directory) for rep in range(args.replicates) for scenario in scenarios]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_replicate, *job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, datasets=len(jobs))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"complete": len(jobs), "seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
