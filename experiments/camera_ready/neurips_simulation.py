#!/usr/bin/env python3
"""Run or summarize the 500-dataset-per-SNR NeurIPS Figure 2 experiment."""
from __future__ import annotations

import correlated as core  # Set numerical-library thread counts before importing NumPy.
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from neurips_snr005 import CONFIG, ORIGINAL_CONFIG, initialize_worker, sha256

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results"
LOW_SNR_ARCHIVE = ARCHIVE / "neurips_snr005"
OUTPUT = ARCHIVE / "neurips_500"
ARCHIVE_REPLICATES = 200
REPLICATES = 500


def checkpoint(directory, rep, scenario):
    snr = CONFIG["snrs"][scenario]
    return directory / "replicates" / f"correlated_snr{snr:g}_rep{rep:04d}.npz"


def archived_checkpoint(rep, scenario):
    if scenario == 0:
        return checkpoint(LOW_SNR_ARCHIVE, rep, scenario), CONFIG
    return checkpoint(ARCHIVE, rep, scenario), ORIGINAL_CONFIG


def run_jobs(jobs, output, workers):
    if workers == 1:
        for count, (rep, scenario) in enumerate(jobs, 1):
            core.run_replicate(rep, scenario, output / "replicates")
            if count % 25 == 0 or count == len(jobs):
                print(f"Completed {count}/{len(jobs)} new datasets", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = [
            pool.submit(core.run_replicate, rep, scenario, output / "replicates")
            for rep, scenario in jobs
        ]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 25 == 0 or count == len(jobs):
                print(f"Completed {count}/{len(jobs)} new datasets", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fresh", action="store_true", help="Do not use archived checkpoints.")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("--replicates must be at least two for Student-t intervals")
    output = args.output.resolve()
    if output in {ARCHIVE.resolve(), LOW_SNR_ARCHIVE.resolve()}:
        parser.error("--output must be separate from the historical results")
    output.mkdir(parents=True, exist_ok=True)

    sources = {}
    jobs = []
    reused_archive = {}
    for scenario in range(len(CONFIG["snrs"])):
        for rep in range(args.replicates):
            key = (rep, scenario)
            path = checkpoint(output, rep, scenario)
            sources[key] = (path, CONFIG)
            if path.is_file():
                continue
            if not args.fresh and rep < ARCHIVE_REPLICATES:
                saved, config = archived_checkpoint(rep, scenario)
                if saved.is_file():
                    sources[key] = (saved, config)
                    reused_archive[str(saved.relative_to(HERE))] = sha256(saved)
                    continue
            jobs.append(key)
    if args.summarize_only and jobs:
        parser.error(f"{len(jobs)} checkpoints are missing; run without --summarize-only")

    manifest = {
        "status": "running",
        "config": CONFIG,
        "replicates_per_snr": args.replicates,
        "datasets": len(sources),
        "workers": args.workers,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "blas_threads": 1,
        "interval": f"pointwise marginal/paired Student-t 95%, df={args.replicates - 1}",
        "source_sha256": {name: sha256(HERE / name) for name in (
            "neurips_simulation.py", "neurips_snr005.py", "correlated.py",
            "gram_fs.cpp",
        )},
        "archived_checkpoints": len(reused_archive),
        "archived_checkpoint_sha256": reused_archive,
        "resumed_checkpoints": len(sources) - len(reused_archive) - len(jobs),
        "new_datasets": len(jobs),
    }
    manifest_path = output / "manifest.json"
    started = time.monotonic()
    if not args.summarize_only:
        initialize_worker()
        manifest["kernel_validation"] = core.validate_kernel()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Using {len(sources) - len(jobs)} saved datasets; fitting {len(jobs)} new datasets",
              flush=True)
        run_jobs(jobs, output, args.workers)

    from correlated_plot import summarize

    manifest["source_sha256"]["correlated_plot.py"] = sha256(HERE / "correlated_plot.py")
    summarize(output, args.replicates, source_for=lambda rep, scenario: sources[rep, scenario],
              config=CONFIG)
    for filename, checksum in reused_archive.items():
        if sha256(HERE / filename) != checksum:
            raise RuntimeError(f"Archived checkpoint changed during the run: {filename}")
    if not args.summarize_only:
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started, validation="passed")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated and summarized {len(sources)} datasets in {output}", flush=True)


if __name__ == "__main__":
    main()
