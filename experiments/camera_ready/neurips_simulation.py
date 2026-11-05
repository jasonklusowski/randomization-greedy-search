#!/usr/bin/env python3
"""Run or summarize the NeurIPS Figure 2 experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import correlated as core
from correlated_summary import summarize

REPLICATES = 1000


def run_jobs(jobs, directory, workers):
    if workers == 1:
        results = (core.run_replicate(*job, directory) for job in jobs)
        report_progress(results, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=core.load_kernel) as pool:
            futures = [pool.submit(core.run_replicate, *job, directory) for job in jobs]
            report_progress((future.result() for future in futures), len(jobs))


def report_progress(results, total):
    for count, _ in enumerate(results, start=1):
        if count % 25 == 0 or count == total:
            print(f"Completed {count}/{total} datasets", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--output", type=Path, default=core.RESULTS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("--replicates must be at least two for Student-t intervals")
    output = args.output.resolve()
    if args.replicates != REPLICATES and output == core.RESULTS.resolve():
        parser.error("use a separate --output directory when changing --replicates")
    directory = output / "replicates"

    jobs = [(rep, scenario) for scenario, snr in enumerate(core.CONFIG["snrs"])
            for rep in range(args.replicates)
            if not (directory / f"correlated_snr{snr:g}_rep{rep:04d}.npz").is_file()]
    if jobs and args.summarize_only:
        parser.error(f"{len(jobs)} checkpoints are missing; run without --summarize-only")
    if jobs:
        print(f"Kernel check: {core.validate_kernel()}", flush=True)
        run_jobs(jobs, directory, args.workers)
    summarize(output, args.replicates)
    print(f"Validated {args.replicates} datasets per SNR and wrote summaries to {output}")


if __name__ == "__main__":
    main()
