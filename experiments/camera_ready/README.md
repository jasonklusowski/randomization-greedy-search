# Reproducing Figures 1 and 2

To regenerate the theoretical curves and redraw the figures,
install the packages in `requirements.txt` and run from the repository root.

```sh
python experiments/camera_ready/compute_weights.py
python experiments/camera_ready/neurips_figure1.py
python experiments/camera_ready/neurips_figure2.py
```

The first command evaluates Theorem 3.1 and its logistic approximation,
checks the weights against an independent positive recurrence and the
logistic bounds, and writes `results/figure1_theory_curves.csv` and
`results/figure1_theory_validation.json`. All 120 computed coordinates agree
with those recovered from the original vector PDF within `1.8e-9`, reflecting
rounding in the PDF. To perform the checks without writing files, use
`python experiments/camera_ready/compute_weights.py --check`.

The plotting commands write `neurips/figures/figure1.pdf` and
`neurips/figures/figure2.pdf`. They do not rerun the simulations. Figure 1
retains all 120 polyline vertices recovered from its original vector PDF,
with extraction provenance and numerical fidelity checks in
`results/figure1_*`. Its dashed curves have midpoint `h(alpha,k)+1/2`, as
stated in the paper. To repeat the extraction from the source PDF, run
`python experiments/camera_ready/neurips_figure1.py --extract-source images/wj_vs_gamma_logistic.pdf`.

The updated Figure 2 uses the saved means and pointwise intervals in
`results/neurips_500/correlated_marginal_summary.csv`, with SNRs
`{0.05, 0.25, 1.0}`.

## Repeating the Figure 2 experiment

The updated comparison uses 500 independent datasets at each SNR in
`{0.05, 0.25, 1.0}`. It reuses the first 200 verified records per SNR and
fits replicates 200–499 with the same settings and seed scheme. The figure
reports training MSE and covariance degrees of freedom with pointwise 95%
Monte Carlo confidence intervals.

Install Python 3.12 and the packages in `requirements.txt`. A C++17 compiler
must be available as `clang++`. Run these commands from the repository root.

```sh
python -m pip install -r experiments/camera_ready/requirements.txt
python experiments/camera_ready/neurips_simulation.py --replicates 500 --workers 4
python experiments/camera_ready/neurips_figure2.py
```

The run resumes from completed replicate files. To validate the saved files
and rebuild their summaries, add `--summarize-only` to the simulation command.
To fit all 1,500 datasets from scratch and redraw the figure in a separate
output directory, run the following commands with a new directory path.

```sh
python experiments/camera_ready/neurips_simulation.py --replicates 500 --workers 4 --fresh --output /tmp/efs-neurips-500
python experiments/camera_ready/neurips_figure2.py --summary /tmp/efs-neurips-500/correlated_marginal_summary.csv --output /tmp/efs-neurips-500/figure2.pdf
```

All settings and random seeds are fixed in the scripts and saved configurations.

See `correlated_README.md` for the estimator, parameter grid, metrics,
uncertainty calculations, and validation checks. The implementation uses
independent base learners and consistent intercept reconstruction for FS and
EFS. There is no additional resampling of the frequencies of selected sets.

## Saved evidence

`results/neurips_500/correlated_replicates.csv` contains metrics for all 1,500
datasets. The 900 new raw files are in `results/neurips_500/replicates/`.
The first 200 raw files for SNR 0.05 are in `results/neurips_snr005/replicates/`,
and the first 200 each for SNRs 0.25 and 1.0 are in `results/replicates/`.
They preserve coefficient paths, intercepts, cross-validation scores,
selected parameters, and seeds. Marginal and paired CSVs in
`results/neurips_500/` contain means, Monte Carlo standard errors, and
pointwise confidence limits. Paired differences use FS and EFS fits to the same
dataset. The run manifest records the software versions, seed configuration,
archive file hashes, and validation status.

## Archived 200-dataset experiment

The original run used SNRs `{0.053, 0.25, 1.0}`. The subsequent 200-dataset
comparison reran SNR 0.05 with the original scenario-0 seeds, changing only
the noise scale, and reused the 400 records at the two higher SNRs. To redraw
that figure from its archived summaries, run the following command.

```sh
python experiments/camera_ready/neurips_figure2.py --replicates 200 --summary experiments/camera_ready/results/neurips_snr005/correlated_marginal_summary.csv --output /tmp/efs-figure2-200.pdf
```

To reproduce the archived fits from scratch, use a separate clone and move
its supplied `experiments/camera_ready/results/` aside before running these
commands.

```sh
python experiments/camera_ready/correlated.py --validate
python experiments/camera_ready/correlated.py --replicates 200 --workers 3
python experiments/camera_ready/correlated_plot.py --replicates 200
python experiments/camera_ready/validate_results.py
python experiments/camera_ready/neurips_snr005.py --workers 3
```

`correlated_plot.py` generates the three original panels with filenames
beginning `correlated_mse_vs_df_by_k_`. The original validation report is
`results/independent_validation.json`. The archive in `results/neurips_snr005/`
also includes its run manifest, validation report, and paired intervals.
