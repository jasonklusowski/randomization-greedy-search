# Reproducing NeurIPS Figures 1 and 2

To regenerate the theoretical curves and redraw the NeurIPS figures,
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
stated in the current NeurIPS caption. Figure 2 uses all saved means and
pointwise intervals in
`results/neurips_snr005/correlated_marginal_summary.csv`, with SNRs
`{0.05, 0.25, 1.0}`.

## Repeating the Figure 2 experiment

The current NeurIPS comparison uses 200 independent datasets at each SNR in
`{0.05, 0.25, 1.0}`. The SNR 0.05 fits were rerun with the original scenario 0
seeds, changing only the noise scale. The 400 fits at SNR 0.25 and 1.0 were
reused without changes. The figure reports training MSE and covariance degrees
of freedom with pointwise 95% Monte Carlo confidence intervals.

Install Python 3.12 and the packages in `requirements.txt`. A C++17 compiler
must be available as `clang++`. Run these commands from the repository root.

```sh
python -m pip install -r experiments/camera_ready/requirements.txt
python experiments/camera_ready/correlated.py --validate
python experiments/camera_ready/correlated.py --replicates 200 --workers 3
python experiments/camera_ready/correlated_plot.py --replicates 200
python experiments/camera_ready/validate_results.py
python experiments/camera_ready/neurips_snr005.py --workers 3
python experiments/camera_ready/neurips_figure2.py
```

The run resumes from completed replicate files after checking their
configurations. To rerun all fits from scratch, make a separate copy of the
repository, move its supplied `experiments/camera_ready/results/`
folder outside that copy, and run the commands above with the default output
paths. This preserves the supplied results while rebuilding the 600 original
records and then the 200 SNR 0.05 records. The validator and SNR 0.05 script
always read the original records from `experiments/camera_ready/results/`.

To rerun only SNR 0.05, retain the supplied results folder and pass a new
`--output` folder to `neurips_snr005.py`. This reuses the 400 records at SNR
0.25 and 1.0 in the default results folder. Then pass
`correlated_marginal_summary.csv` from the new folder to
`neurips_figure2.py --summary`.
All settings and random seeds are fixed in the scripts and saved configurations.

The original three panels at SNRs `{0.053, 0.25, 1.0}` are written to
`experiments/camera_ready/results/` by `correlated_plot.py`.
Their filenames begin `correlated_mse_vs_df_by_k_`. They are retained for
comparison with the earlier experiment. The current NeurIPS paper uses
`neurips/figures/figure2.pdf`.

See `correlated_README.md` for the estimator, parameter grid, metrics,
uncertainty calculations, and validation checks. The implementation uses
independent base learners and consistent intercept reconstruction for FS and
EFS. There is no additional resampling of the frequencies of selected sets.

## Saved evidence

`results/neurips_snr005/correlated_replicates.csv` contains metrics for the 600
current datasets. The 200 new raw files are in
`results/neurips_snr005/replicates/`, and the 400 reused raw files are in
`results/replicates/`. They preserve coefficient paths, intercepts,
cross validation scores, selected parameters, and seeds. The original SNR
0.053 records remain as provenance and are excluded from the NeurIPS summaries.
Marginal and paired CSVs in `results/neurips_snr005/` contain means, Monte Carlo
standard errors, and pointwise confidence limits. Paired differences use FS
and EFS fits to the same dataset.
The original `correlated_plot.py` generates the three separate Figure 2 panels.
The NeurIPS plotting script combines them with a shared legend and marks
`k=10` without changing the estimates or intervals.

The run manifest records computation time, software versions, and simulation
source hashes. `validate_results.py` independently checks the numerical kernel
against direct least squares, recomputes the summary statistics from all raw
replicates, and verifies CV selections and paired identities. Its report is
saved as `results/independent_validation.json` with current validation source
hashes.
The current `results/neurips_snr005/manifest.json` and `validation.json`
record the rerun, seed reuse, original file hashes, unchanged higher SNR
summaries, CV checks, and paired identities.
