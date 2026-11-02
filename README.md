# Understanding Randomization in Greedy Model Search

Code and synthetic results for the paper by Xin Chen, Jason M. Klusowski,
Yan Shuo Tan, and Chang Yu.

This repository contains the simulation and plotting code for Figures 1 and 2,
the saved simulation results, and the calculations used for the uncertainty
statements in Appendix A. Figure 2 uses 200 independent datasets at each SNR
in `{0.05, 0.25, 1.0}`. The 200 older records at SNR `0.053` are retained for
provenance and are excluded from the current figure.

## Installation

Use Python 3.12 and a C++17 compiler available as `clang++`. The simulation
script compiles its C++ kernel automatically. No compiled binary is supplied.
Run the following commands from a terminal.

```sh
git clone https://github.com/jasonklusowski/randomization-greedy-search.git
cd randomization-greedy-search
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/camera_ready/requirements.txt
```

All subsequent commands run from the repository root. The requirements file
pins the Python package versions used for reproduction. The simulations use
the CPU and do not require a GPU or external data. Numerical results can
differ slightly across compilers and numerical libraries.

## Redraw the paper figures

```sh
python experiments/camera_ready/compute_weights.py
python experiments/camera_ready/neurips_figure1.py
python experiments/camera_ready/neurips_figure2.py
```

The figure outputs are `neurips/figures/figure1.pdf` and
`neurips/figures/figure2.pdf`. These commands do not repeat the simulation fits.

`compute_weights.py` evaluates Theorem 3.1 and its logistic approximation to
regenerate all 120 Figure 1 coordinates. It checks the exact weights against
an independent positive recurrence and the logistic bounds. The computed
coordinates agree with those recovered from the original vector PDF within
`1.8e-9`, reflecting rounding in the PDF. It writes
`figure1_theory_curves.csv` and `figure1_theory_validation.json` under
`experiments/camera_ready/results/`.

The plotting script retains the coordinates recovered from the original
PDF to preserve the published figure. That PDF, the recovered coordinates,
and the extraction checks are included. The dashed curves use midpoint
`h(alpha,k)+1/2`, as in the current paper. Figure 2 uses the saved means and
pointwise 95% Monte Carlo intervals for the 600 current datasets.

## Check the saved results

```sh
python experiments/camera_ready/compute_weights.py --check
python experiments/camera_ready/correlated.py --validate
python experiments/camera_ready/validate_results.py
python experiments/camera_ready/neurips_snr005.py --summarize-only
```

The first command checks Figure 1 without writing files. The next two
compare the numerical kernel with direct least squares and check the original
600 records and their summary statistics. The last command rebuilds the
current summaries from the 200 records at SNR
`0.05` and the 400 records at SNR `0.25` and `1.0`. It also checks the selected
cross validation parameters and the identities relating paired differences
in training error, prediction error, and degrees of freedom.

These commands update validation reports and summary files. They do not
rerun the simulation fits. Paired intervals and the evidence for the claims
in Appendix A are in `experiments/camera_ready/results/neurips_snr005/`.

## Repeat all simulation fits

Use a separate clone so the supplied results are preserved. In that clone,
move the supplied results aside before running the experiment.

```sh
mv experiments/camera_ready/results saved_results
python experiments/camera_ready/correlated.py --validate
python experiments/camera_ready/correlated.py --replicates 200 --workers 3
python experiments/camera_ready/correlated_plot.py --replicates 200
python experiments/camera_ready/validate_results.py
python experiments/camera_ready/neurips_snr005.py --workers 3
python experiments/camera_ready/neurips_figure2.py
python experiments/camera_ready/neurips_figure1.py --extract-source images/wj_vs_gamma_logistic.pdf
python experiments/camera_ready/compute_weights.py
```

This reproduces the original 600 fits at SNRs `{0.053, 0.25, 1.0}`, then the
200 fits at SNR `0.05` used in the current paper. The latter step reuses the
400 fits at the two higher SNRs. The final two commands repeat the extraction
and plotting of Figure 1 from the included source PDF, then independently
regenerate its mathematical curves and compare the coordinates.

Keep the default output paths for this sequence. The validator and the SNR
`0.05` script read the original records from
`experiments/camera_ready/results/`. Runs resume from completed replicate
files after checking their configurations. The scripts permit one to four
workers. Seeds and simulation settings are fixed in the scripts and recorded
with the results.

## Files

- `experiments/camera_ready/correlated.py` generates data, tunes the estimators,
  and records the original simulation fits.
- `experiments/camera_ready/gram_fs.cpp` implements the FS and EFS fitting kernel.
- `experiments/camera_ready/neurips_snr005.py` runs the current SNR `0.05`
  experiment and combines its results with the higher SNR records.
- `experiments/camera_ready/correlated_plot.py` calculates summaries and
  intervals for the original experiment.
- `experiments/camera_ready/compute_weights.py` evaluates the Figure 1 curves
  from the theory and checks the published coordinates.
- `experiments/camera_ready/neurips_figure1.py` and `neurips_figure2.py` draw
  the figures used in the paper.
- `experiments/camera_ready/validate_results.py` checks the numerical kernel,
  saved records, and summaries independently.
- `experiments/camera_ready/results/` contains raw records, summaries, seeds,
  software information, source hashes, and validation reports.

The [experiment description](experiments/camera_ready/correlated_README.md)
gives the estimator, tuning grid, metrics, and uncertainty calculations.
The [reproduction notes](experiments/camera_ready/README.md) describe the
saved outputs and how the current results relate to the earlier experiment.

## License

The code is distributed under the [MIT license](LICENSE).
