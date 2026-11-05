# Understanding Randomization in Greedy Model Search

Code and synthetic results for the paper by Xin Chen, Jason M. Klusowski,
Yan Shuo Tan, and Chang Yu.

## Install

Use Python 3.12 and a C++17 compiler available as `clang++`. From the repository
root, create an environment and install the pinned dependencies.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/camera_ready/requirements.txt
```

## Redraw the figures

```sh
python experiments/camera_ready/neurips_figure1.py
python experiments/camera_ready/neurips_figure2.py
```

These commands write `neurips/figures/figure1.pdf` and
`neurips/figures/figure2.pdf`. Figure 1 uses the theoretical curves computed in
`compute_weights.py`. Figure 2 uses the saved marginal summaries.

## Verify the simulation results

```sh
python experiments/camera_ready/neurips_simulation.py --summarize-only
```

This validates the 3,000 replicate files in
`experiments/camera_ready/results/neurips_1000/replicates/` and rebuilds
`correlated_marginal_summary.csv` and `correlated_paired_summary.csv` in
`experiments/camera_ready/results/neurips_1000/` using
`correlated_summary.py`.

The simulation uses 1,000 independent datasets at each SNR in
`{0.05, 0.25, 1.0}`, with `n=1000`, `p=100`, Gaussian AR(1) features
with correlation parameter 0.5, and 10 unit coefficients. EFS averages 500
randomized FS paths per fit. Ten-fold cross-validation selects `(k,m)`, then
EFS refits the full path through `k=20` at the selected `m`. Both methods
fit an intercept and use the same data within each replicate.

The figure shows pointwise 95% Monte Carlo Student-t intervals for mean
training MSE and covariance degrees of freedom. The paired summaries apply
the same interval calculation to the EFS minus FS differences within each
dataset and include prediction error.

## Rerun the simulation

Run the following commands from the repository root with a new output directory.
The fit code in `correlated.py` compiles `gram_fs.cpp` automatically and uses
CPU workers.

```sh
python experiments/camera_ready/neurips_simulation.py --workers 4 --output /tmp/efs-paper-results
python experiments/camera_ready/neurips_figure2.py --summary /tmp/efs-paper-results/correlated_marginal_summary.csv --output /tmp/efs-paper-results/figure2.pdf
```

## License

The code is distributed under the [MIT license](LICENSE).
