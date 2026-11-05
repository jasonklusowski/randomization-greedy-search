# Revised Figure 2: correlated-design comparison

The updated Figure 2 reports uncertainty estimates from 500 independent
datasets per SNR. The simulation implements the estimator defined in the paper.
The earlier raw simulation outputs and exact run configuration were not present
in the manuscript repository or supplementary code archive.

## Run

For the updated figure, run `neurips_simulation.py --replicates 500 --workers 4`
with the supplied records, then `neurips_figure2.py`. The simulation adds 300
datasets per SNR to the first 200 and writes summaries to `results/neurips_500/`.
For the from-scratch commands, see the [reproduction notes](README.md).

The original 200-dataset experiment can be reproduced with the commands below.
It requires Python with NumPy, SciPy, and Matplotlib, plus a C++17 compiler
available as `clang++`. The Python script compiles `gram_fs.cpp` into a temporary
shared library automatically. No compiled binary is required in the repository.

```sh
python experiments/camera_ready/correlated.py --validate
python experiments/camera_ready/correlated.py --benchmark --scenario 0
python experiments/camera_ready/correlated.py --replicates 200 --workers 3
python experiments/camera_ready/correlated_plot.py --replicates 200
python experiments/camera_ready/validate_results.py
```

All outputs go under `experiments/camera_ready/results/`. Runs resume from
completed replicate files, checking that their configuration matches. At most
four workers are permitted and numerical-library thread counts are set to one.
The configuration, source hashes, software versions, validation result, elapsed
time, and seed policy are recorded in the scripts and result manifest.

## Experiment

- 500 independent datasets for each population SNR in `{0.05, 0.25, 1}` in
  the updated figure. The original run used 200 datasets per SNR in
  `{0.053, 0.25, 1}`.
- `n=1000`, `p=100`, Gaussian rows with covariance `Sigma[i,j]=0.5**abs(i-j)`.
- The first ten population coefficients equal one and all others equal zero.
- Noise variance is `beta.T @ Sigma @ beta / SNR`.
- FS and EFS use exactly the same data in each replicate.
- Each EFS fit averages 500 independent randomized FS paths. At each step a
  uniform candidate subset is drawn without replacement from the remaining
  variables. The selected variable maximizes the reduction in residual sum of
  squares and every active model is refitted by least squares.
- Both procedures fit an unpenalized intercept. Each training fold is centered
  separately, and predictions reconstruct the intercept as
  `mean(y_train) - mean(X_train) @ coefficients` for every candidate-set size.
- Ten-fold CV considers `k=1,...,20` and the documented ten-point geometric
  candidate grid. It jointly selects `(k,m)`, then refits the complete path at
  the one selected `m`. Thus the displayed path does not separately retune `m`
  at each displayed `k`. Within each `k`, ties are resolved by the smallest
  candidate-grid index. The best `k` is then selected, choosing the smallest
  `k` if those minima tie.
- Dataset, fold, candidate-selection, and refit seeds are deterministic separate
  streams generated from the base seed, replicate, SNR index, and fit identifiers.

The archived `EFSCV` uses the same joint tuning convention. Its template also
enables repeated resampling of active-set frequencies, which changes the finite
ensemble procedure. This additional resampling is omitted here to implement
the paper's independent base learners. The archive's specialized `m=p` path
also reconstructs its intercept differently from its `m<p` path. The fresh
implementation centers and reconstructs the intercept consistently for both.

## Reported quantities and uncertainty

Training MSE and fixed-design prediction risk are respectively
`mean((y - prediction)**2)` and `mean((f - prediction)**2)`, where `f=X@beta`
is the known true mean at the training design. Prediction risk excludes fresh
observation noise.

The plotted df estimate for each replicate is
`sum((y-f)*(prediction-f))/sigma2`. Its expectation is the covariance degrees of
freedom of the complete fitted procedure, including tuning, intercept fitting,
and ensemble randomness. Designs are regenerated across replicates, so the
reported quantity averages conditional-design df over the design distribution.
The earlier optimism estimator
`n*(prediction_risk-training_mse+sigma2)/(2*sigma2)` is also retained. Their
paired EFS-minus-FS differences agree algebraically and are checked numerically
for every replicate.

Figure 2 displays marginal intervals for training MSE and covariance df.
Marginal and paired intervals are pointwise 95% Student-t intervals for
Monte Carlo means, using replicate standard deviations divided by the square
root of the number of independent datasets. Paired intervals are formed from
within-dataset EFS-minus-FS differences. They include dataset, tuning, and
finite-ensemble variation. Negative paired differences indicate a lower value
for EFS. These are not simultaneous confidence bands.

## Saved outputs

- `replicates/correlated_*.npz`: all replicate metrics, full coefficient and
  intercept paths, CV scores for every `(k,m)`, selected parameters, seeds,
  noise variance, configuration, and runtime.
- `correlated_replicates.csv`: metrics in a long, replicate-level table. The
  selected `m` and `k` columns record EFS tuning and are repeated on the FS rows
  as metadata for the paired dataset.
- `correlated_marginal_summary.csv`: means, standard deviations, Monte Carlo
  standard errors, and pointwise confidence limits for each method and `k`.
- `correlated_paired_summary.csv`: the corresponding paired-difference summaries.
- `correlated_diagnostics.json`: selected-parameter frequencies and consistency
  checks for the two df estimators.
- `correlated_mse_vs_df_*.pdf`: the three original 200-dataset panels with
  marginal uncertainty in both coordinates. The updated combined figure is
  `neurips/figures/figure2.pdf`.
- `independent_validation.json`: direct-least-squares kernel comparisons,
  raw-summary recomputation, CV-selection checks, and paired metric identities.

The kernel validation checks every selected feature against exhaustive
candidate-wise direct least-squares fits under the identical candidate subsets.
It also compares averaged coefficients and intercept reconstruction, including
the `m=1`, `m=p`, and `m` exceeding the remaining-feature count cases.
