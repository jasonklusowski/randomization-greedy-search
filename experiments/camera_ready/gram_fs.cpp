#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Independent randomized forward-selection paths with OLS refitting.
// G and g are the centered, unnormalized Gram matrix and X'y.
// Optional traces are used only for validation against direct least squares.
extern "C" int efs_gram(const double* G, const double* g, int p, int kmax,
                        int m, int ensemble_size, std::uint64_t seed,
                        double* output, int* selected_trace,
                        int* candidate_trace) {
    if (p < 1 || kmax > p || m < 1 || m > p || ensemble_size < 1) return 1;
    std::fill(output, output + (kmax + 1) * p, 0.0);
    const int B = m == p ? 1 : ensemble_size;
    std::mt19937_64 rng(seed);
    std::vector<double> projections(kmax * p), residual(p), variance(p);
    std::vector<double> response_projection(kmax), beta(kmax);
    std::vector<int> remaining(p), selected(kmax);
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < p; ++j) {
            remaining[j] = j;
            residual[j] = g[j];
            variance[j] = G[j * p + j];
        }
        int nremaining = p;
        for (int k = 0; k < kmax; ++k) {
            const int ncandidates = std::min(m, nremaining);
            // A partial Fisher-Yates shuffle gives a uniform candidate subset.
            if (ncandidates < nremaining) {
                for (int r = 0; r < ncandidates; ++r) {
                    std::uniform_int_distribution<int> draw(r, nremaining - 1);
                    std::swap(remaining[r], remaining[draw(rng)]);
                }
            }
            int best_pos = -1;
            double best_score = -1.0;
            for (int r = 0; r < ncandidates; ++r) {
                const int j = remaining[r];
                if (candidate_trace) candidate_trace[(b * kmax + k) * p + r] = j;
                if (variance[j] <= 1e-12 * G[j * p + j]) continue;
                const double score = residual[j] * residual[j] / variance[j];
                if (score > best_score || (score == best_score &&
                    (best_pos < 0 || j < remaining[best_pos]))) {
                    best_score = score;
                    best_pos = r;
                }
            }
            if (best_pos < 0) return 2;
            const int j = remaining[best_pos];
            selected[k] = j;
            if (selected_trace) selected_trace[b * kmax + k] = j;
            const double norm = std::sqrt(variance[j]);
            response_projection[k] = residual[j] / norm;
            double* current = projections.data() + k * p;
            for (int i = 0; i < p; ++i) {
                double inner = G[i * p + j];
                for (int t = 0; t < k; ++t)
                    inner -= projections[t * p + i] * projections[t * p + j];
                current[i] = inner / norm;
            }
            for (int i = 0; i < p; ++i) {
                residual[i] -= current[i] * response_projection[k];
                variance[i] = std::max(0.0, variance[i] - current[i] * current[i]);
            }
            remaining[best_pos] = remaining[nremaining - 1];
            --nremaining;
            // Solve R beta = Q'y to obtain the refitted coefficients at this k.
            for (int r = k; r >= 0; --r) {
                double value = response_projection[r];
                for (int t = r + 1; t <= k; ++t)
                    value -= projections[r * p + selected[t]] * beta[t];
                beta[r] = value / projections[r * p + selected[r]];
                output[(k + 1) * p + selected[r]] += beta[r] / B;
            }
        }
    }
    return 0;
}
