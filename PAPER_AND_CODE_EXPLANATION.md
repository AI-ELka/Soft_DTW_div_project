# Paper + Code Reading Notes (Soft-DTW Divergences)

This note summarizes the main ideas of `paper.pdf` (*Differentiable Divergences Between Time Series*, Blondel–Mensch–Vert, AISTATS 2021 / arXiv:2010.08354) and how they map to the implementation in `sdtw_div/numba_ops.py`.

## 1) Paper in a nutshell

### DTW
Given time series `X = (x_i)_{i=1..m} in R^{m x d}` and `Y = (y_j)_{j=1..n} in R^{n x d}`, define a ground cost matrix `C(X,Y) in R^{m x n}`, typically:

`C_ij = 0.5 * ||x_i - y_j||_2^2`.

Dynamic time warping (DTW) is
`dtw(C) = min_{A in A(m,n)} <A, C>`,
where \(\mathcal{A}(m,n)\) is the set of monotone alignment-path matrices. DTW works well but is non-smooth (min over paths), making it tricky as a gradient-based loss.

### soft-DTW (Cuturi & Blondel 2017)
soft-DTW replaces the min by a soft-min (log-sum-exp) controlled by `gamma > 0`:

`sdtw_gamma(C) = -gamma * log( sum_{A in A(m,n)} exp( - <A, C> / gamma ) )`.

It is differentiable and still computable by dynamic programming in \(\mathcal{O}(mn)\).

Crucially, its gradient w.r.t. the cost matrix is an expected alignment:
`grad_C sdtw_gamma(C) = E_gamma(C)`,
where \(E_\gamma(C)\) is the expectation of \(A\) under the corresponding Gibbs distribution over alignments.

### The key issue: soft-DTW is biased as a loss
For the common squared Euclidean cost, the paper shows:
- `sdtw_gamma(C(X,Y))` can be **negative**.
- More importantly, for `gamma > 0`, `sdtw_gamma(C(X,Y))` is **not minimized at X=Y** (Proposition 2). If you solve `argmin_X sdtw_gamma(C(X,Y))` starting from `X=Y`, the solution typically moves away from `Y` (“denoising” / entropic bias).

### The fix: debiased divergences (Sinkhorn-style correction)
They propose the soft-DTW divergence:

`D_gamma(X,Y) = sdtw_gamma(C(X,Y)) - 0.5*sdtw_gamma(C(X,X)) - 0.5*sdtw_gamma(C(Y,Y))`.

By construction `D_gamma(X,X)=0`. Under certain costs (proved for specific kernel-induced costs; empirically supported/conjectured for squared Euclidean), it is non-negative and equals 0 iff `X=Y`.

They also define a “sharp” variant:
- `sharp_gamma(C) = <E_gamma(C), C>` (removes the entropy term from the variational view of soft-DTW),
- and the corresponding debiased “sharp divergence” `S_gamma(X,Y)` using the same correction terms.

Limits:
- `gamma -> 0`: DTW-like behavior.
- `gamma -> +infty`: relates to mean cost over all paths (and can diverge depending on lengths).

## 2) How the code implements the paper (`sdtw_div/numba_ops.py`)

The file `sdtw_div/numba_ops.py` contains the dynamic programming recurrences and derivatives used in the paper’s methods and experiments.

### Forward DP: computing soft-DTW
- `sdtw_C(C, gamma=..., return_all=...)` computes \(\mathrm{sdtw}_\gamma(C)\).
- Internally, `_sdtw_C(C, V, P)` fills a DP table `V` and stores transition probabilities in `P`.
- `_soft_min_argmin(a,b,c)` implements the local 3-way soft-min used in the recursion (left / diag / up).

### Backward DP: computing expected alignment (gradient wrt cost)
- `sdtw_grad_C(P, ...)` returns `E_gamma(C) = grad_C sdtw_gamma(C)`.
- Internally, `_sdtw_grad_C(P, E)` is the backward recursion.

### Chain rule to get gradient wrt time-series values
- `sdtw_value_and_grad(X, Y, gamma=...)` computes value and `grad_X sdtw_gamma(C(X,Y))` by:
  1) building the cost matrix,
  2) calling `sdtw_value_and_grad_C(C, ...)`,
  3) applying a VJP from cost-space back to \(X\): `squared_euclidean_cost_vjp`.

### “Sharp” soft-DTW and Hessian-vector product
The sharp variant needs a Hessian-vector product. The implementation provides:
- `sdtw_directional_derivative_C(...)`
- `sdtw_hessian_product_C(...)`
which are computed with additional DP passes.

### Debiasing into divergences (the main paper contribution)
- `_divergence(func, X, Y)` implements:
  `L(X,Y) - 0.5*L(X,X) - 0.5*L(Y,Y)`.
- `sdtw_div(X, Y, gamma=...)` corresponds to \(D_\gamma(X,Y)\).
- `sharp_sdtw_div(X, Y, gamma=...)` corresponds to \(S_\gamma(X,Y)\).
- `mean_cost_div(...)` is the analogous correction for mean-cost.

### Barycenters / averaging (paper experiments)
- `barycenter(Ys, X_init, value_and_grad=..., ...)` solves:
  \[
  \min_X \sum_i w_i\,\mathcal{L}(X,Y_i)
  \]
  using L-BFGS, where \(\mathcal{L}\) can be `sdtw_value_and_grad`, `sdtw_div_value_and_grad`, etc.

## 3) Mini-project proposal options (pick one)

### Option A — Bias demonstration + averaging (recommended)
Goal: reproduce the core message on real time series.
- Show that minimizing plain `sdtw_gamma(X,Y)` from `X=Y` yields `X_hat != Y`.
- Show that minimizing `D_gamma(X,Y)` stays at `X=Y` (and yields `D_gamma(Y,Y)=0`).
- Compute barycenters of a small set of real series under:
  - Euclidean mean
  - soft-DTW barycenter
  - soft-DTW divergence barycenter
  - sharp divergence barycenter
- Sweep \(\gamma\) and discuss qualitative differences.

### Option B — k-NN classification with divergence distances
Goal: use `D_gamma` as a distance-like quantity for nearest neighbors.
- Compare DTW vs soft-DTW vs \(D_\gamma\) vs \(S_\gamma\).
- Cross-validate \(\gamma\), report accuracy + runtime.

### Option C — Prototype per class (barycenter classifier)
Goal: class templates that respect warping.
- Learn one barycenter per class under each loss.
- Classify by nearest prototype; compare behavior vs Euclidean prototypes.

## 4) Questions to finalize a concrete plan
1) Solo or pair, and do you need theory-heavy or experiment-heavy?
2) Any required dataset source (e.g., UCR), or free choice?
3) Is code reuse allowed/encouraged, or do you need to re-implement parts?
