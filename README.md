Differentiable Divergences between Time Series
==============================================

An implementation of soft-DTW divergences.

Example
-------

```python
import numpy as np
from sdtw_div.numba_ops import sdtw_div, sdtw_div_value_and_grad

# Two 3-dimensional time series of lengths 5 and 4, respectively.
X = np.random.randn(5, 3)
Y = np.random.randn(4, 3)

# Compute the divergence value. The parameter gamma controls the regularization strength. 
value = sdtw_div(X, Y, gamma=1.0)

# Compute the divergence value and the gradient w.r.t. X.
value, grad = sdtw_div_value_and_grad(X, Y, gamma=1.0)
```
Similarly, we can use `sharp_sdtw_div`, `sharp_sdtw_div_value_and_grad`,
`mean_cost_div` and `mean_cost_div_value_and_grad`.

Project structure
-----------------

- Core library: `sdtw_div/`
  - Soft-DTW and divergences: `sdtw_div/numba_ops.py`
  - Optional DTW baseline wrapper: `sdtw_div/dtw_baseline.py`
  - Airline Passengers experiments: `sdtw_div/airline_experiments.py`
- Report template: `task.tex`
- Figures/results are written to `figures/` and `results/` when running experiments.

Setup
-----

Required:
- Python 3
- `numpy`, `scipy`, `matplotlib`

Optional (speed):
- `numba` (for JIT acceleration)

Optional (DTW baseline):
- `dtaidistance` or `tslearn`

Steps to run experiments (Airline Passengers)
---------------------------------------------

1) Run the experiment pipeline:
```bash
python3 - <<'PY'
from sdtw_div.airline_experiments import run_airline_experiments
run_airline_experiments()
PY
```

2) Inspect outputs:
- Figures in `figures/` (barycenters + bias demo + raw/normalized data)
- Metrics in `results/airline_metrics.json` and `results/airline_metrics.txt`

3) (Optional) Use a classic DTW baseline (requires a package):
```python
from sdtw_div.dtw_baseline import dtw_distance
import numpy as np

x = np.arange(5)
y = np.arange(7)
print(dtw_distance(x, y, backend="dtaidistance"))
```

Install
--------

Run `python setup.py install` or copy the files to your project.

Reference
----------

> Differentiable Divergences between Time Series <br/>
> Mathieu Blondel, Arthur Mensch, Jean-Philippe Vert <br/>
> [arXiv:2010.08354](https://arxiv.org/abs/2010.08354)
