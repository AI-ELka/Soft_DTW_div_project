from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DTWBackend = Literal["dtaidistance", "tslearn"]


@dataclass(frozen=True)
class DtwResult:
    distance: float
    backend: DTWBackend


def dtw_distance(x: np.ndarray, y: np.ndarray, backend: DTWBackend = "dtaidistance") -> DtwResult:
    """Compute classic (hard) DTW distance using an external package."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if backend == "dtaidistance":
        try:
            from dtaidistance import dtw as dtw_lib  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'dtaidistance'. Install with `pip install dtaidistance` "
                "or call dtw_distance(..., backend='tslearn')."
            ) from exc
        return DtwResult(distance=float(dtw_lib.distance(x, y)), backend=backend)

    if backend == "tslearn":
        try:
            from tslearn.metrics import dtw as dtw_fn  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'tslearn'. Install with `pip install tslearn` "
                "or call dtw_distance(..., backend='dtaidistance')."
            ) from exc
        return DtwResult(distance=float(dtw_fn(x, y)), backend=backend)

    raise ValueError(f"Unknown backend: {backend!r}")

