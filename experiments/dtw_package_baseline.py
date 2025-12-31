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
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if backend == "dtaidistance":
        try:
            from dtaidistance import dtw as dtw_lib  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'dtaidistance'. Install with `pip install dtaidistance` "
                "or switch backend='tslearn'."
            ) from exc
        return DtwResult(distance=float(dtw_lib.distance(x, y)), backend=backend)

    if backend == "tslearn":
        try:
            from tslearn.metrics import dtw as dtw_fn  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'tslearn'. Install with `pip install tslearn` "
                "or switch backend='dtaidistance'."
            ) from exc
        return DtwResult(distance=float(dtw_fn(x, y)), backend=backend)

    raise ValueError(f"Unknown backend: {backend!r}")


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    y = rng.normal(size=80)

    for backend in ("dtaidistance", "tslearn"):
        try:
            res = dtw_distance(x, y, backend=backend)  # type: ignore[arg-type]
            print(f"DTW distance ({res.backend}) = {res.distance:.6f}")
        except ModuleNotFoundError as e:
            print(f"{backend}: {e}")


if __name__ == "__main__":
    main()

