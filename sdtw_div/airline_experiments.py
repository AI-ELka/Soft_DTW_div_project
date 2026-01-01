from __future__ import annotations

import csv
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np

from sdtw_div.numba_ops import (
    barycenter,
    euclidean_mean,
    sdtw,
    sdtw_div,
    sdtw_div_value_and_grad,
    sdtw_value_and_grad,
    sharp_sdtw_div,
    sharp_sdtw_div_value_and_grad,
)


AIRLINE_PASSENGERS_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)


@dataclass(frozen=True)
class AirlineDataset:
    passengers: np.ndarray  # shape (T,)
    months: list[str]  # length T


def download_airline_passengers_csv(dest: str | Path, url: str = AIRLINE_PASSENGERS_URL) -> Path:
    """Download the Airline Passengers CSV to `dest`."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r:
        data = r.read()
    dest_path.write_bytes(data)
    return dest_path


def load_airline_passengers(csv_path: str | Path) -> AirlineDataset:
    months: list[str] = []
    passengers: list[float] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            months.append(row["Month"])
            passengers.append(float(row["Passengers"]))
    return AirlineDataset(passengers=np.asarray(passengers, dtype=float), months=months)


def zscore(x: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(x))
    std = float(np.std(x) + eps)
    return (x - mean) / std, mean, std


def year_blocks(x: np.ndarray, months: list[str]) -> tuple[np.ndarray, list[str]]:
    """Reshape a monthly series into yearly blocks.

    Returns:
      blocks: shape (n_years, 12)
      years: list of year strings, length n_years
    """
    if len(x) % 12 != 0:
        raise ValueError("Expected a whole number of years (12 months per year).")
    years = [m.split("-")[0] for m in months[::12]]
    blocks = x.reshape(-1, 12)
    return blocks, years


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mean_obj(obj, X: np.ndarray, Ys: list[np.ndarray]) -> float:
    return float(np.mean([obj(X, Y) for Y in Ys]))


def run_airline_experiments(
    *,
    data_csv: str | Path = "data/airline-passengers.csv",
    output_figures: str | Path = "figures",
    output_results: str | Path = "results",
    gamma: float = 1.0,
    years_range: tuple[str, str] = ("1951", "1958"),
    bias_year: str = "1957",
) -> dict:
    """Run a small real-data experiment (bias + barycenters) and save outputs."""
    data_csv = Path(data_csv)
    if not data_csv.exists():
        download_airline_passengers_csv(data_csv)

    dataset = load_airline_passengers(data_csv)
    passengers = dataset.passengers
    months = dataset.months

    log_passengers = np.log(passengers)
    z, mean, std = zscore(log_passengers)

    blocks, years = year_blocks(z, months)
    blocks = blocks[:, :, None]  # (n_years, 12, 1)

    year_start, year_end = years_range
    start_idx = years.index(year_start)
    end_idx = years.index(year_end) + 1
    Ys = [blocks[i] for i in range(start_idx, end_idx)]

    figures_dir = ensure_dir(output_figures)
    results_dir = ensure_dir(output_results)

    # Barycenters.
    mean_euc = euclidean_mean(Ys)
    mean_sdtw = barycenter(
        Ys,
        X_init="euclidean_mean",
        value_and_grad=functools.partial(sdtw_value_and_grad, gamma=gamma),
        tol=1e-6,
        max_iter=200,
    )
    mean_div = barycenter(
        Ys,
        X_init="euclidean_mean",
        value_and_grad=functools.partial(sdtw_div_value_and_grad, gamma=gamma),
        tol=1e-6,
        max_iter=200,
    )
    mean_sharp = barycenter(
        Ys,
        X_init="euclidean_mean",
        value_and_grad=functools.partial(sharp_sdtw_div_value_and_grad, gamma=gamma),
        tol=1e-6,
        max_iter=200,
    )

    # Bias demo (optimize X for a single Y, initialized at X=Y).
    y_idx = years.index(bias_year)
    Y = blocks[y_idx]
    X0 = Y.copy()

    def solve(arg_value_and_grad, gamma_value: float) -> np.ndarray:
        return barycenter(
            [Y],
            X_init=X0,
            value_and_grad=functools.partial(arg_value_and_grad, gamma=gamma_value),
            tol=1e-9,
            max_iter=200,
        )

    X_sdtw_g = solve(sdtw_value_and_grad, gamma_value=gamma)
    X_sdtw_g01 = solve(sdtw_value_and_grad, gamma_value=0.1)
    X_div_g = solve(sdtw_div_value_and_grad, gamma_value=gamma)

    def l2(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    # Plots: raw + transformed.
    plt.figure(figsize=(7.2, 2.6))
    plt.plot(passengers, lw=1.2)
    plt.title("Airline passengers (monthly, 1949–1960)")
    plt.xlabel("Month index")
    plt.ylabel("Passengers")
    plt.tight_layout()
    plt.savefig(figures_dir / "airline_raw.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 2.6))
    plt.plot(z, lw=1.2)
    plt.title(f"Log + z-score transform (mean={mean:.2f}, std={std:.2f})")
    plt.xlabel("Month index")
    plt.ylabel("Standardized log-passengers")
    plt.tight_layout()
    plt.savefig(figures_dir / "airline_transformed.png", dpi=200)
    plt.close()

    # Plot barycenters.
    xs = np.arange(12)
    plt.figure(figsize=(7.2, 3.4))
    for Y_i, year in zip(Ys, years[start_idx:end_idx], strict=True):
        plt.plot(xs, Y_i[:, 0], color="0.8", lw=1, alpha=0.9)
    plt.plot(xs, mean_euc[:, 0], lw=2.2, label="Euclidean mean")
    plt.plot(xs, mean_sdtw[:, 0], lw=2.2, label="soft-DTW barycenter")
    plt.plot(xs, mean_div[:, 0], lw=2.2, label="soft-DTW divergence barycenter")
    plt.plot(xs, mean_sharp[:, 0], lw=2.2, label="sharp soft-DTW divergence barycenter")
    plt.title(f"Yearly seasonality barycenters ({year_start}–{year_end})")
    plt.xlabel("Month within year")
    plt.ylabel("Standardized log-passengers")
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "airline_barycenters.png", dpi=200)
    plt.close()

    # Plot bias.
    plt.figure(figsize=(7.2, 3.2))
    plt.plot(xs, X0[:, 0], lw=2.2, label=f"init X = Y ({bias_year})")
    plt.plot(xs, X_sdtw_g01[:, 0], lw=2.2, label="argmin soft-DTW (gamma=0.1)")
    plt.plot(xs, X_sdtw_g[:, 0], lw=2.2, label=f"argmin soft-DTW (gamma={gamma:g})")
    plt.plot(xs, X_div_g[:, 0], lw=2.2, label=f"argmin soft-DTW divergence (gamma={gamma:g})")
    plt.title("Soft-DTW denoising bias vs divergence correction")
    plt.xlabel("Month within year")
    plt.ylabel("Standardized log-passengers")
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "airline_denoising_bias.png", dpi=200)
    plt.close()

    metrics = {
        "gamma": float(gamma),
        "n_years": int(len(Ys)),
        "years": years[start_idx:end_idx],
        "bias_year": bias_year,
        "l2_sdtw_gamma_0.1_vs_init": l2(X_sdtw_g01, X0),
        "l2_sdtw_gamma_1_vs_init": l2(X_sdtw_g, X0),
        "l2_div_gamma_1_vs_init": l2(X_div_g, X0),
        "sdtw_Y_Y_gamma_1": float(sdtw(Y, Y, gamma=gamma)),
        "div_Y_Y_gamma_1": float(sdtw_div(Y, Y, gamma=gamma)),
        "mean_sdtw_at_sdtw_barycenter": _mean_obj(functools.partial(sdtw, gamma=gamma), mean_sdtw, Ys),
        "mean_div_at_div_barycenter": _mean_obj(functools.partial(sdtw_div, gamma=gamma), mean_div, Ys),
        "mean_sharp_div_at_sharp_barycenter": _mean_obj(
            functools.partial(sharp_sdtw_div, gamma=gamma), mean_sharp, Ys
        ),
    }

    (results_dir / "airline_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (results_dir / "airline_metrics.txt").write_text(
        "\\begin{tabular}{l r}\\hline\n"
        "Metric & Value\\\\\n\\hline\n"
        + f"$\\gamma$ & {gamma:.2f}\\\\\n"
        + f"Years used & {year_start}--{year_end}\\\\\n"
        + f"$\\|\\hat X_{{\\mathrm{{sdtw}},\\gamma={gamma:g}}}-Y\\|_2$ & {metrics['l2_sdtw_gamma_1_vs_init']:.3f}\\\\\n"
        + f"$\\|\\hat X_{{\\mathrm{{sdtw}},\\gamma=0.1}}-Y\\|_2$ & {metrics['l2_sdtw_gamma_0.1_vs_init']:.3f}\\\\\n"
        + f"$\\|\\hat X_{{D_\\gamma,\\gamma={gamma:g}}}-Y\\|_2$ & {metrics['l2_div_gamma_1_vs_init']:.3f}\\\\\n"
        + f"$\\mathrm{{sdtw}}_\\gamma(Y,Y)$ & {metrics['sdtw_Y_Y_gamma_1']:.3f}\\\\\n"
        + f"$D_\\gamma(Y,Y)$ & {metrics['div_Y_Y_gamma_1']:.3f}\\\\\n"
        "\\hline\\end{tabular}\n"
    )

    return metrics


def run_barycenter_gamma_sweep(
    *,
    data_csv: str | Path = "data/airline-passengers.csv",
    output_figures: str | Path = "figures",
    gamma_values: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 2.0),
    years_range: tuple[str, str] = ("1951", "1958"),
    loss: str = "sdtw_div",
) -> dict:
    """Compare barycenters for multiple gamma values and save a plot."""
    data_csv = Path(data_csv)
    if not data_csv.exists():
        download_airline_passengers_csv(data_csv)

    dataset = load_airline_passengers(data_csv)
    log_passengers = np.log(dataset.passengers)
    z, _, _ = zscore(log_passengers)

    blocks, years = year_blocks(z, dataset.months)
    blocks = blocks[:, :, None]

    year_start, year_end = years_range
    start_idx = years.index(year_start)
    end_idx = years.index(year_end) + 1
    Ys = [blocks[i] for i in range(start_idx, end_idx)]

    if loss == "sdtw":
        value_and_grad = sdtw_value_and_grad
        label = "soft-DTW"
    elif loss == "sdtw_div":
        value_and_grad = sdtw_div_value_and_grad
        label = "soft-DTW divergence"
    elif loss == "sharp_sdtw_div":
        value_and_grad = sharp_sdtw_div_value_and_grad
        label = "sharp soft-DTW divergence"
    else:
        raise ValueError("loss must be one of: sdtw, sdtw_div, sharp_sdtw_div")

    barycenters = []
    for gamma in gamma_values:
        X = barycenter(
            Ys,
            X_init="euclidean_mean",
            value_and_grad=functools.partial(value_and_grad, gamma=gamma),
            tol=1e-6,
            max_iter=200,
        )
        barycenters.append((gamma, X))

    figures_dir = ensure_dir(output_figures)
    xs = np.arange(12)
    plt.figure(figsize=(7.2, 3.4))
    for Y_i in Ys:
        plt.plot(xs, Y_i[:, 0], color="0.85", lw=1, alpha=0.8)
    for gamma, X in barycenters:
        plt.plot(xs, X[:, 0], lw=2.2, label=f"{label} (gamma={gamma:g})")
    plt.title(f"Barycenters vs gamma ({label})")
    plt.xlabel("Month within year")
    plt.ylabel("Standardized log-passengers")
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "airline_barycenters_gamma_sweep.png", dpi=200)
    plt.close()

    return {
        "loss": loss,
        "gamma_values": list(gamma_values),
        "years": years[start_idx:end_idx],
    }
