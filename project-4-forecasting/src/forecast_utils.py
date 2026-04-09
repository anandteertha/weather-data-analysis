from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


Y_COL = "Y_apparent_temperature_C"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(to_builtin(data), indent=2), encoding="utf-8")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def load_weather_series(data_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_csv_path)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / np.abs(actual[mask]))) * 100.0)


def rolling_mean_std(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(np.asarray(series, dtype=float))
    return (
        s.rolling(window=window, min_periods=window).mean().to_numpy(),
        s.rolling(window=window, min_periods=window).std().to_numpy(),
    )


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.zeros(max_lag + 1, dtype=float)
    values = [1.0]
    for lag in range(1, max_lag + 1):
        values.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(values, dtype=float)


def pacf_regression(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    values = [1.0]
    for lag in range(1, max_lag + 1):
        y = x[lag:]
        X = np.column_stack([x[lag - j - 1 : len(x) - j - 1] for j in range(lag)])
        X_design = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        values.append(float(beta[-1]))
    return np.asarray(values, dtype=float)


def select_pacf_cutoff(pacf_values: np.ndarray, significance_band: float) -> int:
    p = 1
    for lag in range(1, len(pacf_values) - 1):
        if abs(pacf_values[lag]) > significance_band:
            p = lag
            continue
        if abs(pacf_values[lag]) <= significance_band and abs(pacf_values[lag + 1]) <= significance_band:
            break
    return max(1, p)


def fit_ar_ols(series: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(series, dtype=float)
    y = x[p:]
    X = np.column_stack([x[p - j - 1 : len(x) - j - 1] for j in range(p)])
    X_design = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    fitted = X_design @ beta
    residuals = y - fitted
    sigma2_hat = float(np.sum(residuals**2) / max(1, len(residuals) - (p + 1)))
    return beta, fitted, residuals, sigma2_hat


def chi_square_normal_gof(
    residuals: np.ndarray,
    sigma2_hat: float,
    n_bins: int = 8,
) -> dict[str, Any]:
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n < 10 or sigma2_hat <= 0:
        return {
            "chi2_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "note": "Insufficient data or non-positive variance.",
        }

    std = float(np.sqrt(sigma2_hat))
    edges = np.linspace(float(np.min(residuals)), float(np.max(residuals)), n_bins + 1)
    edges = np.unique(edges)
    if len(edges) < 3:
        return {
            "chi2_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "note": "Too few unique bin edges.",
        }

    observed, _ = np.histogram(residuals, bins=edges)
    cdf_values = stats.norm.cdf(edges, loc=0.0, scale=std)
    expected = n * np.diff(cdf_values)

    observed = observed.astype(float)
    expected = expected.astype(float)
    while len(expected) > 2 and np.any(expected < 5.0):
        merge_idx = int(np.where(expected < 5.0)[0][0])
        if merge_idx == len(expected) - 1:
            merge_idx -= 1
        observed[merge_idx] += observed[merge_idx + 1]
        expected[merge_idx] += expected[merge_idx + 1]
        observed = np.delete(observed, merge_idx + 1)
        expected = np.delete(expected, merge_idx + 1)

    if np.any(expected <= 0):
        return {
            "chi2_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "note": "Invalid expected counts after bin merging.",
        }

    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    df = int(max(1, len(observed) - 2))
    p_value = float(stats.chi2.sf(chi2_stat, df))
    return {
        "chi2_stat": chi2_stat,
        "df": df,
        "p_value": p_value,
        "n_bins_used": int(len(observed)),
    }


def plot_full_series(dates: pd.Series, values: np.ndarray, train_end: int, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 4.5))
    plt.plot(dates, values, linewidth=1.2, color="steelblue")
    split_date = dates.iloc[train_end - 1]
    plt.axvline(split_date, color="darkred", linestyle="--", linewidth=1.4, label="Train/Test split")
    plt.title("Full apparent temperature time series")
    plt.xlabel("Date")
    plt.ylabel("Apparent temperature (C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_stationarity_features(
    dates: pd.Series,
    values: np.ndarray,
    rolling_mean: np.ndarray,
    rolling_std: np.ndarray,
    month_means: pd.Series,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(dates, values, color="slateblue", linewidth=1.0, label="Original series")
    axes[0].plot(dates, rolling_mean, color="darkorange", linewidth=1.5, label="30-day rolling mean")
    axes[0].set_title("Trend check: original series and 30-day rolling mean")
    axes[0].legend()

    axes[1].plot(dates, rolling_std, color="seagreen", linewidth=1.3)
    axes[1].set_title("Variance check: 30-day rolling standard deviation")
    axes[1].set_ylabel("Rolling std")

    axes[2].plot(month_means.index, month_means.values, marker="o", color="firebrick")
    axes[2].set_title("Seasonality check: average apparent temperature by month")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Mean apparent temperature (C)")
    axes[2].set_xticks(range(1, 13))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_transformations(
    original: np.ndarray,
    diff1: np.ndarray,
    seasonal_diff: np.ndarray,
    log_shifted: np.ndarray,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11))

    axes[0].plot(original, color="steelblue", linewidth=1.0)
    axes[0].set_title("Original series")

    axes[1].plot(diff1, color="darkorange", linewidth=1.0)
    axes[1].set_title("First-differenced series")

    axes[2].plot(seasonal_diff, color="seagreen", linewidth=1.0)
    axes[2].set_title("Seasonally differenced series (lag = 365)")

    axes[3].plot(log_shifted, color="purple", linewidth=1.0)
    axes[3].set_title("Shifted log-transformed series")
    axes[3].set_xlabel("Time index")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_values, y_values, marker="o", color="darkslateblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    out_path: Path,
    xlabel: str = "Time index",
) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 4.5))
    plt.plot(actual, label="Actual", color="black", linewidth=1.2)
    plt.plot(predicted, label="Predicted", color="royalblue", linewidth=1.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Apparent temperature (C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pacf(
    pacf_values: np.ndarray,
    significance_band: float,
    out_path: Path,
    title: str,
) -> None:
    ensure_dir(out_path.parent)
    lags = np.arange(1, len(pacf_values))
    plt.figure(figsize=(9, 4.5))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.axhline(significance_band, color="darkred", linestyle="--", linewidth=1.0)
    plt.axhline(-significance_band, color="darkred", linestyle="--", linewidth=1.0)
    plt.vlines(lags, 0.0, pacf_values[1:], color="teal", linewidth=2.0)
    plt.scatter(lags, pacf_values[1:], color="teal", s=20)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_qq(residuals: np.ndarray, sigma2_hat: float, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    std = float(np.sqrt(sigma2_hat)) if sigma2_hat > 0 else float(np.std(residuals, ddof=1))
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", sparams=(0, std), plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual_hist(residuals: np.ndarray, sigma2_hat: float, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    residuals = np.asarray(residuals, dtype=float)
    std = float(np.sqrt(sigma2_hat)) if sigma2_hat > 0 else float(np.std(residuals, ddof=1))
    xs = np.linspace(float(np.min(residuals)), float(np.max(residuals)), 300)
    plt.figure(figsize=(8, 4.5))
    plt.hist(residuals, bins="auto", density=True, edgecolor="black", alpha=0.85)
    plt.plot(xs, stats.norm.pdf(xs, loc=0.0, scale=std), color="darkred", linewidth=2.0)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual_scatter(residuals: np.ndarray, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(np.arange(len(residuals)), residuals, alpha=0.7, color="slateblue", s=18)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Time index")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_test_comparison(
    actual: np.ndarray,
    sma_pred: np.ndarray,
    exp_pred: np.ndarray,
    ar_pred: np.ndarray,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual", color="black", linewidth=1.3)
    plt.plot(sma_pred, label="Moving average", linewidth=1.1)
    plt.plot(exp_pred, label="Exp. smoothing", linewidth=1.1)
    plt.plot(ar_pred, label="AR model", linewidth=1.1)
    plt.title("Testing-set forecasts: all three models")
    plt.xlabel("Test-set index")
    plt.ylabel("Apparent temperature (C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
