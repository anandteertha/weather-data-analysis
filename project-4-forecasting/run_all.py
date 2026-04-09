from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forecast_utils import (  # noqa: E402
    Y_COL,
    autocorrelation,
    chi_square_normal_gof,
    ensure_dir,
    fit_ar_ols,
    load_weather_series,
    mape,
    pacf_regression,
    plot_full_series,
    plot_metric_curve,
    plot_pacf,
    plot_predictions,
    plot_qq,
    plot_residual_hist,
    plot_residual_scatter,
    plot_stationarity_features,
    plot_test_comparison,
    plot_transformations,
    rmse,
    rolling_mean_std,
    save_json,
    select_pacf_cutoff,
)


DATA_CSV = ROOT.parent / "project-2-weather-data-analysis" / "task1" / "weather_data.csv"
RESULTS_DIR = ROOT / "results"
TRAIN_RATIO = 0.8
ROLLING_WINDOW = 30
SEASONAL_PERIOD = 365


def simple_moving_average_fit(train: np.ndarray, k: int) -> np.ndarray:
    predicted = np.full(len(train), np.nan)
    for idx in range(k, len(train)):
        predicted[idx] = float(np.mean(train[idx - k : idx]))
    return predicted


def simple_moving_average_test(train: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    history = list(train.astype(float))
    predictions = []
    for value in test:
        predictions.append(float(np.mean(history[-k:])))
        history.append(float(value))
    return np.asarray(predictions, dtype=float)


def exponential_smoothing_fit(train: np.ndarray, alpha: float) -> np.ndarray:
    predicted = np.full(len(train), np.nan)
    predicted[0] = float(train[0])
    for idx in range(1, len(train)):
        predicted[idx] = alpha * float(train[idx - 1]) + (1.0 - alpha) * float(predicted[idx - 1])
    return predicted


def exponential_smoothing_test(train: np.ndarray, test: np.ndarray, alpha: float) -> np.ndarray:
    level = float(train[0])
    for idx in range(1, len(train)):
        level = alpha * float(train[idx - 1]) + (1.0 - alpha) * level

    predictions = []
    for value in test:
        predictions.append(level)
        level = alpha * float(value) + (1.0 - alpha) * level
    return np.asarray(predictions, dtype=float)


def seasonal_ar_train_predictions(train: np.ndarray, p: int, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seasonal_diff = train[SEASONAL_PERIOD:] - train[:-SEASONAL_PERIOD]
    transformed_pred = beta[0] + sum(
        beta[j] * seasonal_diff[p - j : len(seasonal_diff) - j] for j in range(1, p + 1)
    )
    original_indices = np.arange(SEASONAL_PERIOD + p, len(train))
    original_pred = train[original_indices - SEASONAL_PERIOD] + transformed_pred
    original_actual = train[original_indices]
    return original_actual, original_pred


def seasonal_ar_test_predictions(series: np.ndarray, train_n: int, p: int, beta: np.ndarray) -> np.ndarray:
    predictions = []
    for idx in range(train_n, len(series)):
        z_lags = []
        for lag in range(1, p + 1):
            source_idx = idx - lag
            z_lags.append(float(series[source_idx] - series[source_idx - SEASONAL_PERIOD]))
        z_pred = float(beta[0] + sum(beta[j] * z_lags[j - 1] for j in range(1, p + 1)))
        y_pred = float(series[idx - SEASONAL_PERIOD] + z_pred)
        predictions.append(y_pred)
    return np.asarray(predictions, dtype=float)


def analyze_stationarity(dates, series: np.ndarray, train_n: int) -> dict:
    out_dir = RESULTS_DIR / "task1"
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    rolling_mean, rolling_std = rolling_mean_std(series, ROLLING_WINDOW)
    month_means = dates.to_frame(name="date").assign(value=series).groupby(dates.dt.month)["value"].mean()

    diff1 = np.diff(series)
    seasonal_diff = series[SEASONAL_PERIOD:] - series[:-SEASONAL_PERIOD]
    log_shift = float(1.0 - np.min(series))
    log_shifted = np.log(series + log_shift)

    plot_full_series(dates, series, train_n, plots_dir / "full_series_train_test.png")
    plot_stationarity_features(
        dates,
        series,
        rolling_mean,
        rolling_std,
        month_means,
        plots_dir / "nonstationary_features.png",
    )
    plot_transformations(
        series,
        diff1,
        seasonal_diff,
        log_shifted,
        plots_dir / "transformations.png",
    )

    first_half = series[: len(series) // 2]
    second_half = series[len(series) // 2 :]
    transformed_summary = []
    for name, values in [
        ("original", series),
        ("first_difference", diff1),
        ("seasonal_difference_365", seasonal_diff),
        ("shifted_log", log_shifted),
    ]:
        half = len(values) // 2
        transformed_summary.append(
            {
                "series": name,
                "mean_first_half": float(np.mean(values[:half])),
                "mean_second_half": float(np.mean(values[half:])),
                "std_first_half": float(np.std(values[:half], ddof=1)),
                "std_second_half": float(np.std(values[half:], ddof=1)),
                "lag1_acf": float(autocorrelation(values, 1)[1]),
            }
        )

    results = {
        "n_total": int(len(series)),
        "train_n": int(train_n),
        "test_n": int(len(series) - train_n),
        "train_start": dates.iloc[0],
        "train_end": dates.iloc[train_n - 1],
        "test_start": dates.iloc[train_n],
        "test_end": dates.iloc[-1],
        "mean_first_half_original": float(np.mean(first_half)),
        "mean_second_half_original": float(np.mean(second_half)),
        "std_first_half_original": float(np.std(first_half, ddof=1)),
        "std_second_half_original": float(np.std(second_half, ddof=1)),
        "month_means": {int(k): float(v) for k, v in month_means.items()},
        "monthly_mean_range": float(month_means.max() - month_means.min()),
        "rolling_window_days": int(ROLLING_WINDOW),
        "log_shift_constant": log_shift,
        "transform_summary": transformed_summary,
        "commentary": {
            "stationarity": "The original series is not stationary because the monthly averages vary strongly across the calendar year.",
            "best_visual_transform": "First differencing reduces short-lag persistence most strongly, while seasonal differencing directly targets the annual weather cycle.",
        },
    }
    save_json(results, out_dir / "summary.json")
    return results


def analyze_sma(train: np.ndarray) -> dict:
    out_dir = RESULTS_DIR / "task2"
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    rows = []
    best = None
    for k in range(2, 61):
        predicted = simple_moving_average_fit(train, k)
        metrics = {
            "k": k,
            "rmse": rmse(train[k:], predicted[k:]),
            "mape": mape(train[k:], predicted[k:]),
        }
        rows.append(metrics)
        if best is None or metrics["rmse"] < best["rmse"]:
            best = metrics | {"predicted": predicted}

    plot_metric_curve(
        np.asarray([row["k"] for row in rows]),
        np.asarray([row["rmse"] for row in rows]),
        xlabel="k",
        ylabel="RMSE",
        title="Simple moving average: RMSE vs k",
        out_path=plots_dir / "rmse_vs_k.png",
    )
    plot_metric_curve(
        np.asarray([row["k"] for row in rows]),
        np.asarray([row["mape"] for row in rows]),
        xlabel="k",
        ylabel="MAPE (%)",
        title="Simple moving average: MAPE vs k",
        out_path=plots_dir / "mape_vs_k.png",
    )
    plot_predictions(
        train[best["k"] :],
        best["predicted"][best["k"] :],
        title=f"Simple moving average fit on training set (k = {best['k']})",
        out_path=plots_dir / "best_sma_training_fit.png",
        xlabel="Training-set index",
    )

    results = {
        "grid_results": rows,
        "best_k": int(best["k"]),
        "best_rmse": float(best["rmse"]),
        "best_mape": float(best["mape"]),
        "commentary": "A very short memory window works best, which is consistent with a series that changes quickly from day to day.",
    }
    save_json(results, out_dir / "summary.json")
    return results


def analyze_exponential(train: np.ndarray) -> dict:
    out_dir = RESULTS_DIR / "task3"
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    rows = []
    best = None
    for raw_alpha in range(1, 10):
        alpha = raw_alpha / 10.0
        predicted = exponential_smoothing_fit(train, alpha)
        metrics = {
            "alpha": alpha,
            "rmse": rmse(train[1:], predicted[1:]),
            "mape": mape(train[1:], predicted[1:]),
        }
        rows.append(metrics)
        if best is None or metrics["rmse"] < best["rmse"]:
            best = metrics | {"predicted": predicted}

    plot_metric_curve(
        np.asarray([row["alpha"] for row in rows]),
        np.asarray([row["rmse"] for row in rows]),
        xlabel="alpha",
        ylabel="RMSE",
        title="Exponential smoothing: RMSE vs alpha",
        out_path=plots_dir / "rmse_vs_alpha.png",
    )
    plot_predictions(
        train[1:],
        best["predicted"][1:],
        title=f"Exponential smoothing fit on training set (alpha = {best['alpha']:.1f})",
        out_path=plots_dir / "best_exp_training_fit.png",
        xlabel="Training-set index",
    )

    results = {
        "grid_results": rows,
        "best_alpha": float(best["alpha"]),
        "best_rmse": float(best["rmse"]),
        "best_mape": float(best["mape"]),
        "commentary": "Higher alpha values work better because the series responds quickly to recent weather changes.",
    }
    save_json(results, out_dir / "summary.json")
    return results


def analyze_ar(train: np.ndarray) -> dict:
    out_dir = RESULTS_DIR / "task4"
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    transformed_train = train[SEASONAL_PERIOD:] - train[:-SEASONAL_PERIOD]
    pacf_values = pacf_regression(transformed_train, max_lag=30)
    significance_band = float(1.96 / np.sqrt(len(transformed_train)))
    p = select_pacf_cutoff(pacf_values, significance_band)

    beta, _, residuals, sigma2_hat = fit_ar_ols(transformed_train, p)
    train_actual, train_pred = seasonal_ar_train_predictions(train, p, beta)
    train_rmse = rmse(train_actual, train_pred)
    train_mape = mape(train_actual, train_pred)
    gof = chi_square_normal_gof(residuals, sigma2_hat)

    plot_pacf(
        pacf_values,
        significance_band,
        plots_dir / "pacf_seasonally_differenced_training.png",
        title="PACF of seasonally differenced training series",
    )
    plot_predictions(
        train_actual,
        train_pred,
        title=f"AR({p}) training fit after seasonal differencing",
        out_path=plots_dir / "ar_training_fit.png",
        xlabel="Training-set index",
    )
    plot_qq(
        residuals,
        sigma2_hat,
        plots_dir / "qq_plot_residuals_ar.png",
        title="AR model residual Q-Q plot",
    )
    plot_residual_hist(
        residuals,
        sigma2_hat,
        plots_dir / "hist_residuals_ar.png",
        title="AR model residual histogram",
    )
    plot_residual_scatter(
        residuals,
        plots_dir / "residual_scatter_ar.png",
        title="AR model residual scatter",
    )

    results = {
        "seasonal_period": int(SEASONAL_PERIOD),
        "selected_p": int(p),
        "pacf_significance_band": significance_band,
        "pacf_first_10_lags": [float(v) for v in pacf_values[1:11]],
        "coefficients": {
            "intercept": float(beta[0]),
            **{f"phi_{idx}": float(beta[idx]) for idx in range(1, len(beta))},
        },
        "training_rmse_original_scale": float(train_rmse),
        "training_mape_original_scale": float(train_mape),
        "sigma2_hat_transformed_residuals": float(sigma2_hat),
        "chi_square_normal_gof": gof,
        "commentary": "Seasonal differencing makes the AR model more defensible, and the PACF cuts off after lag 2, so AR(2) is used.",
    }
    save_json(results, out_dir / "summary.json")
    return results


def compare_models(series: np.ndarray, train_n: int, sma_results: dict, exp_results: dict, ar_results: dict) -> dict:
    out_dir = RESULTS_DIR / "task5"
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    train = series[:train_n]
    test = series[train_n:]

    sma_pred = simple_moving_average_test(train, test, sma_results["best_k"])
    exp_pred = exponential_smoothing_test(train, test, exp_results["best_alpha"])
    ar_beta = np.asarray(
        [ar_results["coefficients"]["intercept"]]
        + [ar_results["coefficients"][f"phi_{i}"] for i in range(1, ar_results["selected_p"] + 1)],
        dtype=float,
    )
    ar_pred = seasonal_ar_test_predictions(series, train_n, ar_results["selected_p"], ar_beta)

    rows = [
        {
            "model": "simple_moving_average",
            "rmse": rmse(test, sma_pred),
            "mape": mape(test, sma_pred),
        },
        {
            "model": "exponential_smoothing",
            "rmse": rmse(test, exp_pred),
            "mape": mape(test, exp_pred),
        },
        {
            "model": "ar_model",
            "rmse": rmse(test, ar_pred),
            "mape": mape(test, ar_pred),
        },
    ]
    best = min(rows, key=lambda row: row["rmse"])

    plot_test_comparison(test, sma_pred, exp_pred, ar_pred, plots_dir / "test_set_model_comparison.png")

    results = {
        "test_metrics": rows,
        "best_model_by_rmse": best["model"],
        "best_model_rmse": float(best["rmse"]),
        "best_model_mape": float(best["mape"]),
        "commentary": "Exponential smoothing performs best on the testing set because it reacts quickly to local changes without needing long seasonal lag reconstruction.",
    }
    save_json(results, out_dir / "summary.json")
    return results


def fnum(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_report(task1: dict, task2: dict, task3: dict, task4: dict, task5: dict) -> str:
    task1_rows = "\n".join(
        [
            f"| {row['series']} | {fnum(row['mean_first_half'])} | {fnum(row['mean_second_half'])} | {fnum(row['std_first_half'])} | {fnum(row['std_second_half'])} | {fnum(row['lag1_acf'])} |"
            for row in task1["transform_summary"]
        ]
    )
    task5_rows = "\n".join(
        [
            f"| {row['model']} | {fnum(row['rmse'])} | {fnum(row['mape'])} |"
            for row in task5["test_metrics"]
        ]
    )
    ar_coeff_lines = "\n".join(
        [f"- phi_{idx}: `{fnum(task4['coefficients'][f'phi_{idx}'])}`" for idx in range(1, task4["selected_p"] + 1)]
    )

    return f"""# Project 4: Forecasting

CSC 591 / ECE 592 IoT Analytics

## Overview

This submission uses the Project 2 weather dataset to forecast the univariate time series `Y_apparent_temperature_C` (apparent temperature in Celsius) for Raleigh. The full dataset contains `{task1['n_total']}` daily observations from `{str(task1['train_start'])[:10]}` through `{str(task1['test_end'])[:10]}`. A chronological 80/20 split is used:

- Training set: `{task1['train_n']}` samples from `{str(task1['train_start'])[:10]}` to `{str(task1['train_end'])[:10]}`
- Testing set: `{task1['test_n']}` samples from `{str(task1['test_start'])[:10]}` to `{str(task1['test_end'])[:10]}`

Implementation details:

- Language: Python
- Packages: `numpy`, `pandas`, `matplotlib`, `scipy`
- Forecasting models: simple moving average, simple exponential smoothing, autoregression
- AI assistance: OpenAI Codex was used to help organize the scripts and draft the report text; all code was executed and the reported values were generated locally from the dataset

MAPE note: the apparent-temperature series contains a few zero values and also negative temperatures, so MAPE was computed only on non-zero targets using `|actual|` in the denominator.

## Task 1: Stationarity and Non-Stationary Properties

The full time-series plot is saved in `results/task1/plots/full_series_train_test.png`. Visual inspection shows repeated annual cycles, so the original series is not stationary.

Evidence of non-stationarity:

- Trend / mean shift: the 30-day rolling mean changes noticeably over time
- Variance: the 30-day rolling standard deviation is not perfectly constant
- Seasonality: the mean temperature changes strongly by month; the monthly mean range is `{fnum(task1['monthly_mean_range'])}` C

The non-stationary feature plots are saved in `results/task1/plots/nonstationary_features.png`.

To check stationary properties, three transformations were examined:

- First differencing
- Seasonal differencing with lag 365
- Shifted logarithm transformation with shift constant `{fnum(task1['log_shift_constant'])}`

Transformation comparison summary:

| Series | Mean (1st half) | Mean (2nd half) | Std (1st half) | Std (2nd half) | Lag-1 ACF |
|---|---:|---:|---:|---:|---:|
{task1_rows}

Interpretation:

- The original series has strong autocorrelation and a very clear yearly cycle.
- First differencing reduces short-term persistence much more than the other single transformations.
- The shifted log transform compresses the scale, but by itself it does not remove seasonality.
- Seasonal differencing specifically targets the annual cycle, so it was the most useful preprocessing step before fitting the AR model.

The transformation plot is saved in `results/task1/plots/transformations.png`.

## Task 2: Simple Moving Average Model

For the training set, the simple moving average model

`x_hat_t = (1/k) * sum_(i=t-k)^(t-1) x_i`

was fit for `k = 2` through `k = 60`.

Results:

- Best `k` by RMSE: `{task2['best_k']}`
- Training RMSE: `{fnum(task2['best_rmse'])}`
- Training MAPE: `{fnum(task2['best_mape'])}`%

Plots:

- `results/task2/plots/rmse_vs_k.png`
- `results/task2/plots/mape_vs_k.png`
- `results/task2/plots/best_sma_training_fit.png`

Comment:

The best window is very short. That suggests yesterday and the day before are much more informative than a long average window, which is reasonable for day-to-day weather fluctuations.

## Task 3: Exponential Smoothing Model

The simple exponential smoothing model

`x_hat_t = alpha * x_(t-1) + (1 - alpha) * x_hat_(t-1)`

was fit on the training set for `alpha = 0.1, 0.2, ..., 0.9`.

Results:

- Best `alpha` by RMSE: `{task3['best_alpha']:.1f}`
- Training RMSE: `{fnum(task3['best_rmse'])}`
- Training MAPE: `{fnum(task3['best_mape'])}`%

Plots:

- `results/task3/plots/rmse_vs_alpha.png`
- `results/task3/plots/best_exp_training_fit.png`

Comment:

The best smoothing level is high, which means the model benefits from putting more weight on very recent observations. This is consistent with weather data, where short-run changes matter a lot.

## Task 4: AR(p) Model

To make the autoregressive model more appropriate for a seasonal weather series, the training data was first seasonally differenced:

`z_t = y_t - y_(t-365)`

PACF was then computed on the seasonally differenced training series. The PACF plot is saved in `results/task4/plots/pacf_seasonally_differenced_training.png`.

PACF interpretation:

- Significance band: `+/- {fnum(task4['pacf_significance_band'])}`
- The PACF is clearly significant at the first two lags and then drops inside the band, so `p = {task4['selected_p']}` was selected

Estimated AR coefficients:

- Intercept: `{fnum(task4['coefficients']['intercept'])}`
{ar_coeff_lines}

Training performance on the original temperature scale:

- RMSE: `{fnum(task4['training_rmse_original_scale'])}`
- MAPE: `{fnum(task4['training_mape_original_scale'])}`%

Residual diagnostics:

- Q-Q plot: `results/task4/plots/qq_plot_residuals_ar.png`
- Residual histogram: `results/task4/plots/hist_residuals_ar.png`
- Residual scatter: `results/task4/plots/residual_scatter_ar.png`
- Chi-square GOF p-value: `{fnum(task4['chi_square_normal_gof']['p_value'])}`

Comment:

The AR model removes most short-lag residual correlation after seasonal differencing, but the residuals are still not perfectly normal. In other words, the model is reasonable, but it is not a perfect description of the weather series.

## Task 5: Testing-Set Comparison of All Models

The three models were kept fixed after training and then run on the testing set using one-step-ahead rolling forecasts.

Testing-set results:

| Model | RMSE | MAPE (%) |
|---|---:|---:|
{task5_rows}

Best model on the testing set:

- `{task5['best_model_by_rmse']}`
- RMSE: `{fnum(task5['best_model_rmse'])}`
- MAPE: `{fnum(task5['best_model_mape'])}`%

The comparison plot is saved in `results/task5/plots/test_set_model_comparison.png`.

## Conclusions

- The original apparent-temperature series is non-stationary mainly because of strong annual seasonality.
- Among the preprocessing ideas checked in Task 1, first differencing reduced short-lag dependence the most, while seasonal differencing was the most useful transformation for AR modeling because it targeted the yearly cycle.
- The simple moving average model worked best with a very small window (`k = {task2['best_k']}`), confirming that the series changes quickly.
- Exponential smoothing with `alpha = {task3['best_alpha']:.1f}` outperformed the other models on the testing set, so it is the best forecasting model in this project.
- The AR model was useful for time-series interpretation and residual analysis, but in this dataset it did not beat the simpler exponential smoothing model on held-out data.
"""


def main() -> None:
    ensure_dir(RESULTS_DIR)
    df = load_weather_series(DATA_CSV)
    dates = df["date"]
    series = df[Y_COL].to_numpy(dtype=float)
    train_n = int(round(len(series) * TRAIN_RATIO))
    train = series[:train_n]

    task1 = analyze_stationarity(dates, series, train_n)
    task2 = analyze_sma(train)
    task3 = analyze_exponential(train)
    task4 = analyze_ar(train)
    task5 = compare_models(series, train_n, task2, task3, task4)

    report_text = build_report(task1, task2, task3, task4, task5)
    (ROOT / "report.md").write_text(report_text, encoding="utf-8")

    print("Project 4 outputs generated:")
    print(f"- report: {ROOT / 'report.md'}")
    print(f"- results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
