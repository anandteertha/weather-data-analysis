# Project 4: Forecasting

CSC 591 / ECE 592 IoT Analytics

## Overview

This submission uses the Project 2 weather dataset to forecast the univariate time series `Y_apparent_temperature_C` (apparent temperature in Celsius) for Raleigh. The full dataset contains `2201` daily observations from `2020-02-20` through `2026-02-28`. A chronological 80/20 split is used:

- Training set: `1761` samples from `2020-02-20` to `2024-12-15`
- Testing set: `440` samples from `2024-12-16` to `2026-02-28`

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
- Seasonality: the mean temperature changes strongly by month; the monthly mean range is `30.8640` C

The non-stationary feature plots are saved in `results/task1/plots/nonstationary_features.png`.

To check stationary properties, three transformations were examined:

- First differencing
- Seasonal differencing with lag 365
- Shifted logarithm transformation with shift constant `13.7000`

Transformation comparison summary:

| Series | Mean (1st half) | Mean (2nd half) | Std (1st half) | Std (2nd half) | Lag-1 ACF |
|---|---:|---:|---:|---:|---:|
| original | 20.3484 | 19.9725 | 11.5545 | 11.7730 | 0.9078 |
| first_difference | 0.0122 | -0.0002 | 5.1168 | 4.8689 | -0.0951 |
| seasonal_difference_365 | 0.1989 | -0.3694 | 7.8375 | 8.3588 | 0.6208 |
| shifted_log | 3.4559 | 3.4352 | 0.4086 | 0.4472 | 0.8651 |

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

- Best `k` by RMSE: `2`
- Training RMSE: `5.3426`
- Training MAPE: `81.1856`%

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

- Best `alpha` by RMSE: `0.8`
- Training RMSE: `4.9156`
- Training MAPE: `72.7927`%

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

- Significance band: `+/- 0.0525`
- The PACF is clearly significant at the first two lags and then drops inside the band, so `p = 2` was selected

Estimated AR coefficients:

- Intercept: `0.1086`
- phi_1: `0.7082`
- phi_2: `-0.1657`

Training performance on the original temperature scale:

- RMSE: `6.2044`
- MAPE: `95.1335`%

Residual diagnostics:

- Q-Q plot: `results/task4/plots/qq_plot_residuals_ar.png`
- Residual histogram: `results/task4/plots/hist_residuals_ar.png`
- Residual scatter: `results/task4/plots/residual_scatter_ar.png`
- Chi-square GOF p-value: `0.0239`

Comment:

The AR model removes most short-lag residual correlation after seasonal differencing, but the residuals are still not perfectly normal. In other words, the model is reasonable, but it is not a perfect description of the weather series.

## Task 5: Testing-Set Comparison of All Models

The three models were kept fixed after training and then run on the testing set using one-step-ahead rolling forecasts.

Testing-set results:

| Model | RMSE | MAPE (%) |
|---|---:|---:|
| simple_moving_average | 5.4479 | 106.8081 |
| exponential_smoothing | 5.0933 | 86.2606 |
| ar_model | 6.5208 | 129.3280 |

Best model on the testing set:

- `exponential_smoothing`
- RMSE: `5.0933`
- MAPE: `86.2606`%

The comparison plot is saved in `results/task5/plots/test_set_model_comparison.png`.

## Conclusions

- The original apparent-temperature series is non-stationary mainly because of strong annual seasonality.
- Among the preprocessing ideas checked in Task 1, first differencing reduced short-lag dependence the most, while seasonal differencing was the most useful transformation for AR modeling because it targeted the yearly cycle.
- The simple moving average model worked best with a very small window (`k = 2`), confirming that the series changes quickly.
- Exponential smoothing with `alpha = 0.8` outperformed the other models on the testing set, so it is the best forecasting model in this project.
- The AR model was useful for time-series interpretation and residual analysis, but in this dataset it did not beat the simpler exponential smoothing model on held-out data.
