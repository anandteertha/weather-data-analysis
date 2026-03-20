# Project 3: Regression and Forecasting

This report documents the full workflow and results for the Project 3 regression analysis. The write-up follows the assignment requirements for:
- Task 1: Basic statistic analysis + outlier removal + correlation study
- Task 2: Simple linear regression and polynomial (quadratic) extension (in X1)
- Task 3: Multivariable linear regression with residual diagnostics

## Dataset

We reuse the weather dataset from Project 2:
- Input variables: X1..X5 and output Y
- Columns:
  - Y: `Y_apparent_temperature_C`
  - X1: `X1_temperature_2m_C`
  - X2: `X2_relative_humidity_2m_pct`
  - X3: `X3_surface_pressure_hPa`
  - X4: `X4_wind_speed_10m_kmh`
  - X5: `X5_cloud_cover_pct`

For modeling, we use the first `n=100` samples from `../project-2-weather-data-analysis/task1/weather_data.csv`, matching the same sample size used in the earlier project work.

Outlier removal (used in Tasks 2 and 3, and for Task 1 correlation):
- Applied a conservative IQR rule with `k=1.5`
- Rows are removed if `Y` or any Xi is outside `[Q1 - k*IQR, Q3 + k*IQR]`

## Modeling Methodology (Coding/Math Used)

Implementation uses Python with:
- `pandas` / `numpy` for data loading and OLS linear algebra inputs
- `matplotlib` for all plots (histograms, boxplots, residual plots)
- `scipy.stats` for Student-t p-values, Normal CDF (chi-square GOF expected counts), and Q-Q plots (`probplot`)

### Ordinary Least Squares (OLS)

For each regression model, we fit:
Y = beta0 + beta1 * X1 + ... + betap * Xp + epsilon

Coefficients are computed by the normal equations:
beta_hat = (X_design' X_design)^(-1) X_design' y

Residuals are:
e = y - y_hat

Estimated residual variance:
sigma^2_hat = SSE / (n - p)
where SSE = sum(e_i^2), n is the number of samples, and p is the number of fitted parameters (including the intercept).

Standard errors and p-values:
- cov(beta_hat) = sigma^2_hat * (X_design' X_design)^(-1)
- t_i = beta_hat_i / se(beta_hat_i)
- p-values use the two-sided Student t distribution with df = n - p.

Goodness of fit:
R^2 = 1 - SSE / SST, where SST = sum((y - y_mean)^2)

Adjusted R^2:
adj-R^2 = 1 - (1 - R^2) * (n - 1) / (n - p)

### Correlation

The correlation matrix uses Pearson correlation computed on the filtered dataset:
corr(A,B) = Cov(A,B) / (std(A) * std(B))

### Residual diagnostics + chi-square normality test (approximate)

For Task 2.4 and Task 3.3, we test whether residuals follow N(0, sigma^2_hat):
- Q-Q plots compare residual quantiles against a Normal distribution with mean 0 and standard deviation sqrt(sigma^2_hat).
- The chi-square test uses binning of residuals into equal-width intervals over the observed residual range.
- Expected bin counts are computed from the Normal CDF:
  E[count_j] = n * (F(upper_j) - F(lower_j))
- To reduce instability, bins with expected counts below 5 are merged with neighboring bins (greedy merge).
- Degrees of freedom are reported as df = k - 2, where k is the number of bins used and the variance parameter is treated as estimated (mean is fixed at 0 by construction of N(0, sigma^2_hat)).

## Task 1: Basic Statistic Analysis

### Task 1.1 (Histogram, mean, variance)

Descriptive stats for each Xi (before outlier removal):

| Variable | Mean | Variance (sample) | Count |
|---|---|---|---|
| X1_temperature_2m_C | 18.6930 | 37.2504 | 100 |
| X2_relative_humidity_2m_pct | 50.7700 | 339.3910 | 100 |
| X3_surface_pressure_hPa | 1004.6750 | 48.7772 | 100 |
| X4_wind_speed_10m_kmh | 13.6640 | 41.9201 | 100 |
| X5_cloud_cover_pct | 61.6800 | 1853.2905 | 100 |

Example histograms (saved plots):
- `results/task1/plots/hist_X1_temperature_2m_C.png`
- `results/task1/plots/hist_X2_relative_humidity_2m_pct.png`
- `results/task1/plots/hist_X3_surface_pressure_hPa.png`
- `results/task1/plots/hist_X4_wind_speed_10m_kmh.png`
- `results/task1/plots/hist_X5_cloud_cover_pct.png`

### Task 1.2 (Box plot outlier removal)

Boxplots saved:
- Before: `results/task1/plots/boxplots_before_outlier_removal.png`
- After: `results/task1/plots/boxplots_after_outlier_removal.png`

Sample size:
- Raw `n_raw` = 100
- Filtered `n_filtered` = 96

### Task 1.3 (Correlation matrix)

Correlation heatmap (filtered dataset):
- `results/task1/plots/correlation_heatmap_filtered.png`
- CSV: `results/task1/correlation_matrix_filtered.csv`

Correlations between Y and each Xi (filtered):
- X1_temperature_2m_C: r=0.981 (positive, strong strength)
- X2_relative_humidity_2m_pct: r=0.070 (positive, weak strength)
- X3_surface_pressure_hPa: r=-0.139 (negative, weak strength)
- X4_wind_speed_10m_kmh: r=-0.276 (negative, weak strength)
- X5_cloud_cover_pct: r=-0.019 (negative, weak strength)


### Task 1.4 (Conclusions on dependencies)

Interpretation rule used:
- `abs(r) >= 0.7`: strong
- `0.3 <= abs(r) < 0.7`: moderate
- `abs(r) < 0.3`: weak

Based on the Y-vs-X correlation magnitudes, variables with the largest absolute correlations are the most likely direct dependencies with apparent temperature.
In this dataset, `X1_temperature_2m_C` is clearly the dominant predictor because its correlation with `Y` is very high (`r=0.981`), whereas the remaining individual correlations with `Y` are weak. Therefore, the one-variable view suggests that apparent temperature is driven primarily by actual temperature. At the same time, weak pairwise correlation does not automatically mean a variable is useless; some variables can still improve a multivariable model by explaining variation that remains after `X1` is already included.

## Task 2: Simple Linear Regression

We fit:
Y = a0 + a1 X1 + epsilon
on the filtered dataset.

### Task 2.1 + 2.2 (Parameter estimates, p-values, R^2, adjusted R^2)

Simple regression coefficients (with standard errors and p-values):

| Coefficient | Estimate | Std. Error | t-stat | p-value |
|---|---|---|---|---|
| a0_intercept | -6.929359 | 0.527452 | -13.137 | 5.3507e-23 |
| a1_X1 | 1.304789 | 0.026797 | 48.691 | 1.76059e-68 |

Model fit:
- sigma^2_hat (MSE) = 2.312804
- R^2 = 0.961863
- adjusted R^2 = 0.961458

### Task 2.3 (Regression line plot)
- `results/task2/plots/regression_line_simple.png`

### Task 2.4 (Residuals analysis)

Q-Q plot of residuals vs N(0, sigma^2):
- `results/task2/plots/qq_plot_residuals_simple.png`

Residual histogram with normal overlay:
- `results/task2/plots/hist_residuals_simple.png`

Chi-square normality test (approximate GOF for N(0, sigma^2_hat)):
- chi2_stat = 0.613573
- df = 3
- p_value = 0.893318

Residual scatter plot (vs fitted values):
- `results/task2/plots/residual_scatter_vs_fitted_simple.png`

### Task 2.5 (Higher-order polynomial: quadratic in X1)

We extend the regression to:
Y = a0 + a1 X1 + a2 X1^2 + epsilon

Quadratic fit coefficients:

| Coefficient | Estimate | Std. Error | t-stat | p-value |
|---|---|---|---|---|
| a0_intercept | -4.554987 | 1.348062 | -3.379 | 0.00106464 |
| a1_X1 | 1.023956 | 0.149451 | 6.851 | 7.81402e-10 |
| a2_X1_squared | 0.007507 | 0.003932 | 1.909 | 0.0593208 |

Model fit:
- sigma^2_hat (MSE) = 2.249507
- R^2 = 0.963302
- adjusted R^2 = 0.962512

Quadratic regression curve plot:
- `results/task2/plots/regression_curve_quadratic.png`

### Task 2.6 (Comment on results)

- The quadratic model changes the fit relative to the simple model by:
  - delta R^2 = 0.001438
  - delta adjusted R^2 = 0.001055
- The quadratic term `a2_X1_squared` has p-value = 0.0593208. Since alpha=0.05, it is not statistically significant at the 5% level.
- Because the improvement in fit is very small and the quadratic term is not significant at the 5% level, the simple linear model in `X1` is the more convincing model for interpretation. The quadratic fit is still useful to report because it shows that adding curvature provides only a marginal benefit for this dataset.

## Task 3: Multi-variable Linear Regression

### Task 3.1 (Initial multivariable model)

Initial model uses all independent variables:
Y = a0 + a1 X1 + a2 X2 + a3 X3 + a4 X4 + a5 X5 + epsilon

Initial full-model fit:
- sigma^2_hat (MSE) = 0.437931
- R^2 = 0.993086
- adjusted R^2 = 0.992702

Initial full-model coefficients:

| Coefficient | Estimate | Std. Error | t-stat | p-value |
|---|---|---|---|---|
| a0_intercept | -28.571115 | 12.070689 | -2.367 | 0.0200771 |
| X1_temperature_2m_C | 1.290988 | 0.012154 | 106.219 | 2.25302e-96 |
| X2_relative_humidity_2m_pct | 0.054794 | 0.004421 | 12.393 | 3.65203e-21 |
| X3_surface_pressure_hPa | 0.021769 | 0.011863 | 1.835 | 0.0698028 |
| X4_wind_speed_10m_kmh | -0.153685 | 0.012095 | -12.706 | 8.67225e-22 |
| X5_cloud_cover_pct | -0.011112 | 0.001903 | -5.841 | 8.1521e-08 |

### Task 3.2 (Variable removal)

We use p-values and adjusted R^2 to decide whether to drop variables:
- Iteratively remove the variable with the largest p-value if p > 0.05
- Only accept a removal if adjusted R^2 does not decrease (conservative rule)

Removal steps (if any):

No variable was removed during the backward-elimination check.

Correlation context (from Task 1): `r(Y, X3_surface_pressure_hPa)` = -0.139 (negative, weak strength). In the final multivariable model, its coefficient p-value is 0.0698028 (> 0.05 means not significant at 5%). If `X3_surface_pressure_hPa` is removed anyway, the adjusted R^2 drops from 0.992702 to 0.992512, R^2 drops from 0.993086 to 0.992827, and sigma^2_hat increases from 0.437931 to 0.449324. For that reason, the final report keeps `X3_surface_pressure_hPa` even though its individual p-value is slightly above 0.05. This decision is consistent with the assignment instruction to consider p-values, R-squared, adjusted R-squared, and the correlation matrix together rather than relying on a single metric.

Final variable set:
- X1_temperature_2m_C, X2_relative_humidity_2m_pct, X3_surface_pressure_hPa, X4_wind_speed_10m_kmh, X5_cloud_cover_pct

Final model fit:
- sigma^2_hat (MSE) = 0.437931
- R^2 = 0.993086
- adjusted R^2 = 0.992702

Final coefficients:

| Coefficient | Estimate | Std. Error | t-stat | p-value |
|---|---|---|---|---|
| a0_intercept | -28.571115 | 12.070689 | -2.367 | 0.0200771 |
| X1_temperature_2m_C | 1.290988 | 0.012154 | 106.219 | 2.25302e-96 |
| X2_relative_humidity_2m_pct | 0.054794 | 0.004421 | 12.393 | 3.65203e-21 |
| X3_surface_pressure_hPa | 0.021769 | 0.011863 | 1.835 | 0.0698028 |
| X4_wind_speed_10m_kmh | -0.153685 | 0.012095 | -12.706 | 8.67225e-22 |
| X5_cloud_cover_pct | -0.011112 | 0.001903 | -5.841 | 8.1521e-08 |

### Task 3.3 (Residuals analysis for final model)

Q-Q plot:
- `results/task3/plots/qq_plot_residuals_multivariable.png`

Residual histogram:
- `results/task3/plots/hist_residuals_multivariable.png`

Chi-square normality test (approximate GOF for N(0, sigma^2_hat)):
- chi2_stat = 6.011604
- df = 4
- p_value = 0.198283

Residual scatter:
- `results/task3/plots/residual_scatter_vs_fitted_multivariable.png`

### Task 3.4 (Comment on results)

- Final model R^2 and adjusted R^2 are high (0.993086 and 0.992702), indicating that the model explains almost all of the observed variation in `Y` for this filtered sample.
- Variables with p-value > 0.05 in the final multivariable model: X3_surface_pressure_hPa (worst p-value = 0.0698028).
- Residual normality check (Task 3.3 chi-square GOF): p-value = 0.198283, so there is no strong evidence against the normal-residual assumption in this approximate test.
- Compared with the one-variable model in Task 2, the multivariable model reduces sigma^2_hat substantially (2.312804 down to 0.437931), showing that the added predictors explain meaningful variation beyond `X1` alone.
- The final signs of the coefficients are physically plausible for this dataset: higher temperature and humidity increase apparent temperature, while higher wind speed and cloud cover reduce it. Surface pressure has a small positive coefficient, but because its p-value is marginal, its effect should be described as weak/inconclusive rather than strong.
- Overall, the multivariable model is the strongest model in this submission because it combines excellent fit, reasonable diagnostics, and coefficient directions that make practical sense for weather data.

## Notes on interpretation

- Significant coefficients (small p-values) indicate likely dependence between Y and that Xi after accounting for the other variables included in the model.
- R^2 / adjusted R^2 compare fit quality while penalizing unnecessary complexity.
- Residual diagnostics aim to validate linear model assumptions (especially approximate normality).
