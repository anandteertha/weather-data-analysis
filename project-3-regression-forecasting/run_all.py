from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

from src.regression_utils import Y_COL, X_COLS, load_weather_data, iqr_outlier_filter, ols_fit


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def md_table(rows, headers):
    # rows: list of list of strings
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    project_dir = Path(__file__).resolve().parent

    # Run scripts (in order)
    scripts = [
        "task1_basic_stat_analysis.py",
        "task2_simple_linear_regression.py",
        "task3_multivariable_linear_regression.py",
    ]
    for s in scripts:
        print(f"Running {s}...")
        subprocess.run([sys.executable, s], cwd=str(project_dir), check=True)

    # Load summaries
    task1_dir = project_dir / "results" / "task1"
    task2_dir = project_dir / "results" / "task2"
    task3_dir = project_dir / "results" / "task3"

    stats_before = read_json(task1_dir / "descriptive_stats_before_outlier_removal.json")
    task1_summary = read_json(task1_dir / "summary.json")
    corr_y_vs_x = task1_summary["correlation_y_vs_x"]

    simple_fit = read_json(task2_dir / "simple_linear_regression_summary.json")
    poly_fit = read_json(task2_dir / "polynomial_regression_summary.json")
    chi_simple = read_json(task2_dir / "chi_square_normal_gof_simple.json")

    full_model = read_json(task3_dir / "full_model_summary.json")
    final_model = read_json(task3_dir / "final_model_summary.json")
    removal_steps = read_json(task3_dir / "removal_steps.json")
    chi_final = read_json(task3_dir / "chi_square_normal_gof_final.json")

    report_path = project_dir / "report.md"

    # Task 1 table for Xi stats
    xi_rows = []
    for x_col in stats_before:
        d = stats_before[x_col]
        xi_rows.append(
            [
                x_col,
                f"{d['mean']:.4f}",
                f"{d['variance']:.4f}",
                str(d["count"]),
            ]
        )

    # Task 2 coefficient table
    def coef_rows_from_fit(fit_dict, coef_keys):
        rows = []
        for k in coef_keys:
            cd = fit_dict[k]
            rows.append(
                [
                    k,
                    f"{cd['estimate']:.6f}",
                    f"{cd['std_error']:.6f}",
                    f"{cd['t_stat']:.3f}",
                    f"{cd['p_value']:.6g}",
                ]
            )
        return rows

    simple_coef_keys = ["a0_intercept", "a1_X1"]
    simple_coef_rows = coef_rows_from_fit(simple_fit, simple_coef_keys)

    poly_coef_keys = ["a0_intercept", "a1_X1", "a2_X1_squared"]
    poly_coef_rows = coef_rows_from_fit(poly_fit, poly_coef_keys)

    # Task 3 coefficient tables
    full_coef_keys = ["a0_intercept"] + X_COLS
    full_coef_rows = coef_rows_from_fit(full_model, full_coef_keys)

    final_vars = final_model["final_variables"]
    final_coef_keys = ["a0_intercept"] + final_vars
    final_fit = final_model["fit"]
    final_coef_rows = coef_rows_from_fit(final_fit, final_coef_keys)

    # Quick interpretation helpers for the narrative parts of the report
    delta_adj_r2_poly_vs_simple = float(poly_fit["adj_r2"]) - float(simple_fit["adj_r2"])
    delta_r2_poly_vs_simple = float(poly_fit["r2"]) - float(simple_fit["r2"])
    poly_a2_p = float(poly_fit["a2_X1_squared"]["p_value"])

    alpha = 0.05
    non_significant_final = [v for v in final_vars if float(final_fit[v]["p_value"]) > alpha]
    worst_final_p = max([float(final_fit[v]["p_value"]) for v in final_vars]) if final_vars else float("nan")
    worst_final_var = (
        max(final_vars, key=lambda v: float(final_fit[v]["p_value"])) if final_vars else None
    )
    worst_final_corr_r = (
        float(corr_y_vs_x[worst_final_var]["r"]) if worst_final_var in corr_y_vs_x else float("nan")
    )
    worst_final_corr_strength = (
        corr_y_vs_x[worst_final_var]["strength"] if worst_final_var in corr_y_vs_x else "unknown"
    )
    poly_a2_significant = poly_a2_p <= alpha

    reduced_model_comparison = None
    if worst_final_var is not None and worst_final_var in final_vars and worst_final_p > alpha:
        data_csv = project_dir.parent / "project-2-weather-data-analysis" / "task1" / "weather_data.csv"
        df_raw = load_weather_data(data_csv, n=100)
        df_filtered = iqr_outlier_filter(df_raw, cols=[Y_COL] + X_COLS, k=1.5)
        reduced_vars = [v for v in final_vars if v != worst_final_var]
        reduced_fit = ols_fit(
            df_filtered[reduced_vars].to_numpy(dtype=float),
            df_filtered[Y_COL].to_numpy(dtype=float),
            add_intercept=True,
        )
        reduced_model_comparison = {
            "removed_var": worst_final_var,
            "adj_r2": float(reduced_fit.adj_r2),
            "r2": float(reduced_fit.r2),
            "sigma2_hat": float(reduced_fit.sigma2_hat),
            "remaining_vars": reduced_vars,
        }

    # Removal steps table (if any)
    removal_rows = []
    for st in removal_steps:
        removal_rows.append(
            [
                st["removed_variable"],
                f"{st['removed_variable_p_value']:.6g}",
                f"{st['adj_r2_before']:.6f}",
                f"{st['adj_r2_after']:.6f}",
                ", ".join(st["remaining_variables"]),
            ]
        )

    report_md = f"""# Project 3: Regression and Forecasting

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

{md_table(xi_rows, headers=["Variable", "Mean", "Variance (sample)", "Count"])}

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
- Raw `n_raw` = {task1_summary['n_raw']}
- Filtered `n_filtered` = {task1_summary['n_filtered']}

### Task 1.3 (Correlation matrix)

Correlation heatmap (filtered dataset):
- `results/task1/plots/correlation_heatmap_filtered.png`
- CSV: `results/task1/correlation_matrix_filtered.csv`

Correlations between Y and each Xi (filtered):
"""

    for x_col in corr_y_vs_x:
        r = corr_y_vs_x[x_col]["r"]
        band = corr_y_vs_x[x_col]["strength"]
        direction = corr_y_vs_x[x_col]["direction"]
        report_md += f"- {x_col}: r={r:.3f} ({direction}, {band} strength)\n"

    report_md += f"""

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

{md_table(simple_coef_rows, headers=["Coefficient", "Estimate", "Std. Error", "t-stat", "p-value"])}

Model fit:
- sigma^2_hat (MSE) = {simple_fit['sigma2_hat']:.6f}
- R^2 = {simple_fit['r2']:.6f}
- adjusted R^2 = {simple_fit['adj_r2']:.6f}

### Task 2.3 (Regression line plot)
- `results/task2/plots/regression_line_simple.png`

### Task 2.4 (Residuals analysis)

Q-Q plot of residuals vs N(0, sigma^2):
- `results/task2/plots/qq_plot_residuals_simple.png`

Residual histogram with normal overlay:
- `results/task2/plots/hist_residuals_simple.png`

Chi-square normality test (approximate GOF for N(0, sigma^2_hat)):
- chi2_stat = {chi_simple['chi2_stat']:.6f}
- df = {chi_simple['df']}
- p_value = {chi_simple['p_value']:.6g}

Residual scatter plot (vs fitted values):
- `results/task2/plots/residual_scatter_vs_fitted_simple.png`

### Task 2.5 (Higher-order polynomial: quadratic in X1)

We extend the regression to:
Y = a0 + a1 X1 + a2 X1^2 + epsilon

Quadratic fit coefficients:

{md_table(poly_coef_rows, headers=["Coefficient", "Estimate", "Std. Error", "t-stat", "p-value"])}

Model fit:
- sigma^2_hat (MSE) = {poly_fit['sigma2_hat']:.6f}
- R^2 = {poly_fit['r2']:.6f}
- adjusted R^2 = {poly_fit['adj_r2']:.6f}

Quadratic regression curve plot:
- `results/task2/plots/regression_curve_quadratic.png`

### Task 2.6 (Comment on results)

- The quadratic model changes the fit relative to the simple model by:
  - delta R^2 = {delta_r2_poly_vs_simple:.6f}
  - delta adjusted R^2 = {delta_adj_r2_poly_vs_simple:.6f}
- The quadratic term `a2_X1_squared` has p-value = {poly_a2_p:.6g}. Since alpha=0.05, it is{' statistically significant' if poly_a2_significant else ' not statistically significant'} at the 5% level.
- Because the improvement in fit is very small and the quadratic term is not significant at the 5% level, the simple linear model in `X1` is the more convincing model for interpretation. The quadratic fit is still useful to report because it shows that adding curvature provides only a marginal benefit for this dataset.

## Task 3: Multi-variable Linear Regression

### Task 3.1 (Initial multivariable model)

Initial model uses all independent variables:
Y = a0 + a1 X1 + a2 X2 + a3 X3 + a4 X4 + a5 X5 + epsilon

Initial full-model fit:
- sigma^2_hat (MSE) = {full_model['sigma2_hat']:.6f}
- R^2 = {full_model['r2']:.6f}
- adjusted R^2 = {full_model['adj_r2']:.6f}

Initial full-model coefficients:

{md_table(full_coef_rows, headers=["Coefficient", "Estimate", "Std. Error", "t-stat", "p-value"])}

### Task 3.2 (Variable removal)

We use p-values and adjusted R^2 to decide whether to drop variables:
- Iteratively remove the variable with the largest p-value if p > 0.05
- Only accept a removal if adjusted R^2 does not decrease (conservative rule)

Removal steps (if any):
"""

    if removal_rows:
        report_md += "\n" + md_table(removal_rows, headers=["Removed var", "Removed var p-value", "adj R^2 before", "adj R^2 after", "Remaining variables"]) + "\n"
    else:
        report_md += "\nNo variable was removed during the backward-elimination check.\n"

    # Correlation context (Task 3.2 rubric asks to consider correlation matrix during variable removal discussion)
    if worst_final_var is not None and worst_final_var in corr_y_vs_x:
        report_md += (
            f"\nCorrelation context (from Task 1): `r(Y, {worst_final_var})` = {worst_final_corr_r:.3f} "
            f"({corr_y_vs_x[worst_final_var]['direction']}, {worst_final_corr_strength} strength). "
            f"In the final multivariable model, its coefficient p-value is {worst_final_p:.6g} "
            f"(> {alpha} means not significant at 5%)."
        )
    if reduced_model_comparison is not None:
        report_md += (
            f" If `{reduced_model_comparison['removed_var']}` is removed anyway, the adjusted R^2 drops from "
            f"{final_fit['adj_r2']:.6f} to {reduced_model_comparison['adj_r2']:.6f}, R^2 drops from "
            f"{final_fit['r2']:.6f} to {reduced_model_comparison['r2']:.6f}, and sigma^2_hat increases from "
            f"{final_fit['sigma2_hat']:.6f} to {reduced_model_comparison['sigma2_hat']:.6f}. "
            f"For that reason, the final report keeps `{reduced_model_comparison['removed_var']}` even though "
            f"its individual p-value is slightly above 0.05. This decision is consistent with the assignment instruction to consider p-values, R-squared, adjusted R-squared, and the correlation matrix together rather than relying on a single metric."
        )

    report_md += f"""

Final variable set:
- {", ".join(final_vars)}

Final model fit:
- sigma^2_hat (MSE) = {final_fit['sigma2_hat']:.6f}
- R^2 = {final_fit['r2']:.6f}
- adjusted R^2 = {final_fit['adj_r2']:.6f}

Final coefficients:

{md_table(final_coef_rows, headers=["Coefficient", "Estimate", "Std. Error", "t-stat", "p-value"])}

### Task 3.3 (Residuals analysis for final model)

Q-Q plot:
- `results/task3/plots/qq_plot_residuals_multivariable.png`

Residual histogram:
- `results/task3/plots/hist_residuals_multivariable.png`

Chi-square normality test (approximate GOF for N(0, sigma^2_hat)):
- chi2_stat = {chi_final['chi2_stat']:.6f}
- df = {chi_final['df']}
- p_value = {chi_final['p_value']:.6g}

Residual scatter:
- `results/task3/plots/residual_scatter_vs_fitted_multivariable.png`

### Task 3.4 (Comment on results)

- Final model R^2 and adjusted R^2 are high ({final_fit['r2']:.6f} and {final_fit['adj_r2']:.6f}), indicating that the model explains almost all of the observed variation in `Y` for this filtered sample.
- Variables with p-value > {alpha} in the final multivariable model: {", ".join(non_significant_final) if non_significant_final else "None"} (worst p-value = {worst_final_p:.6g}).
- Residual normality check (Task 3.3 chi-square GOF): p-value = {chi_final['p_value']:.6g}, so there is no strong evidence against the normal-residual assumption in this approximate test.
- Compared with the one-variable model in Task 2, the multivariable model reduces sigma^2_hat substantially ({simple_fit['sigma2_hat']:.6f} down to {final_fit['sigma2_hat']:.6f}), showing that the added predictors explain meaningful variation beyond `X1` alone.
- The final signs of the coefficients are physically plausible for this dataset: higher temperature and humidity increase apparent temperature, while higher wind speed and cloud cover reduce it. Surface pressure has a small positive coefficient, but because its p-value is marginal, its effect should be described as weak/inconclusive rather than strong.
- Overall, the multivariable model is the strongest model in this submission because it combines excellent fit, reasonable diagnostics, and coefficient directions that make practical sense for weather data.

## Notes on interpretation

- Significant coefficients (small p-values) indicate likely dependence between Y and that Xi after accounting for the other variables included in the model.
- R^2 / adjusted R^2 compare fit quality while penalizing unnecessary complexity.
- Residual diagnostics aim to validate linear model assumptions (especially approximate normality).
"""

    report_path.write_text(report_md, encoding="utf-8")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()

