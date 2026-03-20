from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict, Any, Tuple

import numpy as np

from src.regression_utils import (
    Y_COL,
    X_COLS,
    load_weather_data,
    iqr_outlier_filter,
    ols_fit,
    plot_qq,
    plot_residual_hist,
    chi_square_normal_gof,
    plot_residual_scatter,
)


def fit_model(df, feature_cols: List[str]) -> Tuple[Any, Dict[str, Any]]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[Y_COL].to_numpy(dtype=float)
    fit = ols_fit(X, y, add_intercept=True)
    coef_names = ["a0_intercept"] + feature_cols
    return fit, fit.to_dict(coef_names=coef_names)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_csv = root.parent / "project-2-weather-data-analysis" / "task1" / "weather_data.csv"
    out_dir = root / "results" / "task3"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_weather_data(data_csv, n=100)
    df = iqr_outlier_filter(df_raw, cols=[Y_COL] + X_COLS, k=1.5)

    # Step 3.1: full multivariable regression
    full_fit, full_dict = fit_model(df, X_COLS)

    # Step 3.2: remove variables if needed (p-values / adj-R2 guided)
    alpha = 0.05
    current_vars = X_COLS.copy()
    current_fit = full_fit
    removal_steps: List[Dict[str, Any]] = []

    def pvalue_map(fit_obj, vars_):
        # fit_obj.p_values includes intercept at index 0
        return {v: float(p) for v, p in zip(vars_, fit_obj.p_values[1:])}

    while len(current_vars) >= 2:
        pv_map = pvalue_map(current_fit, current_vars)
        worst_var, worst_p = max(pv_map.items(), key=lambda kv: kv[1])

        if np.isnan(worst_p) or worst_p <= alpha:
            break

        candidate_vars = [v for v in current_vars if v != worst_var]
        candidate_fit, _ = fit_model(df, candidate_vars)

        # Conservative decision rule: only remove if adj-R2 does not decrease.
        if candidate_fit.adj_r2 >= current_fit.adj_r2 - 1e-6:
            removal_steps.append(
                {
                    "removed_variable": worst_var,
                    "removed_variable_p_value": worst_p,
                    "adj_r2_before": float(current_fit.adj_r2),
                    "adj_r2_after": float(candidate_fit.adj_r2),
                    "remaining_variables": candidate_vars,
                }
            )
            current_vars = candidate_vars
            current_fit = candidate_fit
        else:
            break

    final_fit = current_fit
    final_vars = current_vars
    final_dict = final_fit.to_dict(coef_names=["a0_intercept"] + final_vars)

    # correlation matrix for interpretation support
    corr = df[[Y_COL] + final_vars].corr(method="pearson")

    with open(out_dir / "full_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_dict, f, indent=2)
    with open(out_dir / "removal_steps.json", "w", encoding="utf-8") as f:
        json.dump(removal_steps, f, indent=2)

    with open(out_dir / "final_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_variables": final_vars,
                "model": "Y = a0 + sum(ai*Xi) + eps",
                "fit": final_dict,
                "correlation_matrix_final_variables": corr.to_dict(),
            },
            f,
            indent=2,
        )

    # Step 3.3 residual analysis for final model
    plot_qq(
        residuals=final_fit.residuals,
        sigma2_hat=final_fit.sigma2_hat,
        out_path=plots_dir / "qq_plot_residuals_multivariable.png",
        title="Task 3.3a: Q-Q plot residuals vs N(0, sigma^2)",
    )
    plot_residual_hist(
        residuals=final_fit.residuals,
        out_path=plots_dir / "hist_residuals_multivariable.png",
        title="Task 3.3a: Residual histogram (with normal overlay)",
        sigma2_hat=final_fit.sigma2_hat,
    )

    chi_res = chi_square_normal_gof(final_fit.residuals, sigma2_hat=final_fit.sigma2_hat, n_bins=8)
    with open(out_dir / "chi_square_normal_gof_final.json", "w", encoding="utf-8") as f:
        json.dump(chi_res, f, indent=2)

    # scatter plot of residuals for correlation trends (vs fitted)
    plot_residual_scatter(
        x=final_fit.y_hat,
        residuals=final_fit.residuals,
        out_path=plots_dir / "residual_scatter_vs_fitted_multivariable.png",
        title="Task 3.3b: Residual scatter vs fitted values",
    )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_raw": int(len(df_raw)),
                "n_filtered": int(len(df)),
                "full_model_variables": X_COLS,
                "final_model_variables": final_vars,
                "removal_steps": removal_steps,
                "chi_square_normal_gof_final": chi_res,
            },
            f,
            indent=2,
        )

    print("Task 3 complete.")
    print(f"Raw n={len(df_raw)}; Filtered n={len(df)}")
    print(f"Final variables: {final_vars}")


if __name__ == "__main__":
    main()

