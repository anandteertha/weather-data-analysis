from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from src.regression_utils import (
    Y_COL,
    X_COLS,
    load_weather_data,
    iqr_outlier_filter,
    ols_fit,
    plot_regression_line,
    plot_qq,
    plot_residual_hist,
    chi_square_normal_gof,
    plot_residual_scatter,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_csv = root.parent / "project-2-weather-data-analysis" / "task1" / "weather_data.csv"
    out_dir = root / "results" / "task2"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_weather_data(data_csv, n=100)
    df = iqr_outlier_filter(df_raw, cols=[Y_COL] + X_COLS, k=1.5)

    x1 = df[X_COLS[0]].to_numpy(dtype=float).reshape(-1, 1)  # X1 only
    y = df[Y_COL].to_numpy(dtype=float)

    # Task 2.1 + 2.2: Y = a0 + a1 X1 + eps
    fit = ols_fit(x1, y, add_intercept=True)
    coef_names = ["a0_intercept", "a1_X1"]
    fit_dict = fit.to_dict(coef_names=coef_names)

    with open(out_dir / "simple_linear_regression_summary.json", "w", encoding="utf-8") as f:
        json.dump(fit_dict, f, indent=2)

    # Task 2.3: Plot regression line against data.
    plot_regression_line(
        x=x1.reshape(-1),
        y=y,
        y_hat=fit.y_hat,
        out_path=plots_dir / "regression_line_simple.png",
        title="Task 2.3: Simple Linear Regression (Y vs X1)",
    )

    # Task 2.4: Residuals analysis.
    plot_qq(
        residuals=fit.residuals,
        sigma2_hat=fit.sigma2_hat,
        out_path=plots_dir / "qq_plot_residuals_simple.png",
        title="Task 2.4a: Q-Q plot of residuals vs N(0, sigma^2)",
    )
    plot_residual_hist(
        residuals=fit.residuals,
        out_path=plots_dir / "hist_residuals_simple.png",
        title="Task 2.4a: Residual histogram (with normal overlay)",
        sigma2_hat=fit.sigma2_hat,
    )
    chi_res = chi_square_normal_gof(fit.residuals, sigma2_hat=fit.sigma2_hat, n_bins=8)
    with open(out_dir / "chi_square_normal_gof_simple.json", "w", encoding="utf-8") as f:
        json.dump(chi_res, f, indent=2)

    # scatter plot of residuals for correlation trends
    plot_residual_scatter(
        x=fit.y_hat,
        residuals=fit.residuals,
        out_path=plots_dir / "residual_scatter_vs_fitted_simple.png",
        title="Task 2.4b: Residual scatter vs fitted values",
    )

    # Task 2.5: Higher-order polynomial regression (quadratic in X1).
    x1_vec = x1.reshape(-1)
    X_poly = np.column_stack([x1_vec, x1_vec**2])
    fit_poly = ols_fit(X_poly, y, add_intercept=True)
    poly_coef_names = ["a0_intercept", "a1_X1", "a2_X1_squared"]
    fit_poly_dict = fit_poly.to_dict(coef_names=poly_coef_names)

    with open(out_dir / "polynomial_regression_summary.json", "w", encoding="utf-8") as f:
        json.dump(fit_poly_dict, f, indent=2)

    # Plot the quadratic regression curve
    order = np.argsort(x1_vec)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter(x1_vec, y, alpha=0.7, label="Data")
    plt.plot(x1_vec[order], fit_poly.y_hat[order], color="darkred", linewidth=2.0, label="Quadratic fit")
    plt.title("Task 2.5: Quadratic Regression (Y vs X1 and X1^2)")
    plt.xlabel("X1 (Temperature, C)")
    plt.ylabel("Y (Apparent Temperature, C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "regression_curve_quadratic.png", dpi=150)
    plt.close()

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_raw": int(len(df_raw)),
                "n_filtered": int(len(df)),
                "simple_linear_fit": fit_dict,
                "quadratic_fit": fit_poly_dict,
                "chi_square_normal_gof": chi_res,
            },
            f,
            indent=2,
        )

    print("Task 2 complete.")
    print(f"Raw n={len(df_raw)}; Filtered n={len(df)}")
    print("Saved results under results/task2/")


if __name__ == "__main__":
    main()

