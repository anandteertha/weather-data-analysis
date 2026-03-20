from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.regression_utils import (
    Y_COL,
    X_COLS,
    load_weather_data,
    iqr_outlier_filter,
    plot_histograms,
    plot_boxplots,
    plot_correlation_heatmap,
    save_correlation_matrix,
)


def describe_series(s: pd.Series) -> Dict[str, Any]:
    x = s.dropna().to_numpy(dtype=float)
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x, ddof=1)) if len(x) > 1 else np.nan,
        "count": int(len(x)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    data_csv = root.parent / "project-2-weather-data-analysis" / "task1" / "weather_data.csv"
    out_dir = root / "results" / "task1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the same dataset size as Project 2 (first 100 samples).
    df_raw = load_weather_data(data_csv, n=100)

    # Task 1.1: histogram, mean, variance for each Xi
    plot_histograms(df_raw, X_COLS, out_dir / "plots")

    stats_before = {c: describe_series(df_raw[c]) for c in X_COLS}
    with open(out_dir / "descriptive_stats_before_outlier_removal.json", "w", encoding="utf-8") as f:
        json.dump(stats_before, f, indent=2)

    # Save box plot to visualize outliers before filtering
    plot_boxplots(df_raw, X_COLS, out_dir / "plots", filename="boxplots_before_outlier_removal.png")

    # Task 1.2: remove outliers (conservatively via IQR)
    cols_for_filter = [Y_COL] + X_COLS
    df_filtered = iqr_outlier_filter(df_raw, cols=cols_for_filter, k=1.5)

    # Save box plot after filtering
    plot_boxplots(df_filtered, X_COLS, out_dir / "plots", filename="boxplots_after_outlier_removal.png")

    # Task 1.3: correlation matrix for Y and all Xi
    corr_cols = [Y_COL] + X_COLS
    corr = df_filtered[corr_cols].corr(method="pearson")
    save_correlation_matrix(corr, out_dir / "correlation_matrix_filtered.csv")
    plot_correlation_heatmap(corr, out_dir / "plots" / "correlation_heatmap_filtered.png")

    # Task 1.4: conclusions (saved in JSON for report use)
    # Simple interpretation bands.
    def strength_band(r: float) -> str:
        a = abs(r)
        if a >= 0.7:
            return "strong"
        if a >= 0.3:
            return "moderate"
        return "weak"

    y_corr = corr.loc[Y_COL, X_COLS].to_dict()
    conclusions = {}
    for x, r in y_corr.items():
        conclusions[x] = {
            "r": float(r),
            "direction": "positive" if r >= 0 else "negative",
            "strength": strength_band(float(r)),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_raw": int(len(df_raw)),
                "n_filtered": int(len(df_filtered)),
                "correlation_y_vs_x": conclusions,
            },
            f,
            indent=2,
        )

    print("Task 1 complete.")
    print(f"Raw n={len(df_raw)}; Filtered n={len(df_filtered)}")
    print("Saved: correlation matrix and plots under results/task1/")


if __name__ == "__main__":
    main()

