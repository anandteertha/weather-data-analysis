import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

VARS = ["X1_temperature_2m_C", "X2_relative_humidity_2m_pct", "X3_surface_pressure_hPa",
        "X4_wind_speed_10m_kmh", "X5_cloud_cover_pct"]
VAR_LABELS = ["X1 (Temp °C)", "X2 (Humidity %)", "X3 (Pressure hPa)", "X4 (Wind km/h)", "X5 (Cloud %)"]

DATA_PATH = Path(__file__).parent.parent / "task1" / "weather_data.csv"


def main():
    out_dir = Path(__file__).parent
    df = pd.read_csv(DATA_PATH).head(100)
    y_col = "Y_apparent_temperature_C"

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (var, label) in enumerate(zip(VARS, VAR_LABELS)):
        axes[i].boxplot(df[var], vert=True)
        axes[i].set_title(f"Boxplot of {label}")
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.3)

    axes[5].axis("off")
    plt.suptitle("Boxplots for Variables X1-X5", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "task3_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved boxplots to task3/task3_boxplots.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (var, label) in enumerate(zip(VARS, VAR_LABELS)):
        axes[i].scatter(df[var], df[y_col], alpha=0.6, s=30)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Y (Apparent Temp °C)")
        axes[i].set_title(f"Y vs {label}")
        axes[i].grid(True, alpha=0.3)

    axes[5].axis("off")
    plt.suptitle("Scatter Plots - Y vs Each Variable", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "task3_scatter_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved scatter plots to task3/task3_scatter_plots.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_data = df[y_col].dropna()
    y_data.hist(density=True, bins=25, alpha=0.5, label="Histogram", color="steelblue", edgecolor="black")
    kde = scipy_stats.gaussian_kde(y_data)
    x_range = np.linspace(y_data.min(), y_data.max(), 200)
    ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="Density curve")
    ax.set_xlabel("Y (Apparent Temperature °C)")
    ax.set_ylabel("Density")
    ax.set_title("Density Curve of Output Variable Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "task3_density_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved density curve to task3/task3_density_curve.png")


if __name__ == "__main__":
    main()
