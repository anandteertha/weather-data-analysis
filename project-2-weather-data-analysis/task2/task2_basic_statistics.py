import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

VARS = ["X1_temperature_2m_C", "X2_relative_humidity_2m_pct", "X3_surface_pressure_hPa",
        "X4_wind_speed_10m_kmh", "X5_cloud_cover_pct"]
VAR_LABELS = ["X1 (Temp °C)", "X2 (Humidity %)", "X3 (Pressure hPa)", "X4 (Wind km/h)", "X5 (Cloud %)"]

DATA_PATH = Path(__file__).parent.parent / "task1" / "weather_data.csv"


def load_data(n=100):
    df = pd.read_csv(DATA_PATH)
    return df.head(n)


def descriptive_statistics(df):
    stats = []
    for var in VARS:
        data = df[var].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        stats.append({
            "Variable": var,
            "Mean": data.mean(),
            "Median": data.median(),
            "Variance": data.var(),
            "Range": data.max() - data.min(),
            "IQR": q3 - q1
        })
    return pd.DataFrame(stats)


def compute_pmf(data, n_bins=50):
    data = data.dropna()
    vmin, vmax = data.min(), data.max()
    bins = np.linspace(vmin, vmax, n_bins + 1)
    counts, _ = np.histogram(data, bins=bins)
    probs = counts / counts.sum()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, probs


def plot_pmf(df):
    out_dir = Path(__file__).parent

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (var, label) in enumerate(zip(VARS, VAR_LABELS)):
        bin_centers, probs = compute_pmf(df[var])
        width = (bin_centers[1] - bin_centers[0]) * 0.9
        axes[i].bar(bin_centers, probs, width=width, alpha=0.7, edgecolor="navy")
        axes[i].set_title(f"PMF of {label}")
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Probability")
        axes[i].grid(True, alpha=0.3)

    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "task2_pmf_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PMF plots to task2/task2_pmf_plots.png")


def main():
    out_dir = Path(__file__).parent
    df = load_data(100)

    stats_df = descriptive_statistics(df)
    stats_df.to_csv(out_dir / "task2_descriptive_statistics.csv", index=False)
    print("\n=== Descriptive Statistics (100 samples) ===")
    print(stats_df.to_string(index=False))

    plot_pmf(df)

    df.head(20).to_csv(out_dir / "manual_20_samples.csv", index=False)
    print("\nFirst 20 samples saved to task2/manual_20_samples.csv")


if __name__ == "__main__":
    main()
