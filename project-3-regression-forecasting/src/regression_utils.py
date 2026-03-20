from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


Y_COL = "Y_apparent_temperature_C"
X_COLS = [
    "X1_temperature_2m_C",
    "X2_relative_humidity_2m_pct",
    "X3_surface_pressure_hPa",
    "X4_wind_speed_10m_kmh",
    "X5_cloud_cover_pct",
]


def load_weather_data(data_csv_path: Path, n: int = 100) -> pd.DataFrame:
    """
    Project 2 uses the first 100 daily noon observations.
    """
    df = pd.read_csv(data_csv_path)
    if n is not None:
        df = df.head(n)
    return df.copy()


def iqr_outlier_filter(
    df: pd.DataFrame,
    cols: Iterable[str],
    k: float = 1.5,
) -> pd.DataFrame:
    """
    Removes rows where *any* specified column lies outside [Q1-k*IQR, Q3+k*IQR].
    This is applied conservatively (k=1.5).
    """
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        x = df[c].dropna()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - k * iqr
        high = q3 + k * iqr
        mask &= df[c].between(low, high)
    return df.loc[mask].copy()


def ols_fit(X: np.ndarray, y: np.ndarray, add_intercept: bool = True) -> "OLSResult":
    """
    Ordinary least squares with homoscedastic Gaussian residual assumption.
    Computes coefficients, sigma^2, standard errors, t p-values, R^2, adj-R^2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if add_intercept:
        X_design = np.column_stack([np.ones(len(y)), X])
    else:
        X_design = X

    # Beta = (X'X)^-1 X'y
    xtx = X_design.T @ X_design
    xtx_inv = np.linalg.inv(xtx)
    beta_hat = xtx_inv @ (X_design.T @ y)

    y_hat = X_design @ beta_hat
    residuals = y - y_hat

    n = len(y)
    p = X_design.shape[1]  # includes intercept if present
    df_resid = n - p

    SSE = float(residuals.T @ residuals)
    sigma2_hat = SSE / df_resid if df_resid > 0 else np.nan

    cov_beta = sigma2_hat * xtx_inv
    se_beta = np.sqrt(np.diag(cov_beta))

    t_stats = beta_hat / se_beta
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df_resid) if df_resid > 0 else np.full_like(t_stats, np.nan)

    y_mean = float(np.mean(y))
    SST = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - (SSE / SST) if SST > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1.0) / df_resid if df_resid > 0 else np.nan

    return OLSResult(
        beta_hat=beta_hat,
        y_hat=y_hat,
        residuals=residuals,
        sigma2_hat=sigma2_hat,
        se_beta=se_beta,
        t_stats=t_stats,
        p_values=p_values,
        r2=r2,
        adj_r2=adj_r2,
        n=n,
        p=p,
        df_resid=df_resid,
    )


@dataclass
class OLSResult:
    beta_hat: np.ndarray
    y_hat: np.ndarray
    residuals: np.ndarray
    sigma2_hat: float
    se_beta: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    r2: float
    adj_r2: float
    n: int
    p: int
    df_resid: int

    def to_dict(self, coef_names: Optional[List[str]] = None) -> Dict[str, Any]:
        if coef_names is None:
            coef_names = [f"beta_{i}" for i in range(len(self.beta_hat))]
        out = {
            "n": self.n,
            "p": self.p,
            "df_resid": self.df_resid,
            "sigma2_hat": float(self.sigma2_hat),
            "r2": float(self.r2),
            "adj_r2": float(self.adj_r2),
        }
        for name, b, se, t, pv in zip(coef_names, self.beta_hat, self.se_beta, self.t_stats, self.p_values):
            out[name] = {
                "estimate": float(b),
                "std_error": float(se),
                "t_stat": float(t),
                "p_value": float(pv),
            }
        return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _x_label(x_col: str) -> str:
    mapping = {
        "X1_temperature_2m_C": "X1 (Temp C)",
        "X2_relative_humidity_2m_pct": "X2 (Humidity %)",
        "X3_surface_pressure_hPa": "X3 (Pressure hPa)",
        "X4_wind_speed_10m_kmh": "X4 (Wind km/h)",
        "X5_cloud_cover_pct": "X5 (Cloud %)",
        Y_COL: "Y (Apparent Temp C)",
    }
    return mapping.get(x_col, x_col)


def plot_histograms(df: pd.DataFrame, cols: Iterable[str], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for c in cols:
        plt.figure(figsize=(7, 4))
        plt.hist(df[c].dropna().to_numpy(), bins="auto", alpha=0.85, edgecolor="black")
        plt.title(f"Histogram of {_x_label(c)}")
        plt.xlabel(_x_label(c))
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{c}.png", dpi=150)
        plt.close()


def plot_boxplots(df: pd.DataFrame, cols: Iterable[str], out_dir: Path, filename: str) -> None:
    ensure_dir(out_dir)
    data = [df[c].dropna().to_numpy() for c in cols]
    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=[_x_label(c) for c in cols], showfliers=True)
    plt.title("Boxplots (IQR view)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path, title: str = "Correlation matrix") -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)

    # annotate cells (small matrix only)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.iloc[i, j]
            plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_correlation_matrix(corr: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    corr.to_csv(out_path, index=True)


def plot_regression_line(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    y_hat = np.asarray(y_hat).reshape(-1)

    order = np.argsort(x)
    xs = x[order]
    ys = y_hat[order]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.7, label="Data")
    plt.plot(xs, ys, color="darkred", linewidth=2.0, label="OLS fit")
    plt.title(title)
    plt.xlabel(_x_label("X1_temperature_2m_C") if "X1" in title else "X")
    plt.ylabel(_x_label(Y_COL))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_qq(
    residuals: np.ndarray,
    sigma2_hat: float,
    out_path: Path,
    title: str,
) -> None:
    ensure_dir(out_path.parent)
    residuals = np.asarray(residuals).reshape(-1)
    std = float(np.sqrt(sigma2_hat)) if sigma2_hat >= 0 else float(np.std(residuals, ddof=1))

    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", sparams=(0, std), plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual_hist(
    residuals: np.ndarray,
    out_path: Path,
    title: str,
    sigma2_hat: float | None = None,
) -> None:
    ensure_dir(out_path.parent)
    residuals = np.asarray(residuals).reshape(-1)
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins="auto", alpha=0.85, edgecolor="black", density=True)
    # overlay fitted normal with mean 0, variance sigma2 if possible
    mu = 0.0
    if sigma2_hat is not None and np.isfinite(sigma2_hat) and sigma2_hat > 0:
        sigma = float(np.sqrt(sigma2_hat))
    else:
        sigma = float(np.std(residuals, ddof=1))
    xs = np.linspace(residuals.min(), residuals.max(), 200)
    plt.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma), color="navy", linewidth=2)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def chi_square_normal_gof(
    residuals: np.ndarray,
    sigma2_hat: float,
    n_bins: int = 8,
) -> Dict[str, Any]:
    """
    Chi-square goodness-of-fit for residuals following N(0, sigma2_hat).
    Uses a discrete binning of residuals. The test is approximate.
    """
    residuals = np.asarray(residuals).reshape(-1)
    n = len(residuals)
    if n < 10 or sigma2_hat <= 0:
        return {
            "chi2_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "n": n,
            "note": "Not enough samples or sigma2_hat <= 0; chi-square GOF not computed.",
        }

    std = float(np.sqrt(sigma2_hat))

    # Equal-width bins across observed residual range.
    rmin = float(np.min(residuals))
    rmax = float(np.max(residuals))
    if rmin == rmax:
        return {
            "chi2_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "n": n,
            "note": "Residuals are constant; chi-square GOF not computed.",
        }

    # Create candidate bin edges and ensure strictly increasing edges.
    edges = np.linspace(rmin, rmax, n_bins + 1)
    edges = np.unique(edges)
    if len(edges) < 3:
        return {
            "chi2_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "n": n,
            "note": "Too few unique bins; chi-square GOF not computed.",
        }

    obs_counts, _ = np.histogram(residuals, bins=edges)
    # Expected counts from Normal CDF differences.
    cdf_vals = stats.norm.cdf(edges, loc=0.0, scale=std)
    exp_counts = n * np.diff(cdf_vals)

    # Merge bins with very small expectations to reduce instability.
    # Combine adjacent bins until all expected counts >= 5 or only 2 bins remain.
    obs = obs_counts.astype(float)
    exp = exp_counts.astype(float)
    bin_edges = edges

    def can_merge(e: np.ndarray) -> bool:
        # We only merge a bin with its next neighbor, so the last bin cannot be the "left" merge target.
        return (len(e) > 2) and (np.any(e[:-1] < 5.0) or (e[-1] < 5.0 and np.any(e[:-1] >= 5.0)))

    # Greedy merge: pick the first bin with exp < 5. If only the last bin is too small,
    # merge it with the previous bin.
    while can_merge(exp):
        # Prefer merging with the next bin when possible.
        left_candidates = np.where(exp[:-1] < 5.0)[0]
        if len(left_candidates) > 0:
            idx = int(left_candidates[0])
        else:
            # Last bin is too small; merge it with the previous bin.
            idx = len(exp) - 2

        # merge bin idx and idx+1
        obs[idx] = obs[idx] + obs[idx + 1]
        exp[idx] = exp[idx] + exp[idx + 1]
        obs = np.delete(obs, idx + 1)
        exp = np.delete(exp, idx + 1)

        # adjust edges (only for reporting; chi-square uses obs/exp lengths)
        # after deleting a bin, also remove its upper edge so that bin_edges length stays consistent.
        if idx + 1 < len(bin_edges) - 1:
            bin_edges = np.delete(bin_edges, idx + 1)

    k = len(obs)
    if k < 2 or np.any(exp <= 0):
        return {
            "chi2_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "n": n,
            "note": "Invalid expected counts after bin merging.",
        }

    chi2_stat = float(np.sum((obs - exp) ** 2 / exp))

    # degrees of freedom: k-1 minus 1 estimated parameter (variance) in sigma2_hat
    df = int(max(1, k - 2))
    p_value = float(stats.chi2.sf(chi2_stat, df=df))

    return {
        "chi2_stat": chi2_stat,
        "df": df,
        "p_value": p_value,
        "n": n,
        "n_bins_initial": n_bins,
        "n_bins_used": k,
        "expected_min": float(np.min(exp)),
        "note": "Approximate chi-square GOF test for N(0, sigma2_hat).",
    }


def plot_residual_scatter(x: np.ndarray, residuals: np.ndarray, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    x = np.asarray(x).reshape(-1)
    residuals = np.asarray(residuals).reshape(-1)
    plt.figure(figsize=(7, 4))
    plt.scatter(x, residuals, alpha=0.7)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Predictor / fitted axis")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

