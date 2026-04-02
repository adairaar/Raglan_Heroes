"""
Bootstrap sensitivity analysis for Raglan hero pattern factor analysis.
By default uses raw binary item responses directly (no tetrachoric correlation
step). Pass --tetrachoric to compute tetrachoric correlations first and use
them as input to factor analysis instead.

Procedure:
  1. Load data and reproduce the original factor analysis (minres + varimax, 7 factors)
     fit directly on the binary item matrix (or tetrachoric correlation matrix).
  2. Run N bootstrap iterations: resample heroes with replacement, extract factors
     from the raw binary data (or tetrachoric correlations).
  3. Apply orthogonal Procrustes rotation to each bootstrap loading matrix to
     align it with the original varimax solution.
  4. Compute consistency metrics across bootstrap samples:
       - Tucker's congruence coefficient (Phi) per factor
       - Mean absolute deviation of loadings per factor
       - 95% confidence intervals on each loading
       - Factor replication rate (Phi >= 0.85 threshold)

Usage:
    python bootstrap_factor_analysis_binary.py [--n_boot 500] [--n_factors 7] [--seed 42] [--out results/] [--tetrachoric]
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data loading & preprocessing (mirrors rank_raglan.ipynb cell 100 / 155)
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/aaronadair/Downloads/raglan_hero_counts_GH.xlsx"
SHEET_NAME = "cleaned_revised_counts v3"
NUM_FEATS = 14
FEAT_NAMES = ['C1', 'C2', 'C3', 'C4', 'C6', 'C8', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17']


def load_data() -> tuple[pd.DataFrame, list, list]:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    for i in range(1, NUM_FEATS + 1):
        df[i] = df[i].apply(lambda x: 1 if x == "X" else 0)

    feature_names = list(range(1, NUM_FEATS + 1))
    # Drop constant columns (column 7 in the original)
    valid_cols = [c for c in feature_names if df[c].nunique() > 1]
    dropped = set(feature_names) - set(valid_cols)
    if dropped:
        print(f"Dropping constant columns: {dropped}")

    # FEAT_NAMES is ordered to match valid_cols positionally
    if len(valid_cols) != len(FEAT_NAMES):
        raise ValueError(f"Expected {len(FEAT_NAMES)} valid columns after dropping constants, got {len(valid_cols)}")
    return df, valid_cols, list(FEAT_NAMES)


# ---------------------------------------------------------------------------
# Tetrachoric correlation
# ---------------------------------------------------------------------------

def _tetra_pair(x: np.ndarray, y: np.ndarray) -> float:
    """
    Tetrachoric correlation for one pair of binary vectors.
    Finds rho such that the bivariate normal probability P(Z_x > tau_x, Z_y > tau_y; rho)
    equals the observed joint proportion p_xy.
    """
    p_x = x.mean()
    p_y = y.mean()
    p_xy = (x * y).mean()

    if p_x in (0.0, 1.0) or p_y in (0.0, 1.0):
        return np.nan

    tau_x = norm.ppf(1.0 - p_x)
    tau_y = norm.ppf(1.0 - p_y)

    def f(rho):
        cov = [[1.0, rho], [rho, 1.0]]
        phi2 = multivariate_normal.cdf([tau_x, tau_y], mean=[0.0, 0.0], cov=cov)
        return (p_x + p_y - 1.0 + phi2) - p_xy

    try:
        lo, hi = f(-0.9999), f(0.9999)
        if lo * hi > 0:
            return -0.9999 if abs(lo) < abs(hi) else 0.9999
        return brentq(f, -0.9999, 0.9999, xtol=1e-6)
    except Exception:
        return np.nan


def compute_tetrachoric_corr(data: np.ndarray) -> np.ndarray:
    """
    Compute tetrachoric correlation matrix from binary data using scipy.
    Returns (p x p) numpy array. Raises if any pair fails.
    """
    _, p = data.shape
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            r = _tetra_pair(data[:, i], data[:, j])
            R[i, j] = r
            R[j, i] = r

    nan_pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if np.isnan(R[i, j])]
    if nan_pairs:
        raise ValueError(f"Tetrachoric correlation failed for {len(nan_pairs)} pairs: {nan_pairs[:5]}")

    return R


# ---------------------------------------------------------------------------
# Factor analysis on raw binary data (or tetrachoric correlations)
# ---------------------------------------------------------------------------

def run_fa(data: np.ndarray, n_factors: int, use_tetrachoric: bool = False) -> np.ndarray | None:
    """
    Fit minres factor analysis with varimax rotation.
    If use_tetrachoric=True, computes tetrachoric correlations first and passes
    the correlation matrix to FactorAnalyzer. Otherwise fits directly on binary data.
    Returns (p x n_factors) loading matrix, or None on failure.
    """
    if use_tetrachoric:
        # Errors propagate so callers get a meaningful exception rather than a silent None.
        corr_matrix = compute_tetrachoric_corr(data)
        fa = FactorAnalyzer(
            n_factors=n_factors,
            rotation="varimax",
            method="minres",
            is_corr_matrix=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(corr_matrix)
        return fa.loadings_
    else:
        try:
            fa = FactorAnalyzer(
                n_factors=n_factors,
                rotation="varimax",
                method="minres",
                is_corr_matrix=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fa.fit(data)
            return fa.loadings_
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes: find rotation R that best maps source -> target.
    Returns rotated source loadings.
    """
    R, _ = orthogonal_procrustes(source, target)
    return source @ R


# ---------------------------------------------------------------------------
# Tucker's congruence coefficient
# ---------------------------------------------------------------------------

def tucker_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tucker's congruence coefficient between corresponding columns of a and b.
    Returns array of length n_factors.
    """
    num = np.sum(a * b, axis=0)
    denom = np.sqrt(np.sum(a**2, axis=0) * np.sum(b**2, axis=0))
    return num / np.where(denom == 0, 1, denom)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(df: pd.DataFrame, cols: list, n_factors: int, n_boot: int,
              original_loadings: np.ndarray, rng: np.random.Generator,
              use_tetrachoric: bool = False) -> dict:
    """
    Run bootstrap iterations. Returns dict of results arrays.
    If use_tetrachoric=True, each iteration computes tetrachoric correlations
    before factor analysis.
    """
    p = len(cols)
    boot_loadings = []
    phi_per_iter = []
    failed = 0

    for _ in tqdm(range(n_boot), desc="Bootstrap"):
        idx = rng.integers(0, len(df), size=len(df))
        boot_df = df.iloc[idx].reset_index(drop=True)

        # Skip degenerate samples (constant columns)
        boot_cols = [c for c in cols if boot_df[c].nunique() > 1]
        if len(boot_cols) < p:
            failed += 1
            continue

        data = boot_df[boot_cols].to_numpy(dtype=float)
        try:
            loadings = run_fa(data, n_factors, use_tetrachoric=use_tetrachoric)
        except Exception:
            loadings = None
        if loadings is None:
            failed += 1
            continue

        # Procrustes rotation toward original
        aligned = procrustes_align(loadings, original_loadings)
        boot_loadings.append(aligned)
        phi_per_iter.append(tucker_phi(aligned, original_loadings))

    if failed:
        print(f"  Warning: {failed}/{n_boot} bootstrap samples failed (degenerate or non-convergent).")

    boot_loadings = np.array(boot_loadings)   # shape: (n_valid, p, n_factors)
    phi_per_iter = np.array(phi_per_iter)      # shape: (n_valid, n_factors)

    return {
        "boot_loadings": boot_loadings,
        "phi_per_iter": phi_per_iter,
        "n_valid": len(boot_loadings),
        "n_failed": failed,
    }


# ---------------------------------------------------------------------------
# Consistency summary
# ---------------------------------------------------------------------------

def summarise(results: dict, original_loadings: np.ndarray,
              feat_names: list, n_factors: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    boot_loadings = results["boot_loadings"]   # (n_valid, p, n_factors)
    phi = results["phi_per_iter"]              # (n_valid, n_factors)

    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    item_labels = feat_names

    rows = []
    for fi in range(n_factors):
        mean_phi = phi[:, fi].mean()
        replication_rate = (phi[:, fi] >= 0.85).mean()
        rows.append({
            "Factor": factor_labels[fi],
            "Mean Tucker Phi": round(mean_phi, 4),
            "Phi >= 0.85 (replication rate)": f"{replication_rate:.1%}",
            "Phi < 0.85 (unstable)": f"{(~(phi[:, fi] >= 0.85)).mean():.1%}",
        })

    summary_df = pd.DataFrame(rows)

    loading_rows = []
    for ii, item in enumerate(item_labels):
        for fi, fac in enumerate(factor_labels):
            orig = original_loadings[ii, fi]
            boot_vals = boot_loadings[:, ii, fi]
            loading_rows.append({
                "Item": item,
                "Factor": fac,
                "Original Loading": round(orig, 4),
                "Bootstrap Mean": round(boot_vals.mean(), 4),
                "Bootstrap SD": round(boot_vals.std(), 4),
                "CI 2.5%": round(np.percentile(boot_vals, 2.5), 4),
                "CI 97.5%": round(np.percentile(boot_vals, 97.5), 4),
                "Covers Zero": (np.percentile(boot_vals, 2.5) <= 0 <= np.percentile(boot_vals, 97.5)),
            })
    loading_df = pd.DataFrame(loading_rows)

    return summary_df, loading_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_phi_distributions(phi: np.ndarray, n_factors: int, out_dir: str, data_label: str = "binary data, no tetrachoric", file_suffix: str = "_no_tetrachoric"):
    factor_labels = [f"F{i+1}" for i in range(n_factors)]

    ncols = 3
    nrows = int(np.ceil(n_factors / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharey=False)
    axes_flat = axes.flatten()

    for fi in range(n_factors):
        ax = axes_flat[fi]
        vals = phi[:, fi]
        ax.hist(vals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0.85, color="red", linestyle="--", linewidth=1.2, label="0.85 threshold")
        ax.axvline(vals.mean(), color="orange", linestyle="-", linewidth=1.5, label=f"mean={vals.mean():.3f}")
        ax.set_title(factor_labels[fi])
        ax.set_xlabel("Tucker's Phi")
        if fi % ncols == 0:
            ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    for fi in range(n_factors, nrows * ncols):
        axes_flat[fi].set_visible(False)

    fig.suptitle(f"Bootstrap Tucker's Congruence Coefficients per Factor\n({data_label})", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"bootstrap_phi_distributions{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_loading_ci(original_loadings: np.ndarray, boot_loadings: np.ndarray,
                    feat_names: list, n_factors: int, out_dir: str, data_label: str = "binary data, no tetrachoric", file_suffix: str = "_no_tetrachoric"):
    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    item_labels = feat_names
    p = len(feat_names)

    ncols = 3
    nrows = int(np.ceil(n_factors / ncols))
    panel_h = max(3, (p * 0.5 + 1) * 0.50)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, panel_h * nrows),
                             sharey=True)
    axes_flat = axes.flatten()

    for fi in range(n_factors):
        ax = axes_flat[fi]
        orig = original_loadings[:, fi]
        boot_vals = boot_loadings[:, :, fi]
        lo = np.percentile(boot_vals, 2.5, axis=0)
        hi = np.percentile(boot_vals, 97.5, axis=0)
        means = boot_vals.mean(axis=0)

        y = np.arange(p)
        ax.barh(y, hi - lo, left=lo, height=0.5, color="lightsteelblue", alpha=0.7, label="95% CI")
        ax.scatter(orig, y, color="black", zorder=5, s=25, label="Original")
        ax.scatter(means, y, color="orange", zorder=4, s=15, marker="D", label="Boot mean")
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(factor_labels[fi])
        ax.set_xlabel("Loading")
        if fi % ncols == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(item_labels)
        ax.legend(fontsize=7, loc="lower right")

    axes_flat[0].invert_yaxis()

    # Hide any unused subplot slots
    for fi in range(n_factors, nrows * ncols):
        axes_flat[fi].set_visible(False)

    fig.suptitle(f"Bootstrap 95% CIs on Factor Loadings (after Procrustes rotation)\n({data_label})", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"bootstrap_loading_ci{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_loading_heatmap(original_loadings: np.ndarray, boot_loadings: np.ndarray,
                         feat_names: list, n_factors: int, out_dir: str, data_label: str = "binary data, no tetrachoric", file_suffix: str = "_no_tetrachoric"):
    """Heatmap of loading SD across bootstrap samples (instability map)."""
    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    item_labels = feat_names

    sd_mat = boot_loadings.std(axis=0)   # (p, n_factors)

    fig, ax = plt.subplots(figsize=(n_factors * 1.2 + 1, len(feat_names) * 0.5 + 1))
    im = ax.imshow(sd_mat, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="Bootstrap SD")
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factor_labels)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(item_labels)
    for i in range(len(feat_names)):
        for j in range(n_factors):
            ax.text(j, i, f"{sd_mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Loading Instability (Bootstrap SD)\n({data_label})", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"bootstrap_loading_sd_heatmap{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# PCA & unrotated loadings
# ---------------------------------------------------------------------------

def get_pca_loadings(data: np.ndarray, n_factors: int) -> np.ndarray:
    """PCA loadings: eigenvectors scaled by sqrt(eigenvalue), shape (p, n_factors)."""
    pca = PCA(n_components=n_factors)
    pca.fit(data)
    return pca.components_.T * np.sqrt(pca.explained_variance_)


def get_unrotated_loadings(data: np.ndarray, n_factors: int, use_tetrachoric: bool = False) -> np.ndarray | None:
    """Minres factor loadings with no rotation, shape (p, n_factors)."""
    if use_tetrachoric:
        corr_matrix = compute_tetrachoric_corr(data)
        fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method="minres", is_corr_matrix=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(corr_matrix)
        return fa.loadings_
    else:
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method="minres", is_corr_matrix=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fa.fit(data)
            return fa.loadings_
        except Exception:
            return None


def plot_loadings_comparison(pca_loadings: np.ndarray, rotated_loadings: np.ndarray,
                              feat_names: list, n_factors: int, out_dir: str, data_label: str = "binary data", file_suffix: str = "_no_tetrachoric"):
    """Single figure with three side-by-side heatmaps: PCA | Unrotated | Varimax."""
    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    item_labels = feat_names
    p = len(feat_names)

    panels = [
        (pca_loadings,    "PCA Loadings"),
        (rotated_loadings, "Varimax-Rotated FA Loadings\n(minres + varimax)"),
    ]

    # Shared colour scale across all three panels
    all_vals = np.concatenate([m.ravel() for m, _ in panels])
    vmax = np.abs(all_vals).max()
    vmin = -vmax

    fig, axes = plt.subplots(
        1, 2,
        figsize=(n_factors * 0.55 * 2 + 2, p * 0.45 + 2),
        sharey=True,
    )

    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_factors))
        ax.set_xticklabels(factor_labels, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        for i in range(p):
            for j in range(n_factors):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if abs(mat[i, j]) < 0.6 * vmax else "white")

    axes[0].set_yticks(range(p))
    axes[0].set_yticklabels(item_labels, fontsize=8)

    fig.suptitle(f"Factor Loadings Comparison ({data_label})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    # Reserve space on the right for the colorbar, then place it in its own axes
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Loading")

    path = os.path.join(out_dir, f"loadings_comparison_heatmap{file_suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_boot", type=int, default=500, help="Number of bootstrap iterations (default: 500)")
    parser.add_argument("--n_factors", type=int, default=6, help="Number of factors (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default="bootstrap_results_binary", help="Output directory (default: bootstrap_results_binary/)")
    parser.add_argument("--tetrachoric", action="store_true", help="Compute tetrachoric correlations before factor analysis (requires pingouin)")
    args = parser.parse_args()

    use_tetrachoric = args.tetrachoric
    data_label = "tetrachoric correlations" if use_tetrachoric else "binary data, no tetrachoric"
    file_suffix = "_tetrachoric" if use_tetrachoric else "_no_tetrachoric"

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- Original analysis ---
    print("Loading data...")
    df, cols, feat_names = load_data()
    print(f"  {len(df)} heroes, {len(cols)} items: {feat_names}")

    data = df[cols].to_numpy(dtype=float)

    print(f"Running original factor analysis ({args.n_factors} factors, minres + varimax, {data_label})...")
    original_loadings = run_fa(data, args.n_factors, use_tetrachoric=use_tetrachoric)
    if original_loadings is None:
        raise RuntimeError("Original factor analysis failed to converge.")

    factor_labels = [f"F{i+1}" for i in range(args.n_factors)]
    orig_df = pd.DataFrame(original_loadings, index=feat_names, columns=factor_labels)
    print("\nOriginal loadings (target for Procrustes):")
    print(orig_df.round(3).to_string())
    orig_df.to_csv(os.path.join(args.out, f"original_loadings{file_suffix}.csv"))

    # --- Loadings comparison heatmap ---
    print("\nComputing PCA and unrotated loadings for comparison plot...")
    pca_loadings = get_pca_loadings(data, args.n_factors)
    unrotated_loadings = get_unrotated_loadings(data, args.n_factors, use_tetrachoric=use_tetrachoric)
    if unrotated_loadings is None:
        raise RuntimeError("Unrotated factor analysis failed to converge.")
    plot_loadings_comparison(pca_loadings, original_loadings, feat_names, args.n_factors, args.out, data_label=data_label, file_suffix=file_suffix)

    # --- Bootstrap ---
    print(f"\nRunning {args.n_boot} bootstrap iterations...")
    results = bootstrap(df, cols, args.n_factors, args.n_boot, original_loadings, rng, use_tetrachoric=use_tetrachoric)
    print(f"  Completed: {results['n_valid']} valid, {results['n_failed']} failed.")

    # --- Summarise ---
    summary_df, loading_df = summarise(results, original_loadings, feat_names, args.n_factors)

    print("\n=== Factor Replication Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(args.out, f"factor_replication_summary{file_suffix}.csv"), index=False)

    unstable_loadings = loading_df[loading_df["Covers Zero"] & (loading_df["Original Loading"].abs() >= 0.3)]
    if not unstable_loadings.empty:
        print(f"\n  Loadings >= |0.3| in original but CI covers zero (potentially unstable):")
        print(unstable_loadings[["Item", "Factor", "Original Loading", "CI 2.5%", "CI 97.5%"]].to_string(index=False))
    else:
        print("\n  All loadings >= |0.3| have CIs that do not cover zero.")

    loading_df.to_csv(os.path.join(args.out, f"loading_confidence_intervals{file_suffix}.csv"), index=False)

    # --- Plots ---
    print("\nGenerating plots...")
    plot_phi_distributions(results["phi_per_iter"], args.n_factors, args.out, data_label=data_label, file_suffix=file_suffix)
    plot_loading_ci(original_loadings, results["boot_loadings"], feat_names, args.n_factors, args.out, data_label=data_label, file_suffix=file_suffix)
    plot_loading_heatmap(original_loadings, results["boot_loadings"], feat_names, args.n_factors, args.out, data_label=data_label, file_suffix=file_suffix)

    print(f"\nAll results saved to: {args.out}/")


if __name__ == "__main__":
    main()
