"""
Bootstrap sensitivity analysis for Raglan hero pattern IRT analysis.
Mirrors the 2PL IRT analysis from rank_raglan.ipynb using the girth library.

Procedure:
  1. Load binary data from the same Excel sheet, using only FEAT_NAMES columns.
  2. Fit the original 2PL IRT model (MML) to get target discrimination (a)
     and difficulty (b) parameters per item.
  3. Run N bootstrap iterations: resample heroes with replacement, refit 2PL.
  4. Align each bootstrap solution to the original (handle sign flips on a).
  5. Compute consistency metrics across bootstrap samples:
       - 95% confidence intervals on a and b per item
       - Bootstrap SD per parameter
       - Intraclass correlation / Pearson r between original and bootstrap
         parameter vectors (overall stability)

Optional tetrachoric mode (--tetrachoric):
  Instead of fitting girth twopl_mml on raw binary data, estimates 2PL-equivalent
  parameters from a 1-factor normal-ogive decomposition of the tetrachoric
  correlation matrix (Lord & Novick, 1968):
      a_i = lambda_i / sqrt(1 - lambda_i^2)
      b_i = -tau_i / lambda_i   where tau_i = Phi^{-1}(1 - p_i)

Usage:
    python bootstrap_irt.py [--n_boot 500] [--seed 42] [--out irt_bootstrap_results/]
                            [--tetrachoric]
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar  # fallback for ability estimation if girth unavailable
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (mirrors rank_raglan.ipynb)
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/aaronadair/Downloads/raglan_hero_counts_GH.xlsx"
SHEET_NAME = "cleaned_revised_counts v3"
NUM_FEATS = 14
FEAT_NAMES = ['C1', 'C2', 'C3', 'C4', 'C6', 'C8', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list, list]:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    for i in range(1, NUM_FEATS + 1):
        df[i] = df[i].apply(lambda x: 1 if x == "X" else 0)

    feature_names = list(range(1, NUM_FEATS + 1))
    valid_cols = [c for c in feature_names if df[c].nunique() > 1]
    dropped = set(feature_names) - set(valid_cols)
    if dropped:
        print(f"Dropping constant columns: {dropped}")

    if len(valid_cols) != len(FEAT_NAMES):
        raise ValueError(f"Expected {len(FEAT_NAMES)} valid columns, got {len(valid_cols)}")

    hero_names = df["Name"].tolist()
    return df, valid_cols, hero_names


# ---------------------------------------------------------------------------
# Tetrachoric correlation helpers
# ---------------------------------------------------------------------------

def _tetrachoric_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Compute pairwise tetrachoric correlation matrix."""
    from pyrelimri.tetrachoric_correlation import tetrachoric_corr
    p = len(cols)
    mat = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = tetrachoric_corr(df[cols[i]].values, df[cols[j]].values)
            mat[i, j] = mat[j, i] = float(np.clip(val, -1, 1))
    mat = np.where(np.isfinite(mat), mat, 0.0)
    np.fill_diagonal(mat, 1.0)
    return mat


def _make_positive_definite(mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    if eigvals.min() < 1e-6:
        eigvals = np.maximum(eigvals, 1e-6)
        mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(mat))
        mat = mat / np.outer(d, d)
        np.fill_diagonal(mat, 1.0)
    return mat


# ---------------------------------------------------------------------------
# IRT fitting (mirrors rank_raglan.ipynb cell 117)
# ---------------------------------------------------------------------------

def fit_2pl(df: pd.DataFrame, cols: list) -> dict | None:
    """
    Fit 2PL IRT model via MML using girth.
    Input data is transposed to (items x persons) as required by girth.
    Returns dict with 'Discrimination', 'Difficulty', 'Ability', or None on failure.
    """
    try:
        from girth import twopl_mml
    except ImportError:
        raise ImportError("girth is required.\nInstall with: pip install girth")

    data = df[cols].T.to_numpy()  # shape: (n_items, n_persons)

    # Skip if any item is constant (girth will fail)
    if any(data[i].std() == 0 for i in range(data.shape[0])):
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimates = twopl_mml(data)
        return estimates
    except Exception:
        return None


def fit_irt_tetrachoric(df: pd.DataFrame, cols: list) -> dict | None:
    """
    Estimate 2PL-equivalent IRT parameters via 1-factor normal-ogive FA on the
    tetrachoric correlation matrix (Lord & Novick, 1968).
        a_i = lambda_i / sqrt(1 - lambda_i^2)
        b_i = -tau_i / lambda_i   where tau_i = Phi^{-1}(1 - p_i)
    Returns dict with 'Discrimination' and 'Difficulty', or None on failure.
    """
    from factor_analyzer import FactorAnalyzer
    from scipy.special import ndtri

    if any(df[c].std() == 0 for c in cols):
        return None

    try:
        corr_mat = _tetrachoric_matrix(df, cols)
        corr_mat = _make_positive_definite(corr_mat)

        fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres", is_corr_matrix=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(corr_mat)

        lam = np.clip(fa.loadings_[:, 0], -0.9999, 0.9999)
        props = np.clip(df[cols].mean().values, 1e-4, 1.0 - 1e-4)
        tau = ndtri(1.0 - props)

        a = lam / np.sqrt(1.0 - lam ** 2)
        b = -tau / np.where(np.abs(lam) > 1e-8, lam, np.nan)

        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return None

        return {"Discrimination": a, "Difficulty": b}
    except Exception:
        return None


def fit_item_params(df: pd.DataFrame, cols: list,
                    use_tetrachoric: bool) -> dict | None:
    """Dispatch to the appropriate fitting method."""
    if use_tetrachoric:
        return fit_irt_tetrachoric(df, cols)
    return fit_2pl(df, cols)


# ---------------------------------------------------------------------------
# Sign alignment
# ---------------------------------------------------------------------------

def align_signs(boot_a: np.ndarray, boot_b: np.ndarray,
                orig_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each item, if the bootstrap discrimination has the opposite sign to the
    original, flip both a and b for that item (equivalent 2PL parameterisation).
    """
    signs = np.sign(orig_a) * np.sign(boot_a)
    boot_a = boot_a * signs
    boot_b = boot_b * signs
    return boot_a, boot_b


# ---------------------------------------------------------------------------
# Ability estimation
# ---------------------------------------------------------------------------

def estimate_ability(responses: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Estimate person ability (theta) using girth's ability_eap, matching the
    EAP method used internally by twopl_mml (via grm_mml). EAP integrates over
    a N(0,1) prior, so it returns finite estimates even for perfect/zero scores
    and avoids the MLE ceiling that produces 0 SD for top-scoring heroes.
    Falls back to bounded scipy MLE if girth is unavailable.
    responses: (n_persons, n_items) binary matrix
    Returns: (n_persons,) ability estimates
    """
    try:
        from girth import ability_eap as girth_ability_eap
        # Use EAP (Expected A Posteriori) to match what twopl_mml uses internally
        # (grm_mml calls _ability_eap_abstract). EAP integrates over a N(0,1)
        # prior, giving finite regularised estimates even for perfect/zero scores,
        # avoiding the MLE ceiling that produces 0 SD for top-scoring heroes.
        # girth expects (n_items, n_persons).
        return np.asarray(girth_ability_eap(responses.T.astype(int), b, a))
    except ImportError:
        pass

    # Fallback: bounded scipy MLE
    thetas = np.zeros(len(responses))
    for i, resp in enumerate(responses):
        if resp.sum() == 0:
            thetas[i] = -6.0
            continue
        if resp.sum() == len(resp):
            thetas[i] = 6.0
            continue
        def neg_ll(theta, r=resp):
            p = np.clip(1.0 / (1.0 + np.exp(-a * (theta - b))), 1e-9, 1.0 - 1e-9)
            return -np.sum(r * np.log(p) + (1.0 - r) * np.log(1.0 - p))
        thetas[i] = minimize_scalar(neg_ll, bounds=(-6.0, 6.0), method="bounded").x
    return thetas


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(df: pd.DataFrame, cols: list, n_boot: int,
              orig_a: np.ndarray, orig_b: np.ndarray,
              rng: np.random.Generator,
              use_tetrachoric: bool = False) -> dict:
    orig_data = df[cols].to_numpy(dtype=float)   # (n_heroes, n_items) — fixed target for ability
    boot_a_list = []
    boot_b_list = []
    boot_theta_list = []
    failed = 0
    p = len(cols)

    for _ in tqdm(range(n_boot), desc="Bootstrap"):
        idx = rng.integers(0, len(df), size=len(df))
        boot_df = df.iloc[idx].reset_index(drop=True)

        # Drop items that became constant in this resample
        boot_cols = [c for c in cols if boot_df[c].nunique() > 1]
        if len(boot_cols) < p:
            failed += 1
            continue

        estimates = fit_item_params(boot_df, boot_cols, use_tetrachoric)
        if estimates is None:
            failed += 1
            continue

        a = estimates['Discrimination']
        b = estimates['Difficulty']

        if len(a) != p or len(b) != p:
            failed += 1
            continue

        a, b = align_signs(a, b, orig_a)
        theta = estimate_ability(orig_data, a, b)
        boot_a_list.append(a)
        boot_b_list.append(b)
        boot_theta_list.append(theta)

    if failed:
        print(f"  Warning: {failed}/{n_boot} bootstrap samples failed.")

    return {
        "boot_a":     np.array(boot_a_list),      # (n_valid, n_items)
        "boot_b":     np.array(boot_b_list),      # (n_valid, n_items)
        "boot_theta": np.array(boot_theta_list),  # (n_valid, n_heroes)
        "n_valid": len(boot_a_list),
        "n_failed": failed,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(results: dict, orig_a: np.ndarray, orig_b: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    boot_a = results["boot_a"]   # (n_valid, p)
    boot_b = results["boot_b"]

    rows_a, rows_b = [], []
    for ii, name in enumerate(FEAT_NAMES):
        for param, orig_val, boot_vals, rows in [
            ("a (Discrimination)", orig_a[ii], boot_a[:, ii], rows_a),
            ("b (Difficulty)",     orig_b[ii], boot_b[:, ii], rows_b),
        ]:
            rows.append({
                "Item": name,
                "Parameter": param,
                "Original": round(orig_val, 4),
                "Bootstrap Mean": round(boot_vals.mean(), 4),
                "Bootstrap SD": round(boot_vals.std(), 4),
                "CI 2.5%": round(np.percentile(boot_vals, 2.5), 4),
                "CI 97.5%": round(np.percentile(boot_vals, 97.5), 4),
                "Covers Zero": (np.percentile(boot_vals, 2.5) <= 0 <= np.percentile(boot_vals, 97.5)),
            })

    ci_df = pd.DataFrame(rows_a + rows_b)

    # Vector-level stability: Pearson r between original and each bootstrap sample
    r_a = [pearsonr(orig_a, boot_a[i])[0] for i in range(len(boot_a))]
    r_b = [pearsonr(orig_b, boot_b[i])[0] for i in range(len(boot_b))]

    stability_df = pd.DataFrame({
        "Parameter": ["a (Discrimination)", "b (Difficulty)"],
        "Mean r with original": [round(np.mean(r_a), 4), round(np.mean(r_b), 4)],
        "SD r": [round(np.std(r_a), 4), round(np.std(r_b), 4)],
        "r >= 0.90 rate": [f"{np.mean(np.array(r_a) >= 0.9):.1%}",
                           f"{np.mean(np.array(r_b) >= 0.9):.1%}"],
    })

    return ci_df, stability_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_parameter_ci(results: dict, orig_a: np.ndarray, orig_b: np.ndarray, out_dir: str, file_suffix: str = "_no_tetrachoric", data_label: str = "twopl_mml"):
    """Two-panel CI plot: discrimination (a) on left, difficulty (b) on right."""
    boot_a = results["boot_a"]
    boot_b = results["boot_b"]
    p = len(FEAT_NAMES)
    y = np.arange(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, p * 0.5 + 1)), sharey=True)

    for ax, orig, boot, label, color in [
        (axes[0], orig_a, boot_a, "Discrimination (a)", "steelblue"),
        (axes[1], orig_b, boot_b, "Difficulty (b)",     "darkorange"),
    ]:
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        means = boot.mean(axis=0)

        ax.barh(y, hi - lo, left=lo, height=0.5, color=color, alpha=0.35, label="95% CI")
        ax.scatter(orig, y, color="black", zorder=5, s=30, label="Original")
        ax.scatter(means, y, color=color, zorder=4, s=18, marker="D", label="Boot mean")
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Parameter value")
        ax.legend(fontsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(FEAT_NAMES)
    axes[0].invert_yaxis()

    fig.suptitle(f"Bootstrap 95% CIs on 2PL IRT Parameters ({data_label})", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"irt_bootstrap_parameter_ci{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_stability_distributions(results: dict, orig_a: np.ndarray,
                                  orig_b: np.ndarray, out_dir: str, file_suffix: str = "_no_tetrachoric", data_label: str = "twopl_mml"):
    """Distributions of per-bootstrap Pearson r vs original, for a and b."""
    boot_a = results["boot_a"]
    boot_b = results["boot_b"]

    r_a = [pearsonr(orig_a, boot_a[i])[0] for i in range(len(boot_a))]
    r_b = [pearsonr(orig_b, boot_b[i])[0] for i in range(len(boot_b))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, r_vals, label, color in [
        (axes[0], r_a, "Discrimination (a)", "steelblue"),
        (axes[1], r_b, "Difficulty (b)",     "darkorange"),
    ]:
        ax.hist(r_vals, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(0.90, color="red", linestyle="--", linewidth=1.2, label="r = 0.90")
        ax.axvline(np.mean(r_vals), color="black", linestyle="-", linewidth=1.5,
                   label=f"mean = {np.mean(r_vals):.3f}")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Pearson r with original")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(f"Bootstrap Stability: Correlation of Parameter Vectors with Original 2PL Solution ({data_label})",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"irt_bootstrap_stability{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ability_stability(orig_theta: np.ndarray, boot_theta: np.ndarray,
                           out_dir: str, file_suffix: str = "_no_tetrachoric",
                           data_label: str = "twopl_mml"):
    """Distribution of per-bootstrap Pearson r vs original ability vector."""
    r_vals = [pearsonr(orig_theta, boot_theta[i])[0] for i in range(len(boot_theta))]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r_vals, bins=30, color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(0.90, color="red", linestyle="--", linewidth=1.2, label="r = 0.90")
    ax.axvline(np.mean(r_vals), color="black", linestyle="-", linewidth=1.5,
               label=f"mean = {np.mean(r_vals):.3f}")
    ax.set_title("Ability (θ)", fontweight="bold")
    ax.set_xlabel("Pearson r with original")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    fig.suptitle(f"Bootstrap Stability: \nCorrelation of Ability Vector with Original \n({data_label})",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"irt_bootstrap_ability_stability{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_parameter_heatmap(orig_a: np.ndarray, orig_b: np.ndarray,
                            boot_a: np.ndarray, boot_b: np.ndarray, out_dir: str, file_suffix: str = "_no_tetrachoric", data_label: str = "twopl_mml"):
    """Heatmap of bootstrap SD for a and b parameters (instability map)."""
    sd_a = boot_a.std(axis=0)
    sd_b = boot_b.std(axis=0)

    # Stack into (2, p) matrix: rows = [a, b], cols = items
    sd_mat = np.vstack([sd_a, sd_b])
    orig_mat = np.vstack([orig_a, orig_b])

    fig, axes = plt.subplots(1, 2, figsize=(len(FEAT_NAMES) * 0.7 + 2, 3), sharey=False)

    for ax, mat, title, cmap in [
        (axes[0], sd_mat,   "Bootstrap SD",      "YlOrRd"),
        (axes[1], orig_mat, "Original Estimates", "RdBu_r"),
    ]:
        vmax = np.abs(mat).max() if title == "Original Estimates" else mat.max()
        vmin = -vmax if title == "Original Estimates" else 0
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(FEAT_NAMES)))
        ax.set_xticklabels(FEAT_NAMES, fontsize=8, rotation=45, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["a (Discrim.)", "b (Difficulty)"], fontsize=8)
        ax.set_title(title, fontweight="bold")
        for i in range(2):
            for j in range(len(FEAT_NAMES)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.suptitle(f"2PL IRT Parameter Summary ({data_label})", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"irt_parameter_heatmap{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ability_ci(orig_theta: np.ndarray, boot_theta: np.ndarray,
                    hero_labels: list, out_dir: str,
                    file_suffix: str = "_no_tetrachoric", data_label: str = "twopl_mml"):
    """CI plot for hero ability estimates (mirrors plot_parameter_ci)."""
    n = len(hero_labels)
    y = np.arange(n)
    lo = np.percentile(boot_theta, 2.5, axis=0)
    hi = np.percentile(boot_theta, 97.5, axis=0)
    means = boot_theta.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, max(4.25, (n * 0.35 + 1) * 0.85)))
    ax.barh(y, hi - lo, left=lo, height=0.45, color="mediumpurple", alpha=0.35, label="95% CI")
    ax.scatter(orig_theta, y, color="black", zorder=5, s=30, label="Original")
    ax.scatter(means, y, color="mediumpurple", zorder=4, s=18, marker="D", label="Boot mean")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(hero_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("Ability (θ)")
    ax.set_title("Ability (θ)", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle(f"Bootstrap 95% CIs on Hero Ability Estimates ({data_label})", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"irt_bootstrap_ability_ci{file_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_boot", type=int, default=500, help="Number of bootstrap iterations (default: 500)")
    parser.add_argument("--seed",   type=int, default=42,  help="Random seed (default: 42)")
    parser.add_argument("--out",    type=str, default="irt_bootstrap_results",
                        help="Output directory (default: irt_bootstrap_results/)")
    parser.add_argument("--tetrachoric", action="store_true",
                        help="Use tetrachoric correlation + 1-factor normal-ogive to estimate "
                             "IRT parameters instead of girth twopl_mml (default: off)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    method_label = "tetrachoric" if args.tetrachoric else "twopl_mml"
    file_suffix = "_tetrachoric" if args.tetrachoric else "_no_tetrachoric"
    print(f"Fitting method: {method_label}")

    # --- Load & original fit ---
    print("Loading data...")
    df, cols, hero_names = load_data()
    print(f"  {len(df)} heroes, {len(cols)} items: {FEAT_NAMES}")

    print("Fitting original IRT model...")
    orig_estimates = fit_item_params(df, cols, args.tetrachoric)
    if orig_estimates is None:
        raise RuntimeError("Original IRT fit failed.")

    orig_a = orig_estimates['Discrimination']
    orig_b = orig_estimates['Difficulty']
    orig_data = df[cols].to_numpy(dtype=float)
    # Use the ability estimates returned by the fitting method directly when
    # available (non-tetrachoric path: girth's twopl_mml, matches raglan_analysis.py).
    # Fall back to custom MLE for the tetrachoric path where girth is not used.
    if 'Ability' in orig_estimates:
        orig_theta = orig_estimates['Ability']
    else:
        orig_theta = estimate_ability(orig_data, orig_a, orig_b)
    hero_labels = hero_names

    orig_df = pd.DataFrame(
        [orig_a, orig_b],
        index=["Discrimination (a)", "Difficulty (b)"],
        columns=FEAT_NAMES,
    )
    print("\nOriginal 2PL parameters:")
    print(orig_df.round(4).to_string())

    # --- Bootstrap ---
    print(f"\nRunning {args.n_boot} bootstrap iterations...")
    results = bootstrap(df, cols, args.n_boot, orig_a, orig_b, rng,
                        use_tetrachoric=args.tetrachoric)
    print(f"  Completed: {results['n_valid']} valid, {results['n_failed']} failed.")

    # --- Summarise ---
    ci_df, stability_df = summarise(results, orig_a, orig_b)

    # Append ability stability row (needs boot_theta, computed below)
    boot_theta = results["boot_theta"]   # (n_valid, n_heroes)
    r_theta = [pearsonr(orig_theta, boot_theta[i])[0] for i in range(len(boot_theta))]
    ability_stability_row = pd.DataFrame([{
        "Parameter": "θ (Ability)",
        "Mean r with original": round(np.mean(r_theta), 4),
        "SD r": round(np.std(r_theta), 4),
        "r >= 0.90 rate": f"{np.mean(np.array(r_theta) >= 0.9):.1%}",
    }])
    stability_df = pd.concat([stability_df, ability_stability_row], ignore_index=True)

    print("\n=== Parameter Stability (Pearson r with original) ===")
    print(stability_df.to_string(index=False))
    stability_df.to_csv(os.path.join(args.out, f"irt_stability_summary{file_suffix}.csv"), index=False)

    unstable = ci_df[ci_df["Covers Zero"] & (ci_df["Original"].abs() >= 0.3)]
    if not unstable.empty:
        print("\n  Parameters >= |0.3| in original whose CI covers zero (potentially unstable):")
        print(unstable[["Item", "Parameter", "Original", "CI 2.5%", "CI 97.5%"]].to_string(index=False))
    else:
        print("\n  All parameters >= |0.3| have CIs that do not cover zero.")

    ci_df.to_csv(os.path.join(args.out, f"irt_bootstrap_ci{file_suffix}.csv"), index=False)

    # --- Ability summary ---
    ability_rows = []
    for hi, label in enumerate(hero_labels):
        boot_vals = boot_theta[:, hi]
        ability_rows.append({
            "Hero":            label,
            "Original Ability": round(orig_theta[hi], 4),
            "Bootstrap Mean":  round(boot_vals.mean(), 4),
            "Bootstrap SD":    round(boot_vals.std(), 4),
            "CI 2.5%":         round(np.percentile(boot_vals, 2.5), 4),
            "CI 97.5%":        round(np.percentile(boot_vals, 97.5), 4),
        })
    ability_df = pd.DataFrame(ability_rows).sort_values("Original Ability", ascending=False).reset_index(drop=True)
    ability_df.to_csv(os.path.join(args.out, f"irt_ability{file_suffix}.csv"), index=False)

    # --- Augment original parameters CSV with bootstrap SD and 95% CIs ---
    boot_a = results["boot_a"]
    boot_b = results["boot_b"]
    for row_name, vals in [
        ("Discrimination (a) SD",       boot_a.std(axis=0)),
        ("Discrimination (a) CI 2.5%",  np.percentile(boot_a, 2.5, axis=0)),
        ("Discrimination (a) CI 97.5%", np.percentile(boot_a, 97.5, axis=0)),
        ("Difficulty (b) SD",           boot_b.std(axis=0)),
        ("Difficulty (b) CI 2.5%",      np.percentile(boot_b, 2.5, axis=0)),
        ("Difficulty (b) CI 97.5%",     np.percentile(boot_b, 97.5, axis=0)),
    ]:
        orig_df.loc[row_name] = np.round(vals, 4)
    orig_df.to_csv(os.path.join(args.out, f"irt_original_parameters{file_suffix}.csv"))

    # --- Plots ---
    print("\nGenerating plots...")
    plot_parameter_ci(results, orig_a, orig_b, args.out, file_suffix=file_suffix, data_label=method_label)
    plot_stability_distributions(results, orig_a, orig_b, args.out, file_suffix=file_suffix, data_label=method_label)
    plot_parameter_heatmap(orig_a, orig_b, results["boot_a"], results["boot_b"], args.out, file_suffix=file_suffix, data_label=method_label)
    sort_idx = np.argsort(orig_theta)[::-1]
    plot_ability_ci(orig_theta[sort_idx], boot_theta[:, sort_idx],
                    [hero_labels[i] for i in sort_idx],
                    args.out, file_suffix=file_suffix, data_label=method_label)
    plot_ability_stability(orig_theta, boot_theta, args.out, file_suffix=file_suffix, data_label=method_label)

    print(f"\nAll results saved to: {args.out}/")


if __name__ == "__main__":
    main()
