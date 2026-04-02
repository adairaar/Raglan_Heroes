"""
Raglan Hero Pattern Analysis
Cleaned script from rank_raglan.ipynb.

Runs the full analysis pipeline on both v2 and v3 of the Excel data and
saves all results to per-version output directories.

Pipeline:
  1.  Load & binarize data
  2.  Compute tetrachoric correlation matrix
  3.  Bartlett's sphericity test & KMO sampling adequacy
  4.  Factor analysis (minres + varimax, 7 factors) on tetrachoric correlations
  5.  2PL IRT model (MML) for item parameters and hero ability estimates
  6.  Yen's Q3 local dependence between items
  7.  Hierarchical clustering dendrograms (heroes and criteria, Hamming/average)
  8.  Total score distribution fitted with a binomial (MLE)
  9.  Chi-squared goodness-of-fit: uniform, fitted binomial, binomial(p=0.5)
  10. Save all results (CSV + plots + LaTeX)

Usage:
    python raglan_analysis.py [--out results/]
"""

import argparse
import os
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, binom, chi2_contingency, chisquare
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)

# ---------------------------------------------------------------------------
# Version configurations
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/aaronadair/Downloads/raglan_hero_counts_GH.xlsx"

VERSIONS = {
    "v2": {
        "sheet": "cleaned_revised_counts v2",
        "num_feats": 17,
        "feat_names": [f"C{i}" for i in range(1, 18)],  # C1–C17
    },
    "v3": {
        "sheet": "cleaned_revised_counts v3",
        "num_feats": 14,
        # All 14 column positions mapped to their v2 C-labels (C5, C7, C10
        # were dropped from the criteria before this sheet was created;
        # C9 is constant in this dataset and will be dropped at runtime).
        "feat_names": [
            "C1", "C2", "C3", "C4", "C6", "C8", "C9",
            "C11", "C12", "C13", "C14", "C15", "C16", "C17",
        ],
    },
}

N_FACTORS = 6

VERSION_LABELS = {
    "v2": "All Criteria",
    "v3": "Non-Redundant Criteria",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(version: str) -> tuple[pd.DataFrame, list, list]:
    """
    Load and binarize the Excel data for a given version.
    Returns (df, col_indices, feat_names) where col_indices are the integer
    column keys used in df and feat_names are the corresponding C-labels.
    """
    cfg = VERSIONS[version]
    df = pd.read_excel(DATA_PATH, sheet_name=cfg["sheet"])

    num_feats = cfg["num_feats"]
    for i in range(1, num_feats + 1):
        df[i] = df[i].apply(lambda x: 1 if x == "X" else 0)

    col_indices = list(range(1, num_feats + 1))

    # Drop constant columns
    valid_indices = [c for c in col_indices if df[c].nunique() > 1]
    dropped = set(col_indices) - set(valid_indices)
    if dropped:
        print(f"  [{version}] Dropping constant columns at positions: {dropped}")

    # Align feat_names to valid_indices by position
    all_names = cfg["feat_names"]
    if len(all_names) == len(col_indices):
        name_map = dict(zip(col_indices, all_names))
        feat_names = [name_map[c] for c in valid_indices]
    else:
        feat_names = [f"C{c}" for c in valid_indices]

    return df, valid_indices, feat_names


# ---------------------------------------------------------------------------
# Tetrachoric correlation
# ---------------------------------------------------------------------------

def tetrachoric_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Pairwise tetrachoric correlations for binary columns, returns (p x p) array."""
    try:
        from pyrelimri.tetrachoric_correlation import tetrachoric_corr
    except ImportError:
        raise ImportError("pip install pyrelimri")

    n = len(cols)
    mat = np.eye(n)
    for (i, ci), (j, cj) in itertools.combinations(enumerate(cols), 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = tetrachoric_corr(df[ci].values, df[cj].values)
        mat[i, j] = mat[j, i] = val

    mat = np.clip(mat, -1, 1)
    mat = np.where(np.isfinite(mat), mat, 0)
    np.fill_diagonal(mat, 1.0)
    return mat


def make_positive_definite(mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    if eigvals.min() < 1e-6:
        eigvals = np.maximum(eigvals, 1e-6)
        mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(mat))
        mat = mat / np.outer(d, d)
        np.fill_diagonal(mat, 1.0)
    return mat


# ---------------------------------------------------------------------------
# Factor analysis
# ---------------------------------------------------------------------------

def run_factor_analysis(corr_mat: np.ndarray, n_factors: int,
                        feat_names: list) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Minres FA with varimax rotation on tetrachoric correlation matrix.
    Returns (loadings_df, variance_df, eigenvalues).
    """
    corr_mat = make_positive_definite(corr_mat)
    fa = FactorAnalyzer(
        n_factors=n_factors, rotation="varimax",
        method="minres", is_corr_matrix=True,
        # method="minres", is_corr_matrix=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa.fit(corr_mat)

    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(fa.loadings_, index=feat_names, columns=factor_labels)
    variance_df = pd.DataFrame(
        fa.get_factor_variance(),
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=factor_labels,
    )
    eigenvalues, _ = fa.get_eigenvalues()
    return loadings_df, variance_df, eigenvalues


# ---------------------------------------------------------------------------
# IRT (2PL)
# ---------------------------------------------------------------------------

def run_irt(df: pd.DataFrame, cols: list) -> dict | None:
    """Fit 2PL IRT model via MML. Returns girth estimates dict or None."""
    try:
        from girth import twopl_mml
    except ImportError:
        raise ImportError("pip install girth")

    data = df[cols].T.to_numpy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return twopl_mml(data)
    except Exception as e:
        print(f"  IRT fit failed: {e}")
        return None


def calculate_q3(response_df: pd.DataFrame, ability: np.ndarray,
                 discrimination: np.ndarray, difficulty: np.ndarray,
                 feat_names: list) -> pd.DataFrame:
    """
    Yen's Q3 local dependence statistic for all item pairs.
    Uses 2PL expected probabilities: P = 1 / (1 + exp(-a*(theta - b))).
    """
    n_persons, n_items = response_df.shape
    theta = np.array(ability)

    theta_mat = np.tile(theta, (n_items, 1)).T          # (n_persons, n_items)
    a_mat = np.tile(discrimination, (n_persons, 1))
    b_mat = np.tile(difficulty, (n_persons, 1))

    expected = 1.0 / (1.0 + np.exp(-a_mat * (theta_mat - b_mat)))
    residuals = response_df.values - expected

    q3 = pd.DataFrame(np.zeros((n_items, n_items)), index=feat_names, columns=feat_names)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            r, _ = pearsonr(residuals[:, i], residuals[:, j])
            q3.iloc[i, j] = q3.iloc[j, i] = r

    return q3



# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def save_latex(latex_str: str, path: str):
    with open(path, "w") as f:
        f.write(latex_str)
    print(f"  Saved LaTeX: {path}")


def compute_factor_scores(df: pd.DataFrame, cols: list,
                           loadings_df: pd.DataFrame,
                           deadband: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Project binary responses onto FA loadings (from run_factor_analysis).
    deadband: fraction of each factor's max absolute loading below which a
              loading is zeroed out (0.0 = no deadbanding, 0.5 = half-max, etc.).
    Returns (scaled_scores_df, binary_components_df, deadbanded_loadings_df).
    """
    components = loadings_df.to_numpy().copy()   # (p, n_factors)

    if deadband > 0.0:
        max_abs = np.abs(components).max(axis=0)   # (n_factors,)
        threshold = max_abs * deadband
        components = np.where(np.abs(components) >= threshold, components, 0.0)

    raw_scores = df[cols].to_numpy(dtype=float) @ components  # (n_heroes, n_factors)

    factor_labels = list(loadings_df.columns)
    raw_df = pd.DataFrame(raw_scores, index=df["Name"], columns=factor_labels)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(raw_df) * np.sign(raw_df.mean().values)
    scaled_df = pd.DataFrame(scaled, index=df["Name"], columns=factor_labels)

    binary_df = pd.DataFrame(
        np.where(scaled_df >= 0, "X", ""),
        index=df["Name"], columns=factor_labels,
    ).sort_index()

    deadbanded_df = pd.DataFrame(components, index=loadings_df.index, columns=factor_labels)

    return scaled_df, binary_df, deadbanded_df


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------

def hierarchical_clustering(df: pd.DataFrame, cols: list,
                             feat_names: list, out_dir: str, version: str, ver: str = ""):
    """
    Dendrograms for both heroes (rows) and criteria (columns).
    Uses Hamming distance with average linkage, mirroring the notebook.
    Saves each as a separate file.
    """
    suffix = f"_{ver}" if ver else ""

    # --- Heroes dendrogram ---
    hero_dist = pdist(df[cols].to_numpy(dtype=float), metric="hamming")
    hero_link = linkage(hero_dist, method="average")
    fig, ax = plt.subplots(figsize=(12, 7))
    dendrogram(hero_link, labels=df["Name"].tolist(), leaf_rotation=90, ax=ax)
    ax.set_title(f"Heroes — Hamming Distance, Average Linkage ({version})")
    ax.set_xlabel("Hero")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    path = os.path.join(out_dir, f"dendrogram_heroes{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # --- Criteria dendrogram ---
    feat_dist = pdist(df[cols].T.to_numpy(dtype=float), metric="hamming")
    feat_link = linkage(feat_dist, method="average")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(feat_link, labels=feat_names, leaf_rotation=90, ax=ax)
    ax.set_title(f"Criteria — Hamming Distance, Average Linkage ({version})")
    ax.set_xlabel("Criterion")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    path = os.path.join(out_dir, f"dendrogram_criteria{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Binomial distribution fit
# ---------------------------------------------------------------------------

def fit_and_plot_binomial(scores: list, n_items: int,
                           out_dir: str, version: str, ver: str = "") -> tuple[int, float]:
    """
    Fit a binomial distribution to total hero scores via MLE, plot observed
    counts vs fitted binomial PMF on dual axes, and return (n, p).
    """
    import scipy.stats as _stats

    result = _stats.fit(binom, scores, bounds={"n": [n_items]})
    n, p = int(result.params.n), result.params.p
    print(f"  Fitted binomial: n={n}, p={p:.4f}")

    score_min = int(n // 2)
    score_max = n
    x = np.arange(score_min, score_max + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(x, binom.pmf(x, n, p), "bo", ms=7, label="Fitted binom PMF")
    ax1.vlines(x, 0, binom.pmf(x, n, p), colors="b", lw=4, alpha=0.5)
    ax1.plot(x, binom.pmf(x, n, p), "r-", lw=2, label="Fitted binom curve")
    ax1.set_ylim(ymin=0)
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper left", frameon=False)

    ax2 = ax1.twinx()
    ax2.hist(scores,
             bins=np.arange(score_min - 0.25, score_max + 0.75, 0.5),
             density=False, color="grey", alpha=0.65, label="Observed counts")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_ylabel("Count")
    ax2.legend(loc="center left", bbox_to_anchor=(0, 0.7), frameon=False)

    plt.title(f"Hero Score Distribution with Fitted Binomial ({version})")
    plt.xlabel("Total Score")
    plt.xticks(x)
    plt.tight_layout()
    suffix = f"_{ver}" if ver else ""
    path = os.path.join(out_dir, f"binomial_fit{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return n, p


# ---------------------------------------------------------------------------
# Chi-squared goodness-of-fit tests
# ---------------------------------------------------------------------------

def chi_squared_tests(scores: list, n: int, p_fitted: float,
                       out_dir: str, version: str,
                       name: str = "chi_squared_tests",
                       max_score: int = None) -> pd.DataFrame:
    """
    Three goodness-of-fit tests on the observed score distribution:
      1. Uniform distribution
      2. Fitted binomial (n, p_fitted)
      3. Binomial with p = 0.5

    Uses scipy chi2_contingency on [observed, expected] rows.
    Score range: [max_score // 2, max_score]. If max_score is not provided,
    falls back to n (the fitted binomial parameter).
    """
    if max_score is None:
        max_score = n
    score_min = int(max_score // 2) 
    score_range = range(score_min, max_score + 1)
    observed = np.array([scores.count(i) for i in score_range], dtype=float)
    tot = observed.sum()

    hypotheses = {
        "Uniform":          np.full(len(observed), tot / len(observed)),
        "Fitted binomial":  np.array([binom.pmf(x, n, p_fitted) * tot for x in score_range]),
        "Binomial (p=0.5)": np.array([binom.pmf(x, n, 0.5) * tot for x in score_range]),
    }

    k = len(score_range)
    rows = []
    for label, expected_raw in hypotheses.items():
        # Normalise expected to sum to tot
        expected = expected_raw * (tot / expected_raw.sum())
        chi2, p_val = chisquare(observed, f_exp=expected)
        dof = k - 1
        rows.append({
            "Hypothesis": label,
            "Chi-square": round(chi2, 4),
            "p-value": round(p_val, 6),
            "df": dof,
        })
        print(f"  {label}: chi2={chi2:.4f}, p={p_val:.4g}, df={dof}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    save_latex(
        result_df.to_latex(index=False, float_format="{:.4f}".format),
        os.path.join(out_dir, f"latex_{name}.txt"),
    )
    return result_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scree(eigenvalues: np.ndarray, n_factors: int, out_dir: str, version: str, ver: str = ""):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(1, len(eigenvalues) + 1)
    ax.scatter(x, eigenvalues, color="steelblue", zorder=3)
    ax.plot(x, eigenvalues, color="steelblue")
    ax.axvline(n_factors, color="red", linestyle="--", linewidth=1, label=f"{n_factors} factors retained")
    ax.axhline(1, color="grey", linestyle=":", linewidth=1, label="Eigenvalue = 1")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Scree Plot ({version})")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()
    plt.tight_layout()
    suffix = f"_{ver}" if ver else ""
    plt.savefig(os.path.join(out_dir, f"scree_plot{suffix}.png"), dpi=150)
    plt.close()


def plot_loadings_heatmap(loadings_df: pd.DataFrame, out_dir: str, version: str, ver: str = ""):
    n_factors = loadings_df.shape[1]
    fig, ax = plt.subplots(figsize=(n_factors * 1.2, len(loadings_df) * 0.5 + 1))
    sns.heatmap(
        loadings_df, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Factor Loadings — Tetrachoric, minres + varimax ({version})")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Item")
    plt.tight_layout()
    suffix = f"_{ver}" if ver else ""
    plt.savefig(os.path.join(out_dir, f"factor_loadings_heatmap{suffix}.png"), dpi=150)
    plt.close()


def plot_irt_heatmap(discrimination: np.ndarray, difficulty: np.ndarray,
                     feat_names: list, out_dir: str, version: str, ver: str = ""):
    mat = pd.DataFrame(
        [discrimination, difficulty],
        index=["Discrimination (a)", "Difficulty (b)"],
        columns=feat_names,
    )
    fig, ax = plt.subplots(figsize=(len(feat_names) * 0.9 + 2, 2.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"2PL IRT Parameters ({version})")
    plt.tight_layout()
    suffix = f"_{ver}" if ver else ""
    plt.savefig(os.path.join(out_dir, f"irt_parameters_heatmap{suffix}.png"), dpi=150)
    plt.close()


def plot_q3_heatmap(q3: pd.DataFrame, out_dir: str, version: str, ver: str = ""):
    fig, ax = plt.subplots(figsize=(len(q3) * 0.6 + 2, len(q3) * 0.6 + 1))
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones(q3.shape, dtype=bool), k=0)
    sns.heatmap(
        q3, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, mask=mask, ax=ax,
    )
    ax.set_title(f"Yen's Q3 Local Dependence ({version})")
    plt.tight_layout()
    suffix = f"_{ver}" if ver else ""
    plt.savefig(os.path.join(out_dir, f"q3_local_dependence{suffix}.png"), dpi=150)
    plt.close()



def plot_tetrachoric_heatmap(corr_mat: np.ndarray, feat_names: list,
                              out_dir: str, version: str, ver: str = ""):
    df_corr = pd.DataFrame(corr_mat, index=feat_names, columns=feat_names)
    fig, ax = plt.subplots(figsize=(len(feat_names) * 0.6 + 2, len(feat_names) * 0.6 + 1))
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Tetrachoric Correlation Matrix ({version})")
    plt.tight_layout()
    suffix = f"_{ver}" if ver else ""
    plt.savefig(os.path.join(out_dir, f"tetrachoric_correlation{suffix}.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_version(version: str, base_out: str, deadband: float = 0.0):
    out_dir = os.path.join(base_out, version)
    label = VERSION_LABELS[version]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Running analysis: {version} ({label})")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # 1. Load data
    print("\n[1] Loading data...")
    df, cols, feat_names = load_data(version)
    print(f"  {len(df)} heroes, {len(cols)} items: {feat_names}")

    # 2. Tetrachoric correlation
    print("\n[2] Computing tetrachoric correlation matrix...")
    corr_mat = tetrachoric_matrix(df, cols)
    corr_df = pd.DataFrame(corr_mat, index=feat_names, columns=feat_names)
    corr_df.to_csv(os.path.join(out_dir, f"tetrachoric_correlation_{version}.csv"))
    plot_tetrachoric_heatmap(corr_mat, feat_names, out_dir, label, ver=version)

    # 3. Bartlett & KMO (run on raw data with small jitter for numerical stability)
    print("\n[3] Bartlett's test & KMO...")
    rng = np.random.default_rng(42)
    jittered = df[cols].to_numpy(dtype=float) + rng.uniform(-0.01, 0.01, size=(len(df), len(cols)))
    jittered_df = pd.DataFrame(jittered, columns=cols)
    chi2, p_bartlett = calculate_bartlett_sphericity(jittered_df)
    _, kmo_model = calculate_kmo(jittered_df)
    print(f"  Bartlett's chi-square: {chi2:.2f},  p = {p_bartlett:.4g}")
    print(f"  KMO (model):           {kmo_model:.4f}")
    pd.DataFrame({
        "Test": ["Bartlett chi-square", "Bartlett p-value", "KMO (model)"],
        "Value": [round(chi2, 4), round(p_bartlett, 6), round(kmo_model, 4)],
    }).to_csv(os.path.join(out_dir, f"bartlett_kmo_{version}.csv"), index=False)

    # 4. Factor analysis
    print(f"\n[4] Factor analysis ({N_FACTORS} factors, minres + varimax)...")
    loadings_df, variance_df, eigenvalues = run_factor_analysis(corr_mat, N_FACTORS, feat_names)
    # loadings_df, variance_df, eigenvalues = run_factor_analysis(df[feat_names], N_FACTORS, feat_names)
    print(loadings_df.round(3).to_string())
    print("\nVariance explained:")
    print(variance_df.round(3).to_string())
    loadings_df.to_csv(os.path.join(out_dir, f"factor_loadings_{version}.csv"))
    variance_df.to_csv(os.path.join(out_dir, f"factor_variance_{version}.csv"))
    plot_loadings_heatmap(loadings_df, out_dir, label, ver=version)
    plot_scree(eigenvalues, N_FACTORS, out_dir, label, ver=version)

    # Factor scores (MinMax scaled), binary component matrix, and deadbanded loadings
    scaled_scores, binary_components, deadbanded_loadings = compute_factor_scores(df, cols, loadings_df, deadband=deadband)
    deadbanded_loadings.to_csv(os.path.join(out_dir, f"factor_loadings_deadbanded_{version}.csv"))
    save_latex(
        scaled_scores.reset_index().to_latex(index=False, float_format="{:.2f}".format),
        os.path.join(out_dir, f"latex_factor_scores_{version}.txt"),
    )
    save_latex(
        binary_components.reset_index().to_latex(index=False),
        os.path.join(out_dir, f"latex_binary_components_{version}.txt"),
    )

    # 5. IRT
    print("\n[5] Fitting 2PL IRT model...")
    estimates = run_irt(df, cols)
    if estimates is not None:
        a = estimates["Discrimination"]
        b = estimates["Difficulty"]
        theta = estimates["Ability"]

        irt_params = pd.DataFrame(
            {"Item": feat_names, "Discrimination (a)": a, "Difficulty (b)": b}
        )
        print(irt_params.round(4).to_string(index=False))
        irt_params.to_csv(os.path.join(out_dir, f"irt_parameters_{version}.csv"), index=False)
        irt_latex = pd.DataFrame(
            [a, b],
            index=["Discrimination (a)", "Difficulty (b)"],
            columns=feat_names,
        )
        save_latex(
            irt_latex.T.to_latex(index=True, float_format="{:.2f}".format),
            os.path.join(out_dir, f"latex_irt_parameters_{version}.txt"),
        )

        hero_abilities = pd.DataFrame({
            "Name": df["Name"].values,
            "Ability (theta)": np.round(theta, 4),
            "Total Score": df[cols].sum(axis=1).values,
        }).sort_values("Ability (theta)", ascending=False).reset_index(drop=True)
        print("\nHero ranking by IRT ability:")
        print(hero_abilities.to_string(index=False))
        hero_abilities.to_csv(os.path.join(out_dir, f"hero_ability_ranking_{version}.csv"), index=False)
        save_latex(
            hero_abilities.to_latex(index=False, float_format="{:.2f}".format),
            os.path.join(out_dir, f"latex_hero_ability_ranking_{version}.txt"),
        )

        plot_irt_heatmap(a, b, feat_names, out_dir, label, ver=version)

        # 6. Q3 local dependence
        print("\n[6] Computing Yen's Q3 local dependence...")
        q3 = calculate_q3(df[cols], theta, a, b, feat_names)
        mean_abs_q3 = q3.abs().to_numpy()[~np.eye(len(feat_names), dtype=bool)].mean()
        n_pairs = len(feat_names) * (len(feat_names) - 1) / 2
        print(f"  Mean |Q3| (off-diagonal): {mean_abs_q3:.4f}  (over {int(n_pairs)} pairs)")
        q3.to_csv(os.path.join(out_dir, f"q3_local_dependence_{version}.csv"))
        plot_q3_heatmap(q3, out_dir, label, ver=version)
    else:
        print("  IRT skipped due to fit failure.")

    # 7. Hierarchical clustering dendrograms
    print("\n[7] Hierarchical clustering dendrograms...")
    hierarchical_clustering(df, cols, feat_names, out_dir, label, ver=version)

    # 8. Binomial distribution fit to total scores
    print("\n[8] Fitting binomial distribution to total scores...")
    scores = df["TOTAL"].dropna().astype(int).tolist()
    n_fitted, p_fitted = fit_and_plot_binomial(scores, VERSIONS[version]["num_feats"], out_dir, label, ver=version)

    # 9. Chi-squared goodness-of-fit tests
    print("\n[9] Chi-squared goodness-of-fit tests...")
    chi_sq_df = chi_squared_tests(scores, n_fitted, p_fitted, out_dir, label,
                                   name=f"chi_squared_tests_{version}",
                                   max_score=VERSIONS[version]["num_feats"])
    print(chi_sq_df.to_string(index=False))

    # 10. Bimodality tests on revised Raglan score distribution
    print("\n[10] Bimodality tests on revised Raglan score distribution...")
    scores_arr = np.array(scores, dtype=float)
    bim_row = bimodality_tests(scores_arr, f"Revised Raglan scores ({version})")
    _save_bimodality_results(
        [bim_row], out_dir,
        f"bimodality_tests_scores_{version}.csv",
        f"latex_bimodality_tests_scores_{version}.txt",
    )
    sil_rows = _silverman_multimode_test(scores_arr)
    _save_silverman_results(
        [sil_rows], [f"Revised Raglan scores ({version})"],
        out_dir,
        f"silverman_modes_scores_{version}.csv",
        f"latex_silverman_modes_scores_{version}.txt",
    )

    print(f"\n  Done. Results saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Original Raglan counts analysis (independent of v2/v3)
# ---------------------------------------------------------------------------

# Reference values from rank_raglan.ipynb cell 1 (kept for comparison):
# HIST_RR_COUNTS = [
#     15, 13, 19, 12, 16, 12, 17, 16, 14, 15, 13, 13, 13,
#     14, 12, 12, 14, 13, 15, 15, 14, 12, 16, 15, 15, 12,
#     12, 19, 13, 12, 12, 15,
# ]
# MYTH_RR_COUNTS = [21, 19, 18, 17, 17, 15, 14, 13, 12, 19, 15, 12, 20, 19, 14, 18, 17, 13]


def load_original_counts() -> tuple[list, list]:
    """
    Load historical and mythical Raglan hero scores from the Excel file.
    Sheet "Initial Results (3)", column AB:
      - HIST: rows 2–33  (1-indexed) → iloc[1:33]
      - MYTH: rows 36–53 (1-indexed) → iloc[35:53]
    """
    df = pd.read_excel(DATA_PATH, sheet_name="Initial Results (3)", header=None)
    col = 27  # column AB (0-indexed)
    hist = [int(v) for v in df.iloc[1:33,  col].tolist()]
    myth = [int(v) for v in df.iloc[35:53, col].tolist()]
    return hist, myth


def plot_box_violin_original(hist_counts: list, myth_counts: list, out_dir: str):
    """Side-by-side box plot and violin plot of historical vs mythical scores."""
    ymin = min(hist_counts + myth_counts) - 0.5
    ymax = max(hist_counts + myth_counts) + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].boxplot(
        [hist_counts, myth_counts],
        tick_labels=["Historical", "Mythical"],
    )
    axes[0].set_title("Raglan Score Box Plot")
    axes[0].set_ylabel("Raglan Score")
    axes[0].set_ylim(ymin, ymax)

    axes[1].violinplot([hist_counts, myth_counts], showmedians=True)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Historical", "Mythical"])
    axes[1].set_title("Raglan Score Violin Plot")
    axes[1].set_ylim(ymin, ymax)

    plt.suptitle("Distribution of Raglan Scores", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "box_violin_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def chi_squared_contingency_original(hist_counts: list, myth_counts: list,
                                      out_dir: str) -> pd.DataFrame:
    """
    Chi-squared contingency table test comparing hist vs myth score distributions.
    Constructs a crosstab of group × score value.
    """
    data = pd.DataFrame({
        "Type": (["Historical"] * len(hist_counts) +
                 ["Mythical"]   * len(myth_counts)),
        "Score": hist_counts + myth_counts,
    })
    contingency_table = pd.crosstab(data["Type"], data["Score"])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    print(f"  Chi-squared: {chi2:.4f},  p = {p_val:.4g},  df = {dof}")

    result_df = pd.DataFrame({
        "Statistic": ["Chi-squared", "p-value", "Degrees of freedom"],
        "Value": [round(chi2, 4), round(p_val, 6), dof],
    })
    result_df.to_csv(os.path.join(out_dir, "chi_squared_hist_vs_myth.csv"), index=False)
    save_latex(
        result_df.to_latex(index=False, float_format="{:.4f}".format),
        os.path.join(out_dir, "latex_chi_squared_hist_vs_myth.txt"),
    )

    contingency_table.to_csv(os.path.join(out_dir, "contingency_table_hist_vs_myth.csv"))
    save_latex(
        contingency_table.to_latex(),
        os.path.join(out_dir, "latex_contingency_table_hist_vs_myth.txt"),
    )
    return result_df


def plot_binomial_original(hist_counts: list, myth_counts: list, out_dir: str) -> dict:
    """
    Dual-axis binomial fit plots for both historical and mythical Raglan scores.
    n is fixed at 22 (the original Raglan criterion count).
    Returns dict mapping group name to fitted (n, p).
    """
    import scipy.stats as _stats

    fitted_params = {}
    for counts, group in [(hist_counts, "Historical"), (myth_counts, "Mythical")]:
        result = _stats.fit(binom, counts, bounds={"n": [22]})
        n, p = int(result.params.n), result.params.p
        fitted_params[group] = (n, p)
        print(f"  {group}: n={n}, p={p:.4f}")

        x = np.arange(10, n + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(x, binom.pmf(x, n, p), "bo", ms=7, label="Fitted binom PMF")
        ax1.vlines(x, 0, binom.pmf(x, n, p), colors="b", lw=4, alpha=0.5)
        ax1.plot(x, binom.pmf(x, n, p), "r-", lw=2, label="Fitted binom curve")
        ax1.set_ylim(ymin=0)
        ax1.set_ylabel("Probability")
        ax1.legend(loc="upper left", frameon=False)

        ax2 = ax1.twinx()
        ax2.hist(counts,
                 bins=np.arange(9.75, n + 0.75, 0.5),
                 density=False, color="grey", alpha=0.65, label="Observed counts")
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax2.set_ylabel("Count")
        ax2.legend(loc="center left", bbox_to_anchor=(0, 0.7), frameon=False)

        plt.title(f"Raglan Scores — {group} Heroes\nwith Fitted Binomial (n={n}, p={p:.3f})")
        plt.xlabel("Raglan Score")
        plt.xticks(x)
        plt.tight_layout()
        fname = f"binomial_fit_{group.lower()}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    return fitted_params


# ---------------------------------------------------------------------------
# Bimodality tests
# ---------------------------------------------------------------------------

def _count_kde_modes(data: np.ndarray, bandwidth: float, n_grid: int = 512) -> int:
    """Count local maxima of a Gaussian KDE using scipy.signal.find_peaks."""
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    std = data.std(ddof=1)
    if std == 0:
        return 1
    kde = gaussian_kde(data, bw_method=bandwidth / std)
    lo = data.min() - 3 * bandwidth
    hi = data.max() + 3 * bandwidth
    x_grid = np.linspace(lo, hi, n_grid)
    dens = kde(x_grid)
    peaks, _ = find_peaks(dens)
    return max(1, len(peaks))


def _silverman_test(data: np.ndarray, n_boot: int = 999, seed: int = 42) -> tuple[float, float]:
    """
    Silverman's (1981) bandwidth test for unimodality.
    Finds the critical bandwidth h_c at which the KDE transitions to 1 mode,
    then estimates the p-value via bootstrap: proportion of resamples that
    exhibit >1 mode under h_c.
    Returns (h_c, p_value).
    """
    std = data.std(ddof=1)
    if std == 0:
        return (0.0, 1.0)
    h_grid = np.linspace(0.01 * std, 2.0 * std, 300)
    # Find smallest h where KDE has ≤ 1 mode
    h_c = h_grid[-1]
    for h in h_grid:
        if _count_kde_modes(data, h) <= 1:
            h_c = h
            break
    rng = np.random.default_rng(seed)
    n = len(data)
    count_multi = sum(
        1 for _ in range(n_boot)
        if _count_kde_modes(
            rng.choice(data, size=n, replace=True) + rng.normal(0, h_c, size=n),
            h_c,
        ) > 1
    )
    return float(h_c), float(count_multi / n_boot)


def _silverman_multimode_test(data: np.ndarray, max_modes: int = 4,
                              n_boot: int = 999, seed: int = 42) -> list[dict]:
    """
    Silverman's (1981) bandwidth test generalised to k = 1 … max_modes.

    For each k:
      h_k  — smallest bandwidth at which the KDE has ≤ k modes (critical width).
      p    — bootstrap p-value: proportion of smoothed resamples with > k modes
              under h_k (H0: at most k modes; low p → reject).
      AIC, BIC — from a k-component Gaussian Mixture Model fitted to the data.

    Returns a list of dicts (one per k) suitable for pd.DataFrame.
    """
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

    std = data.std(ddof=1)
    n = len(data)
    rng = np.random.default_rng(seed)
    rows = []

    for k in range(1, max_modes + 1):
        row = {"k (modes)": k}

        # --- Critical bandwidth h_k ---
        if std == 0:
            h_k = 0.0
        else:
            h_grid = np.linspace(0.01 * std, 3.0 * std, 500)
            h_k = h_grid[-1]
            for h in h_grid:
                if _count_kde_modes(data, h) <= k:
                    h_k = h
                    break
        row["h_k (critical bandwidth)"] = round(float(h_k), 6)

        # --- Bootstrap p-value ---
        if std == 0 or h_k == 0:
            p_val = 1.0
        else:
            count_reject = sum(
                1 for _ in range(n_boot)
                if _count_kde_modes(
                    rng.choice(data, size=n, replace=True)
                    + rng.normal(0, h_k, size=n),
                    h_k,
                ) > k
            )
            p_val = count_reject / n_boot
        row["p-value"] = round(float(p_val), 4)

        # --- Classical GMM AIC / BIC for k components ---
        X = data.reshape(-1, 1)
        gmm = GaussianMixture(n_components=k, random_state=seed,
                              n_init=5, max_iter=500).fit(X)
        row["GMM AIC"] = round(float(gmm.aic(X)), 4)
        row["GMM BIC"] = round(float(gmm.bic(X)), 4)

        # --- Bayesian GMM for k components ---
        bgmm = BayesianGaussianMixture(n_components=k, random_state=seed,
                                        n_init=5, max_iter=500).fit(X)
        bgmm_effective = int(np.sum(bgmm.weights_ > 0.05))
        row["BGMM Lower Bound"] = round(float(bgmm.lower_bound_), 4)
        row["BGMM Effective Components"] = bgmm_effective

        rows.append(row)

    return rows


def _save_silverman_results(all_rows: list[list[dict]], labels: list[str],
                             out_dir: str, csv_name: str, latex_name: str) -> pd.DataFrame:
    """
    Combine per-distribution Silverman multi-mode rows into one DataFrame,
    prepend a 'Distribution' column, and save CSV + LaTeX.
    """
    combined = []
    for label, rows in zip(labels, all_rows):
        for row in rows:
            combined.append({"Distribution": label, **row})
    df = pd.DataFrame(combined)
    df.to_csv(os.path.join(out_dir, csv_name), index=False)
    save_latex(
        df.to_latex(index=False, float_format="{:.4f}".format),
        os.path.join(out_dir, latex_name),
    )
    print(f"  Saved: {csv_name}, {latex_name}")
    return df


def _van_der_eijk_A(data: np.ndarray, K: int = None) -> float:
    """
    Van der Eijk's (2001) A measure of agreement/concentration.
    Data are binned into K ordered categories (Sturges' rule, capped at 10).
    Decomposes the histogram into uniform layers via the layer-peeling algorithm:
      for each contiguous non-zero run of span S, A_layer = 1 - (S-1)/(K-1).
    A = 1: all cases in one bin (maximum concentration / unimodal spike).
    A = 0: perfectly uniform over all K bins.
    Lower A indicates greater spread; bimodal distributions with broad peaks
    produce lower A than sharply concentrated ones.
    """
    if K is None:
        K = min(max(5, int(np.ceil(1 + np.log2(len(data))))), 10)

    counts, _ = np.histogram(data, bins=K)
    N = float(counts.sum())
    if N == 0:
        return float("nan")

    remaining = counts.astype(float)
    A_total = 0.0

    while remaining.sum() > 1e-9:
        # Identify strictly contiguous non-zero runs
        runs, i = [], 0
        while i < K:
            if remaining[i] > 1e-9:
                j = i + 1
                while j < K and remaining[j] > 1e-9:
                    j += 1
                runs.append((i, j - 1))
                i = j
            else:
                i += 1
        if not runs:
            break
        for s, e in runs:
            S = e - s + 1
            m = remaining[s:e + 1].min()
            if m < 1e-9:
                continue
            A_layer = 1.0 - (S - 1) / (K - 1) if K > 1 else 1.0
            A_total += (S * m / N) * A_layer
            remaining[s:e + 1] -= m

    return round(float(A_total), 6)


def bimodality_tests(data, label: str, seed: int = 42) -> dict:
    """
    Run seven bimodality tests on a 1-D array.
    Tests: Hartigan's Dip Test, Bimodality Coefficient, GMM BIC comparison,
    Silverman's bandwidth test, Ashman's D, van der Eijk's A,
    KDE peak count (Scott's rule, scipy.signal.find_peaks).
    Returns a dict suitable as a DataFrame row.
    """
    from scipy.stats import skew, kurtosis as scipy_kurtosis, gaussian_kde
    from scipy.signal import find_peaks
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from diptest import diptest

    x = np.asarray(data, dtype=float)
    n = len(x)
    row = {"Distribution": label, "N": n}

    # 1. Hartigan's Dip Test (H0: unimodal)
    dip, dip_p = diptest(x)
    row["Dip Stat"] = round(float(dip), 6)
    row["Dip p-value"] = round(float(dip_p), 6)

    # 2. Bimodality Coefficient (BC; SAS formula; BC > 0.555 suggests bimodality)
    s = skew(x)
    k = scipy_kurtosis(x)          # excess kurtosis (Fisher)
    denom = (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))) if n > 3 else (k + 3)
    bc = float((s ** 2 + 1) / denom) if denom != 0 else float("nan")
    row["Bimodality Coef"] = round(bc, 6)
    row["BC > 0.555"] = "Yes" if bc > 0.555 else "No"

    # 3. GMM (classical) — compare BIC for k=1 vs k=2 components
    X = x.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=seed).fit(X)
    gmm2 = GaussianMixture(n_components=2, random_state=seed).fit(X)
    bic1, bic2 = gmm1.bic(X), gmm2.bic(X)
    delta_bic = bic1 - bic2          # positive → k=2 preferred
    row["GMM BIC k=1"] = round(float(bic1), 4)
    row["GMM BIC k=2"] = round(float(bic2), 4)
    row["GMM ΔBIC (k=1−k=2)"] = round(float(delta_bic), 4)
    row["GMM Bimodal"] = "Yes" if delta_bic > 0 else "No"

    # 3b. Bayesian GMM (BGMM) — k=2 components; effective components inferred from weights
    bgmm2 = BayesianGaussianMixture(n_components=2, random_state=seed,
                                     n_init=5, max_iter=500).fit(X)
    bgmm_weights = bgmm2.weights_
    bgmm_effective = int(np.sum(bgmm_weights > 0.05))   # components with >5% weight
    # BGMM does not expose AIC/BIC directly; use lower bound on log-likelihood
    bgmm_lb = bgmm2.lower_bound_
    row["BGMM Effective Components"] = bgmm_effective
    row["BGMM Lower Bound"] = round(float(bgmm_lb), 4)
    row["BGMM Bimodal"] = "Yes" if bgmm_effective >= 2 else "No"

    # 4. Silverman's Test (p-value ≤ 0.05 → reject unimodality)
    sil_h, sil_p = _silverman_test(x, seed=seed)
    row["Silverman h_c"] = round(sil_h, 6)
    row["Silverman p-value"] = round(sil_p, 4)

    # 5. Ashman's D — separation of the two classical GMM components (D > 2 → well-separated)
    mu1, mu2 = gmm2.means_[:, 0]
    sig1, sig2 = np.sqrt(gmm2.covariances_[:, 0, 0])
    ashman_d = float(np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(sig1 ** 2 + sig2 ** 2))
    row["Ashman's D"] = round(ashman_d, 6)
    row["D > 2"] = "Yes" if ashman_d > 2 else "No"

    # 6. Van der Eijk's A (concentration; A→1: sharply peaked, A→0: spread/uniform)
    row["Van der Eijk's A"] = _van_der_eijk_A(x)

    # 7. KDE peak count — Scott's rule bandwidth, find_peaks on 512-point grid
    if x.std(ddof=1) > 0:
        kde = gaussian_kde(x)   # Scott's rule by default
        x_grid = np.linspace(x.min(), x.max(), 512)
        dens = kde(x_grid)
        peaks, _ = find_peaks(dens)
        n_peaks = max(1, len(peaks))
    else:
        n_peaks = 1
    row["KDE N Peaks"] = n_peaks
    row["KDE Bimodal"] = "Yes" if n_peaks >= 2 else "No"

    return row


def _save_bimodality_results(rows: list[dict], out_dir: str,
                              csv_name: str, latex_name: str) -> pd.DataFrame:
    """Build DataFrame from bimodality test rows, save CSV and LaTeX."""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, csv_name), index=False)
    numeric_cols = df.select_dtypes(include="number").columns
    fmt = {c: "{:.4f}".format for c in numeric_cols}
    save_latex(
        df.to_latex(index=False, float_format="{:.4f}".format),
        os.path.join(out_dir, latex_name),
    )
    print(f"  Saved: {csv_name}, {latex_name}")
    return df


def analyse_rate_differences(out_dir: str):
    """
    Load per-criterion rates for HIST (row 57) and MYTH (row 60) from the
    Excel sheet "Initial Results (3)", columns 5–26.
    Computes MYTH - HIST differences, plots a histogram (Freedman-Diaconis
    bins), and runs a one-sample t-test and Wilcoxon signed-rank test against
    H0: mean difference = 0.
    """
    from scipy.stats import ttest_1samp, wilcoxon

    df = pd.read_excel(DATA_PATH, sheet_name="Initial Results (3)", header=None)
    hist_rates = df.iloc[56, 5:27].astype(float).tolist()  # row 57, 1-indexed
    myth_rates = df.iloc[59, 5:27].astype(float).tolist()  # row 60, 1-indexed
    diffs = [m - h for m, h in zip(myth_rates, hist_rates)]

    print(f"  HIST rates: {[round(v, 4) for v in hist_rates]}")
    print(f"  MYTH rates: {[round(v, 4) for v in myth_rates]}")
    print(f"  Differences (MYTH - HIST): {[round(v, 4) for v in diffs]}")

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diffs, bins="fd", color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero (null)")
    ax.axvline(np.mean(diffs), color="orange", linewidth=1.5,
               label=f"Mean = {np.mean(diffs):.3f}")
    ax.set_xlabel("Rate Difference (Mythical − Historical)")
    ax.set_ylabel("Count")
    ax.set_title("Differences in Criterion Rates: Mythical vs Historical")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(out_dir, "rate_differences_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # --- Statistical tests ---
    t_stat, t_pval = ttest_1samp(diffs, popmean=0)
    w_stat, w_pval = wilcoxon(diffs)

    print(f"  t-test:   t = {t_stat:.4f},  p = {t_pval:.4g}")
    print(f"  Wilcoxon: W = {w_stat:.4f},  p = {w_pval:.4g}")

    result_df = pd.DataFrame({
        "Test":      ["One-sample t-test", "Wilcoxon signed-rank"],
        "Statistic": [round(t_stat, 4),    round(w_stat, 4)],
        "p-value":   [round(t_pval, 6),    round(w_pval, 6)],
        "H0":        ["mean diff = 0",     "median diff = 0"],
    })
    result_df.to_csv(os.path.join(out_dir, "rate_difference_tests.csv"), index=False)
    save_latex(
        result_df.to_latex(index=False, float_format="{:.4f}".format),
        os.path.join(out_dir, "latex_rate_difference_tests.txt"),
    )

    # --- Bimodality tests on hist_rates, myth_rates, and their differences ---
    print("\n  Bimodality tests...")
    bim_rows = [
        bimodality_tests(hist_rates, "HIST rates"),
        bimodality_tests(myth_rates, "MYTH rates"),
        bimodality_tests(diffs,       "MYTH − HIST differences"),
    ]
    _save_bimodality_results(
        bim_rows, out_dir,
        "bimodality_tests_rates.csv",
        "latex_bimodality_tests_rates.txt",
    )

    # Silverman multi-mode tests on each rate distribution
    labels_r = ["HIST rates", "MYTH rates", "MYTH − HIST differences"]
    arrays_r  = [np.asarray(hist_rates), np.asarray(myth_rates), np.asarray(diffs)]
    sil_rows_r = [_silverman_multimode_test(a) for a in arrays_r]
    _save_silverman_results(
        sil_rows_r, labels_r, out_dir,
        "silverman_modes_rates.csv",
        "latex_silverman_modes_rates.txt",
    )

    return result_df


def run_original_raglan(base_out: str):
    out_dir = os.path.join(base_out, "original")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Running original Raglan counts analysis")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    print("\n[0] Loading counts from Excel...")
    hist_counts, myth_counts = load_original_counts()
    print(f"  HIST ({len(hist_counts)} values): {hist_counts}")
    print(f"  MYTH ({len(myth_counts)} values): {myth_counts}")

    print("\n[1] Bimodality tests on score distributions...")
    bim_rows = [
        bimodality_tests(hist_counts, "Historical counts"),
        bimodality_tests(myth_counts, "Mythical counts"),
    ]
    _save_bimodality_results(
        bim_rows, out_dir,
        "bimodality_tests_counts.csv",
        "latex_bimodality_tests_counts.txt",
    )

    labels_c = ["Historical counts", "Mythical counts"]
    arrays_c  = [np.asarray(hist_counts, dtype=float),
                 np.asarray(myth_counts, dtype=float)]
    sil_rows_c = [_silverman_multimode_test(a) for a in arrays_c]
    _save_silverman_results(
        sil_rows_c, labels_c, out_dir,
        "silverman_modes_counts.csv",
        "latex_silverman_modes_counts.txt",
    )

    print("\n[2] Box and violin plots...")
    plot_box_violin_original(hist_counts, myth_counts, out_dir)

    print("\n[3] Chi-squared contingency table (historical vs mythical)...")
    result_df = chi_squared_contingency_original(hist_counts, myth_counts, out_dir)
    print(result_df.to_string(index=False))

    print("\n[4] Binomial fits to historical and mythical scores...")
    fitted_params = plot_binomial_original(hist_counts, myth_counts, out_dir)

    print("\n[5] Chi-squared goodness-of-fit tests on Raglan scores...")
    for group, counts in [("Historical", hist_counts), ("Mythical", myth_counts)]:
        n, p_fitted = fitted_params[group]
        name = f"chi_squared_tests_{group.lower()}"
        print(f"  {group} (n={n}, p_fitted={p_fitted:.4f}):")
        chi_squared_tests(counts, n, p_fitted, out_dir, version="original", name=name)

    print("\n[6] Rate differences (Mythical − Historical) per criterion...")
    rate_diff_df = analyse_rate_differences(out_dir)
    print(rate_diff_df.to_string(index=False))

    print(f"\n  Done. Results saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="raglan_results",
                        help="Base output directory (default: raglan_results/)")
    parser.add_argument("--version", type=str, default=None,
                        choices=["v2", "v3"],
                        help="Run only this version (default: run both)")
    parser.add_argument("--deadband", type=float, default=0.0,
                        help="Deadband threshold as a fraction of each factor's max absolute "
                             "loading (0.0 = no deadbanding, 0.5 = half-max; default: 0.0)")
    args = parser.parse_args()

    run_original_raglan(args.out)

    versions = [args.version] if args.version else ["v2", "v3"]
    for v in versions:
        run_version(v, args.out, deadband=args.deadband)

    print(f"\nAll done. Results in: {args.out}/")


if __name__ == "__main__":
    main()
