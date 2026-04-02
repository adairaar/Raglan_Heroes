# Raglan Hero Pattern Analysis

This repository contains Python scripts for a psychometric analysis of the "Raglan hero pattern" — a set of binary criteria (C1–C17) used to classify mythological heroes. The analysis applies factor analysis and Item Response Theory (IRT) to the binary item response data, with bootstrap sensitivity analyses to assess the stability of the results.

## Data

All scripts read from a single Excel workbook:

```
/Users/aaronadair/Downloads/raglan_hero_counts_GH.xlsx
```

Sheets used:

| Sheet | Description |
|---|---|
| `cleaned_revised_counts v2` | Full 17-criterion dataset |
| `cleaned_revised_counts v3` | Reduced 13-criterion dataset (C5, C7, C10 dropped; C9 constant at runtime) |
| `Initial Results (3)` | Per-criterion pass rates for historical and mythical heroes (used by `raglan_analysis.py`) |

Each row is a hero; each criterion column contains `"X"` (present) or blank (absent), binarized to 1/0 at load time.

---

## Scripts

### `raglan_analysis.py` — Main analysis pipeline

Runs the full psychometric analysis on both data versions (v2 and v3), plus a separate analysis of the original Raglan counts (historical vs mythical). All results are saved under `raglan_results/`.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--out` | `raglan_results` | Base output directory |
| `--version` | *(both)* | Run only one version: `v2` or `v3` |
| `--deadband` | `0.0` | Deadband threshold as a fraction of each factor's max absolute loading |

**Pipeline steps (v2 / v3):**
1. Load & binarize data
2. Compute tetrachoric correlation matrix
3. Bartlett's sphericity test & KMO sampling adequacy
4. Factor analysis (minres + varimax, 6 factors) on tetrachoric correlations
5. 2PL IRT model (MML via `girth`) for item discrimination/difficulty and hero ability estimates
6. Yen's Q3 local dependence check between items
7. Hierarchical clustering dendrograms (heroes and criteria, Hamming/average linkage)
8. Total score distribution fitted with a binomial (MLE)
9. Chi-squared goodness-of-fit tests (uniform, fitted binomial, binomial p=0.5)
10. Bimodality tests on the revised Raglan score distribution

**Pipeline steps (original Raglan counts):**
1. Bimodality tests on historical and mythical score distributions
2. Box and violin plots
3. Chi-squared contingency table (historical vs mythical)
4. Binomial fits to historical and mythical scores
5. Chi-squared goodness-of-fit tests
6. Rate differences (Mythical − Historical) per criterion, including bimodality tests on each rate distribution

**Bimodality tests** (applied to all distributions):
- Hartigan's Dip Test (H0: unimodal)
- Bimodality Coefficient (BC > 0.555 suggests bimodality)
- Classical Gaussian Mixture Model (GMM): BIC comparison of k=1 vs k=2 components
- Bayesian Gaussian Mixture Model (BGMM): effective component count from variational inference
- Silverman's bandwidth test (bootstrap p-value)
- Ashman's D (GMM component separation; D > 2 → well-separated)
- Van der Eijk's A (concentration measure; A → 1: peaked, A → 0: spread/uniform)
- KDE peak count (Scott's rule bandwidth, `scipy.signal.find_peaks`)

**Silverman multi-mode tests** (saved separately for all distributions):
- Tests for k = 1, 2, 3, 4 modes
- Returns critical bandwidth h_k, bootstrap p-value, classical GMM AIC/BIC, and Bayesian GMM lower bound and effective components for each k

**Usage:**
```bash
python raglan_analysis.py [--out raglan_results/] [--version v3] [--deadband 0.0]
```

---

### `bootstrap_factor_analysis_binary.py` — Bootstrap sensitivity for factor analysis

Assesses the stability of the factor structure by running bootstrap resampling on the raw binary item matrix. An optional `--tetrachoric` flag computes tetrachoric correlations first and uses the correlation matrix as input to factor analysis instead.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--n_boot` | `500` | Number of bootstrap iterations |
| `--n_factors` | `6` | Number of factors to extract |
| `--seed` | `42` | Random seed |
| `--out` | `bootstrap_results_binary` | Output directory |
| `--tetrachoric` | off | Use tetrachoric correlations before factor analysis (requires `scipy`) |

**Procedure:**
1. Reproduce the original varimax-rotated factor analysis (minres, 6 factors)
2. Run N bootstrap iterations: resample heroes with replacement, re-extract factors
3. Apply orthogonal Procrustes rotation to each bootstrap loading matrix to align with the original solution
4. Compute consistency metrics: Tucker's congruence coefficient (Phi) per factor, 95% CIs on each loading, and factor replication rate (Phi ≥ 0.85)

**Outputs** (saved to `--out`, filenames include `_tetrachoric` or `_no_tetrachoric` suffix):
- `original_loadings_{suffix}.csv`
- `factor_replication_summary_{suffix}.csv`
- `loading_confidence_intervals_{suffix}.csv`
- `bootstrap_phi_distributions_{suffix}.png`
- `bootstrap_loading_ci_{suffix}.png`
- `bootstrap_loading_sd_heatmap_{suffix}.png`
- `loadings_comparison_heatmap_{suffix}.png`

**Usage:**
```bash
python bootstrap_factor_analysis_binary.py [--n_boot 500] [--n_factors 6] [--seed 42] [--out bootstrap_results_binary/] [--tetrachoric]
```

---

### `bootstrap_irt.py` — Bootstrap sensitivity for IRT

Assesses the stability of 2PL IRT item parameters (discrimination *a*, difficulty *b*) and hero ability estimates (θ) via bootstrap resampling.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--n_boot` | `500` | Number of bootstrap iterations |
| `--seed` | `42` | Random seed |
| `--out` | `irt_bootstrap_results` | Output directory |
| `--tetrachoric` | off | Estimate IRT parameters via 1-factor normal-ogive on tetrachoric correlations instead of `girth` MML |

**Procedure:**
1. Fit the original 2PL IRT model (MML via `girth`) to get *a*, *b*, and θ parameters
2. Run N bootstrap iterations: resample heroes with replacement, refit 2PL
3. Align each bootstrap solution to the original (handle sign flips on *a*)
4. For each bootstrap iteration, estimate hero ability (θ) on the original heroes using the bootstrap item parameters via `girth.ability_eap` (EAP with N(0,1) prior, matching the method used internally by `twopl_mml`)
5. Compute 95% CIs, bootstrap SD, and Pearson *r* stability metrics for *a*, *b*, and θ

**Tetrachoric mode** (`--tetrachoric`): estimates 2PL-equivalent parameters from a 1-factor normal-ogive decomposition of the tetrachoric correlation matrix (Lord & Novick, 1968):
- `a_i = λ_i / √(1 − λ_i²)`
- `b_i = −τ_i / λ_i`

Ability in tetrachoric mode is estimated via `girth.ability_eap`.

**Outputs** (saved to `--out`, filenames include `_tetrachoric` or `_no_tetrachoric` suffix):
- `irt_original_parameters_{suffix}.csv` — original *a*, *b*, bootstrap SD, and 95% CIs per item
- `irt_bootstrap_ci_{suffix}.csv` — per-item 95% bootstrap CIs for *a* and *b*
- `irt_stability_summary_{suffix}.csv` — Pearson *r* stability for *a*, *b*, and θ
- `irt_ability_{suffix}.csv` — per-hero θ estimates with bootstrap mean, SD, and 95% CIs (sorted by original θ descending)
- `irt_bootstrap_parameter_ci_{suffix}.png` — CI plot for *a* and *b*
- `irt_bootstrap_stability_{suffix}.png` — distributions of per-bootstrap Pearson *r* vs original for *a* and *b*
- `irt_parameter_heatmap_{suffix}.png` — bootstrap SD and original estimate heatmaps for *a* and *b*
- `irt_bootstrap_ability_ci_{suffix}.png` — CI plot for hero ability θ (sorted by original θ)
- `irt_bootstrap_ability_stability_{suffix}.png` — distribution of per-bootstrap Pearson *r* vs original for θ

**Usage:**
```bash
python bootstrap_irt.py [--n_boot 500] [--seed 42] [--out irt_bootstrap_results/] [--tetrachoric]
```

---

## Dependencies

Install required packages with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn factor_analyzer girth tqdm openpyxl pyrelimri diptest
```

| Package | Used for |
|---|---|
| `numpy`, `pandas` | Array operations and data handling |
| `matplotlib`, `seaborn` | Plotting |
| `scipy` | Statistical tests, Procrustes rotation, KDE, `find_peaks`, tetrachoric correlation |
| `scikit-learn` | PCA, preprocessing, `GaussianMixture`, `BayesianGaussianMixture` |
| `factor_analyzer` | Minres + varimax factor analysis, Bartlett's test, KMO |
| `girth` | 2PL IRT model via MML (`twopl_mml`), ability estimation (`ability_eap`) |
| `tqdm` | Progress bars |
| `openpyxl` | Reading `.xlsx` files |
| `pyrelimri` | Tetrachoric correlation (used in `raglan_analysis.py`) |
| `diptest` | Hartigan's Dip Test for unimodality |

---

## Output Directory Structure

```
raglan_results/
├── original/    # Original Raglan counts analysis (historical vs mythical)
├── v2/          # Results for full 17-criterion dataset
└── v3/          # Results for reduced 13-criterion dataset

bootstrap_results_binary/
│   # Bootstrap factor analysis outputs
│   # Files named with _no_tetrachoric or _tetrachoric suffix

irt_bootstrap_results/
│   # Bootstrap IRT outputs
│   # Files named with _no_tetrachoric or _tetrachoric suffix
```
