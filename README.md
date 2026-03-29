# Raglan Hero Pattern Analysis

This repository contains Python scripts for a psychometric analysis of the "Raglan hero pattern" — a set of binary criteria (C1–C17) used to classify mythological heroes. The analysis applies factor analysis and Item Response Theory (IRT) to the binary item response data, with bootstrap sensitivity analyses to assess the stability of the results.

## Data

All scripts read from a single Excel workbook:

```
/Users/aaronadair/Downloads/raglan_hero_counts_GH.xlsx
```

Two sheets are used:

| Sheet | Description |
|---|---|
| `cleaned_revised_counts v2` | Full 17-criterion dataset |
| `cleaned_revised_counts v3` | Reduced 13-criterion dataset (C5, C7, C10 dropped; C9 constant at runtime) |

Each row is a hero; each criterion column contains `"X"` (present) or blank (absent), binarized to 1/0 at load time.

---

## Scripts

### `raglan_analysis.py` — Main analysis pipeline

Runs the full psychometric analysis on both data versions (v2 and v3) and saves results to per-version output directories under `raglan_results/`.

**Pipeline steps:**
1. Load & binarize data
2. Compute tetrachoric correlation matrix
3. Bartlett's sphericity test & KMO sampling adequacy
4. Factor analysis (minres + varimax, 6 factors) on tetrachoric correlations
5. 2PL IRT model (MML) for item discrimination/difficulty and hero ability estimates
6. Yen's Q3 local dependence check between items
7. Hierarchical clustering dendrograms (heroes and criteria, Hamming/average linkage)
8. Total score distribution fitted with a binomial (MLE)
9. Chi-squared goodness-of-fit tests (uniform, fitted binomial, binomial p=0.5)
10. Save all results (CSV, plots, LaTeX)

**Usage:**
```bash
python raglan_analysis.py [--out raglan_results/]
```

---

### `bootstrap_factor_analysis_binary.py` — Bootstrap sensitivity for factor analysis

Assesses the stability of the factor structure by running bootstrap resampling directly on the raw binary item matrix (no tetrachoric correlation step).

**Procedure:**
1. Reproduce the original varimax-rotated factor analysis (minres, 6 factors) on binary data
2. Run N bootstrap iterations: resample heroes with replacement, re-extract factors
3. Apply orthogonal Procrustes rotation to each bootstrap loading matrix to align with the original solution
4. Compute consistency metrics: Tucker's congruence coefficient (Phi) per factor, mean absolute deviation, 95% CIs on each loading, and factor replication rate (Phi ≥ 0.85)

**Outputs** (saved to `bootstrap_results_binary/`):
- `original_loadings.csv` — factor loadings from the original solution
- `factor_replication_summary.csv` — per-factor Tucker's Phi and replication rates
- `loading_confidence_intervals.csv` — 95% bootstrap CIs on every loading
- `bootstrap_phi_distributions.png` — histograms of Tucker's Phi per factor
- `bootstrap_loading_ci.png` — CI plots for each factor's loadings
- `bootstrap_loading_sd_heatmap.png` — heatmap of loading instability (bootstrap SD)
- `loadings_comparison_heatmap.png` — side-by-side comparison of PCA and varimax-rotated FA loadings

**Usage:**
```bash
python bootstrap_factor_analysis_binary.py [--n_boot 500] [--n_factors 6] [--seed 42] [--out bootstrap_results_binary/]
```

---

### `bootstrap_irt.py` — Bootstrap sensitivity for IRT

Assesses the stability of the 2PL IRT item parameters (discrimination *a* and difficulty *b*) via bootstrap resampling.

**Procedure:**
1. Fit the original 2PL IRT model (MML via `girth`) to get target *a* and *b* parameters
2. Run N bootstrap iterations: resample heroes with replacement, refit 2PL
3. Align each bootstrap solution to the original (handle sign flips on *a*)
4. Compute 95% CIs, bootstrap SD, and Pearson *r* stability metrics

An optional `--tetrachoric` mode estimates 2PL-equivalent parameters from a 1-factor normal-ogive decomposition of the tetrachoric correlation matrix instead of fitting MML directly (Lord & Novick, 1968):
- `a_i = λ_i / √(1 − λ_i²)`
- `b_i = −τ_i / λ_i`

**Outputs** (saved to `irt_bootstrap_results/`):
- `irt_original_parameters.csv` — original *a* and *b* estimates
- `irt_bootstrap_ci.csv` — 95% bootstrap CIs on all parameters
- `irt_stability_summary.csv` — Pearson *r* stability summary per parameter type
- `irt_bootstrap_parameter_ci.png` — CI plot for *a* and *b* parameters
- `irt_bootstrap_stability.png` — distributions of per-bootstrap Pearson *r* vs original
- `irt_parameter_heatmap.png` — heatmap of bootstrap SD and original estimates

**Usage:**
```bash
python bootstrap_irt.py [--n_boot 500] [--seed 42] [--out irt_bootstrap_results/] [--tetrachoric]
```

---

## Dependencies

Install required packages with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn factor_analyzer girth tqdm openpyxl pyrelimri
```

| Package | Used for |
|---|---|
| `numpy`, `pandas` | Array operations and data handling |
| `matplotlib`, `seaborn` | Plotting |
| `scipy` | Statistical tests, Procrustes rotation, clustering |
| `scikit-learn` | PCA, preprocessing |
| `factor_analyzer` | Minres + varimax factor analysis, Bartlett's test, KMO |
| `girth` | 2PL IRT model via MML |
| `tqdm` | Progress bars |
| `openpyxl` | Reading `.xlsx` files |
| `pyrelimri` | Tetrachoric correlation (optional, for `--tetrachoric` mode) |

---

## Output Directory Structure

```
raglan_results/
├── v2/          # Results for full 17-criterion dataset
└── v3/          # Results for reduced 13-criterion dataset

bootstrap_results_binary/
│   # Bootstrap factor analysis outputs (see above)

irt_bootstrap_results/
│   # Bootstrap IRT outputs (see above)
```
