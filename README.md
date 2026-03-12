# Raglan Hero Ranking Analysis

This repository contains a Jupyter notebook (`rank_raglan.ipynb`) that performs a multi-method statistical and psychometric analysis of heroes/characters scored against a set of binary features derived from Lord Raglan's heroic archetype framework. The goal is to develop a robust, data-driven ranking of heroes based on their pattern of feature presence across the dataset.

## Background

Lord Raglan's heroic biography pattern is a classical framework from folkloristics that identifies a set of recurring traits shared by mythological and historical heroes (e.g., unusual birth, trials, rise to power, fall, etc.). Each hero in the dataset is coded for whether they exhibit each trait, producing a binary presence/absence matrix. This analysis applies modern statistical and machine learning methods to explore the structure of those ratings and derive principled ability estimates.

## Data

**Input file:** `raglan_hero_counts_GH.xlsx`

The workbook contains three sheets of cleaned binary feature matrices:

| Sheet | Features |
|---|---|
| `cleaned_counts` | 22 features |
| `cleaned_revised_counts v2` | 17 features |
| `cleaned_revised_counts v3` | 14 features |

Each row is a named hero or character. Each feature column contains either `"X"` (trait present) or blank (trait absent). A `TOTAL` column records the sum of traits for each hero. The notebook separates heroes into two broad categories — **historical** and **mythological** — for comparative analysis.

## Analysis Overview

The notebook is organized into the following major sections:

### 1. Exploratory Data Analysis & Distribution Testing
- Loads and cleans the raw data from Excel
- Computes descriptive statistics and visualizes total score distributions for historical vs. mythological heroes using boxplots, violin plots, and histograms
- Tests for bimodality in the score distributions using the **Hartigan dip test** and **Kernel Density Estimation (KDE)**

### 2. Distribution Fitting
- Fits **Beta**, **Binomial**, and **Poisson** distributions to the normalized count data
- Extracts fitted parameters (α, β for Beta; n, p for Binomial; λ for Poisson) and evaluates goodness-of-fit with chi-squared tests
- Produces probability mass/density function plots comparing observed data to theoretical distributions

### 3. Factor Analysis & Dimensionality Reduction
- Tests data suitability for factor analysis using **Bartlett's test of sphericity** and the **Kaiser-Meyer-Olkin (KMO)** measure of sampling adequacy
- Applies **Principal Component Analysis (PCA)** and **Factor Analysis** (with unrotated and Varimax rotation) to identify latent structure in the feature matrix
- Produces scree plots showing cumulative explained variance across components (with a 95% threshold reference line)
- Sign-adjusts and normalizes component loadings for interpretability

### 4. Item Response Theory (IRT) Modeling
This is the core ranking method. The notebook fits a **2-Parameter Logistic (2PL) IRT model** to the binary response matrix using the `girth` library (`twopl_mml` function).

The model estimates three sets of parameters:
- **Discrimination (a):** How strongly each feature differentiates between heroes with high vs. low latent scores
- **Difficulty (b):** The threshold level at which a hero has a 50% probability of exhibiting the trait
- **Ability (θ):** The latent "heroic score" for each individual hero — the primary ranking output

Outputs include:
- Heatmap of discrimination vs. difficulty for each feature
- Histogram of ability estimates across all heroes
- Q3 residual correlation statistic for assessing model fit
- Named rankings of heroes sorted by IRT ability estimate

### 5. Clustering & UMAP Visualization
- Calculates pairwise **Hamming** and **Jaccard** distances between heroes based on their feature profiles
- Performs **hierarchical clustering** with dendrograms
- Applies **HDBSCAN** (density-based clustering) with Jaccard distance to identify natural groupings
- Uses **UMAP** for 2D non-linear dimensionality reduction, coloring projected points by cluster membership

### 6. Correlation Analysis
- Computes the **Pearson correlation matrix** between all features
- Visualizes the matrix as a heatmap to identify feature co-occurrence patterns
- Adds jitter to binary feature columns to improve scatter plot readability

## Key Functions

| Function | Purpose |
|---|---|
| `negative_log_likelihood(params, k, n)` | Negative log-likelihood for distribution fitting |
| `fit_function(k, lamb)` | Poisson PMF for curve fitting |
| `neg_log_likelihood(p, data, n_trials)` | Negative log-likelihood for binomial fitting |
| `calculate_q3(response_data, theta_estimates, item_params_df)` | Computes Q3 residual correlation statistic for IRT model fit |
| `minmax_col(col_vals, high_val, low_val)` | Min-max scaling of a column to a custom range |
| `add_jitter(column, scale)` | Adds random noise to a column for scatter plot visualization |

## Dependencies

Install the required packages before running the notebook:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
pip install factor_analyzer girth diptest hdbscan umap-learn arviz
```

> **Note:** The `girth` package is used for IRT modeling. Some packages (`hdbscan`, `umap-learn`) may require specific versions depending on your Python environment.

## Usage

1. Place `raglan_hero_counts_GH.xlsx` in the same directory as the notebook (or update the file path in the data-loading cells).
2. Launch Jupyter and open `rank_raglan.ipynb`.
3. Run all cells in order. Cells are organized sequentially; later sections depend on data and variables defined in earlier ones.

## Outputs

The notebook produces the following outputs inline:

- Distribution plots (histograms, KDE, boxplots, violin plots)
- PCA scree plots and component loading tables
- IRT parameter heatmaps and hero ability rankings
- Hierarchical clustering dendrograms
- UMAP 2D projections colored by cluster
- Feature correlation heatmaps
- Fitted distribution PMF/PDF plots with observed data overlaid

## Repository Structure

```
.
├── rank_raglan.ipynb          # Main analysis notebook
├── raglan_hero_counts_GH.xlsx # Input data (binary feature matrix)
└── README.md                  # This file
```
