import os
import numpy as np
import pandas as pd
import glob
import pertpy as pt

def dense_X(adata):
    X = adata.X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def filter_subset(
    subset,
    contrast_col,
    baseline,
    min_cells=10,
    min_samples_detected=2,
    min_count=5,
):
    subset = subset[subset.obs["cell_count"] >= min_cells].copy()

    counts = subset.obs[contrast_col].astype(str).value_counts()
    valid_levels = counts[counts >= min_samples_detected].index

    subset = subset[subset.obs[contrast_col].astype(str).isin(valid_levels)].copy()

    if subset.n_obs == 0:
        return None

    if baseline not in set(subset.obs[contrast_col].astype(str)):
        return None

    if subset.obs[contrast_col].astype(str).nunique() < 2:
        return None

    X = dense_X(subset)
    vals = subset.obs[contrast_col].astype(str).values

    base_mask = vals == str(baseline)
    other_mask = ~base_mask

    n_base_detected = (X[base_mask] >= min_count).sum(axis=0)
    n_other_detected = (X[other_mask] >= min_count).sum(axis=0)

    gene_filter = (
        ((n_base_detected >= 2) | (n_other_detected >= 2))
        & (subset.var["n_cells_by_counts"] >= min_cells)
    )

    subset = subset[:, gene_filter].copy()

    if subset.n_vars == 0:
        return None

    return subset

def run_edger_general(
    subset,
    contrast_col,
    baseline,
    min_count=5,
    design=None,
):
    if design is None:
        design = f"~{contrast_col}"

    subset = subset.copy()
    subset.obs[contrast_col] = subset.obs[contrast_col].astype("category")

    edgr = pt.tl.EdgeR(subset, design=design)
    edgr.fit()

    X = dense_X(subset)

    libsize = X.sum(axis=1)
    logcpm = np.log2((X + 1) / (libsize[:, None] + X.shape[1]) * 1e6)

    vals = subset.obs[contrast_col].astype(str).values

    base_mask = vals == str(baseline)
    base_logcpm = logcpm[base_mask]

    base_mean = base_logcpm.mean(axis=0)
    base_var = base_logcpm.var(axis=0, ddof=1)
    base_n = base_logcpm.shape[0]

    n_base_detected = (X[base_mask] >= min_count).sum(axis=0)

    results = {}

    for level in subset.obs[contrast_col].cat.categories:
        level = str(level)

        if level == str(baseline):
            continue

        res_df = edgr.test_contrasts(
            edgr.contrast(
                column=contrast_col,
                baseline=baseline,
                group_to_compare=level,
            )
        )

        level_mask = vals == level
        level_logcpm = logcpm[level_mask]

        level_mean = level_logcpm.mean(axis=0)
        level_var = level_logcpm.var(axis=0, ddof=1)
        level_n = level_logcpm.shape[0]

        manual_log_fc = level_mean - base_mean
        manual_log_fc_se = np.sqrt(level_var / level_n + base_var / base_n)

        n_level_detected = (X[level_mask] >= min_count).sum(axis=0)

        gene_stats = pd.DataFrame(
            {
                "manual_log_fc": manual_log_fc,
                "manual_log_se": manual_log_fc_se,
                f"n_{baseline}_detected": n_base_detected,
                f"n_{level}_detected": n_level_detected,
            },
            index=subset.var_names,
        )

        if "variable" in res_df.columns:
            res_df = res_df.set_index("variable")

        res_df = res_df.join(gene_stats)

        res_df["comparison_supported"] = (
            (res_df[f"n_{baseline}_detected"] >= 2)
            | (res_df[f"n_{level}_detected"] >= 1)
        )

        results[level] = res_df

    if not results:
        return pd.DataFrame()

    out = (
        pd.concat(results, names=[contrast_col])
        .reset_index(level=contrast_col)
        .reset_index(names="gene")
    )

    return out

def default_case_selector(obs, stratum):
    mask = np.ones(len(obs), dtype=bool)
    for col, val in stratum.items():
        mask &= obs[col].astype(str).values == str(val)
    return mask

def default_reference_selector(obs, stratum, contrast_col, baseline):
    mask = default_case_selector(obs, stratum)
    mask &= obs[contrast_col].astype(str).values == str(baseline)
    return mask

def run_de_by_strata(
    pseudobulk,
    strata_cols,
    contrast_col,
    baseline,
    out_dir,
    filename_prefix="DE",
    min_cells=10,
    min_samples_detected=2,
    min_count=5,
    case_selector=None,
    reference_selector=None,
):
    os.makedirs(out_dir, exist_ok=True)

    if case_selector is None:
        case_selector = default_case_selector

    if reference_selector is None:
        reference_selector = default_reference_selector

    strata = (
        pseudobulk.obs[strata_cols]
        .drop_duplicates()
        .sort_values(strata_cols)
    )
    if contrast_col == "cytokine" and "concentration" in strata.columns:
        strata = strata[strata["concentration"].astype(str) != "Ctrl"]

    for _, row in strata.iterrows():
        stratum = row.to_dict()

        label = "__".join(f"{k}-{v}" for k, v in stratum.items())
        print(f"Processing {label} | contrast={contrast_col}...")

        filepath = os.path.join(
            out_dir,
            f"{filename_prefix}__{label}__by-{contrast_col}.parquet",
        )

        merged_files = glob.glob(os.path.join(out_dir, "all_DE_results*"))
        if merged_files:
            print(f"Skipping {label}; merged results file already exists")
            continue

        if os.path.exists(filepath):
            print(f"Skipping {label}; results already exist")
            continue

        obs = pseudobulk.obs

        case_mask = case_selector(obs, stratum)
        ref_mask = reference_selector(
            obs,
            stratum,
            contrast_col=contrast_col,
            baseline=baseline,
        )

        subset = pseudobulk[case_mask | ref_mask].copy()

        subset = filter_subset(
            subset,
            contrast_col=contrast_col,
            baseline=baseline,
            min_cells=min_cells,
            min_samples_detected=min_samples_detected,
            min_count=min_count,
        )

        if subset is None:
            print(f"Skipping {label}; insufficient samples or genes")
            continue

        results = run_edger_general(
            subset,
            contrast_col=contrast_col,
            baseline=baseline,
            min_count=min_count,
        )

        if results.empty:
            print(f"Skipping {label}; no contrasts returned")
            continue

        for col, val in stratum.items():
            results[col] = val

        results["reference_label"] = str(baseline)

        if "contrast" in results.columns:
            results = results.drop(columns="contrast")

        results.to_parquet(filepath, index=False)