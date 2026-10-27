import scipy.stats as stats
import numpy as np
import pandas as pd

# ------------------------------ FITTING UTILS ----------------------------

def determine_censored_range(
    df,
    x_col="log10_sim_supernatant",
    y_col="log10_raw_conc",
    buffer_val=0.1,
    numerical_tol=1e-5,
    min_interval_width=np.log10(2),
    minimum_points=5,
    boundary_fraction_threshold=0.8,
):
    """
    Identify x-intervals dominated by protein censoring.

    After identifying directly censored intervals:
    - every interval at or below the highest LLOD-censored interval is
      classified as lower-censored;
    - every interval at or above the lowest ULOD-censored interval is
      classified as upper-censored.
    """
    df = df.copy()

    x = df[x_col]
    y = df[y_col]

    total_range = x.max() - x.min()

    n_intervals = max(1, int(np.floor(total_range / min_interval_width)))
    boundaries = np.linspace(x.min(), x.max(), n_intervals + 1)

    df["x_interval"] = pd.cut(x, bins=boundaries, include_lowest=True)
    df["at_llod"] = (y <= df["log10_llod"] + buffer_val + numerical_tol)
    df["at_ulod"] = (y >= df["log10_ulod"] - buffer_val - numerical_tol)
    df["at_y_boundary"] = df["at_llod"] | df["at_ulod"]

    boundary_by_interval = (
        df.groupby("x_interval", observed=True)
        .agg(
            n=("at_y_boundary", "size"),
            n_at_llod=("at_llod", "sum"),
            n_at_ulod=("at_ulod", "sum"),
            n_at_boundary=("at_y_boundary", "sum"),
            fraction_at_llod=("at_llod", "mean"),
            fraction_at_ulod=("at_ulod", "mean"),
            fraction_at_boundary=("at_y_boundary", "mean"),
        )
        .reset_index()
        .sort_values("x_interval")
        .reset_index(drop=True)
    )

    enough_points = boundary_by_interval["n"].ge(minimum_points)

    # Directly supported censoring within each interval.
    boundary_by_interval["directly_llod_censored"] = enough_points & boundary_by_interval["fraction_at_llod"].ge(boundary_fraction_threshold)
    boundary_by_interval["directly_ulod_censored"] = enough_points & boundary_by_interval["fraction_at_ulod"].ge(boundary_fraction_threshold)
    
    # Propagate LLOD censoring downwards:
    # True for every interval up to the highest directly LLOD-censored one.
    directly_llod = boundary_by_interval["directly_llod_censored"].to_numpy()
    if directly_llod.any():
        largest_llod_position = np.flatnonzero(directly_llod).max()
        boundary_by_interval["llod_censored"] = np.arange(len(boundary_by_interval)) <= largest_llod_position
    else:
        boundary_by_interval["llod_censored"] = False

    # Propagate ULOD censoring upwards:
    # True for every interval from the lowest directly ULOD-censored one.
    directly_ulod = boundary_by_interval["directly_ulod_censored"].to_numpy()

    if directly_ulod.any():
        smallest_ulod_position = np.flatnonzero(directly_ulod).min()
        boundary_by_interval["ulod_censored"] = np.arange(len(boundary_by_interval)) >= smallest_ulod_position
    else:
        boundary_by_interval["ulod_censored"] = False

    boundary_by_interval["mostly_censored"] = boundary_by_interval["llod_censored"] | boundary_by_interval["ulod_censored"]

    columns_to_merge = [
        "x_interval",
        "directly_llod_censored",
        "directly_ulod_censored",
        "llod_censored",
        "ulod_censored",
        "mostly_censored",
        "fraction_at_boundary",
        "fraction_at_llod",
        "fraction_at_ulod",
    ]

    df = df.merge(boundary_by_interval[columns_to_merge], on="x_interval", how="left")

    return df, boundary_by_interval

def fit_with_floating_start(x, y, max_float=1):
    x = np.asarray(x)
    y = np.asarray(y)

    vals_by_float = []

    for start in np.arange(0, max_float + 0.01, 0.2):
        mask = x > start
        n = mask.sum()

        if n <= 2:
            continue

        (m, b), residuals, _, _, _ = np.polyfit(
            x[mask],
            y[mask],
            deg=1,
            full=True,
        )

        rss = residuals[0] if residuals.size else 0.0
        rmse = np.sqrt(rss / (n - 2))
        if start == 0:
            rmse_0 = rmse

        vals_by_float.append({
            "start": start,
            "n": n,
            "slope": m,
            "intercept": b,
            "rss": rss,
            "rmse": rmse,
            "fraction_remaining": n / len(mask),
            "improvement": rmse / rmse_0
        })

    vals_by_float = pd.DataFrame(vals_by_float)

    if len(vals_by_float[vals_by_float['improvement'] < 0.7]) > 0:
        row_index = vals_by_float[vals_by_float['improvement'] < 0.7].index[0]
    else:
        row_index = 0
    row = vals_by_float.loc[row_index]
    slope, intercept, start = row['slope'], row['intercept'], row['start']

    return slope, intercept, start

def run_fit(df, floating_start=False):
    protein = df['protein'].iloc[0]
    mask_uncensored = ~df['mostly_censored']
    df_uncensored = df[mask_uncensored].copy()

    if len(df_uncensored) > 0:
        x = df_uncensored['log10_sim_supernatant']
        y = df_uncensored['log10_raw_conc']
        identity = df_uncensored['row_identity']

        x_min = x.min()
        x_max = x.max()

        if len(df_uncensored) > 10 and (x_max - x_min) >= np.log10(5):
            if floating_start:
                slope, intercept, start = fit_with_floating_start(x.values, y.values)
            else:
                slope, intercept = np.polyfit(x.values, y.values, 1)
                start = x.min()
            mask_for_fit = x >= start
            x = x[mask_for_fit]
            y = y[mask_for_fit]
            identity = identity[mask_for_fit]
            effective_fitted_points = len(x)
            effective_range = x.max() - x.min()
            corr = stats.pearsonr(x, y).statistic
        else:
            effective_fitted_points = 0
            effective_range = 0
            corr = np.nan
            slope = np.nan
            intercept = np.nan
            start = np.nan
            mask_for_fit = np.nan
            x = np.nan
            y = np.nan
            identity = np.nan
    else:
        effective_fitted_points = 0
        effective_range = 0
        corr = np.nan
        intercept = np.nan
        slope = np.nan
        start = np.nan
        x_min = np.nan
        x_max = np.nan
        mask_for_fit = np.nan
        x = np.nan
        y = np.nan
        identity = np.nan

    result = {
        "protein": protein,
        "corr": corr,
        "slope": slope,
        "intercept": intercept,
        "start": start,
        "fitted_points": effective_fitted_points,
        "effective_range": effective_range,
        "x_min": x_min,
        "x_max": x_max,
    }

    return result, x, y, identity, mask_uncensored, mask_for_fit

def log10_or_nan(values: pd.Series) -> pd.Series:
    return np.log10(pd.to_numeric(values, errors="coerce")).replace([np.inf, -np.inf], np.nan)
def aggregate_min(subunit_expression: pd.DataFrame) -> pd.Series:
    return subunit_expression.min(axis=0)
def aggregate_mean(subunit_expression: pd.DataFrame) -> pd.Series:
    return subunit_expression.mean(axis=0)
def aggregate_sum(subunit_expression: pd.DataFrame) -> pd.Series:
    return subunit_expression.sum(axis=0)

AGGREGATORS = {
    "min": aggregate_min,
    "mean": aggregate_mean,
    "sum": aggregate_sum,
}

def calculate_cell_state(
    expression: pd.DataFrame,
    marker_genes: list[str],
) -> pd.Series:
    """Return one zero-centred state score per (cell type, cytokine, donor).

    Each marker is log-transformed and z-scored *within cell type* across all treatment/donor pseudobulks.
    """
    genes = [gene for gene in marker_genes if gene in expression.index]
    if len(genes) < 3 and len(genes) < len(marker_genes):
        raise ValueError(f"Need at least three state markers in expression; found {genes}")

    values = np.log1p(expression.loc[genes])
    result = pd.Series(index=values.columns, dtype=float, name="inflammatory_state")
    for cell_type in values.columns.get_level_values("cell_type").unique():
        columns = values.columns[values.columns.get_level_values("cell_type") == cell_type]
        within_type = values.loc[:, columns]
        z = within_type.sub(within_type.mean(axis=1), axis=0).div(
            within_type.std(axis=1).replace(0, np.nan), axis=0
        )
        result.loc[columns] = z.mean(axis=0)
    return result.fillna(0)

def state_gain_for_protein(
    production_index: pd.MultiIndex,
    cell_states: pd.Series,
    state_sensitivity: pd.Series,
    max_abs_log_gain: float = 3.0,
) -> pd.Series:
    """
    Calculate exp(beta[protein, cell] * state[cell, sample]).
    """
    beta = pd.Series(
        production_index.get_level_values("cell_type"),
        index=production_index,
    ).map(state_sensitivity).fillna(0.0)

    state = cell_states.reindex(production_index)
    if state.isna().any():
        raise ValueError("cell_state does not cover every cell-type/treatment/donor production entry")

    gain = np.exp((beta * state).clip(-max_abs_log_gain, max_abs_log_gain))
    return gain

def calculate_supernatant(
        row, count_df_norm_needed, cell_fraction_vector,
        cell_states=None, state_sensitivity=0, subunit_aggregation='min',
        censored_treatments_all=None, timepoint0_weight=0
        ):
    
    # calculate effective expression values per cell type
    aggregator = AGGREGATORS[subunit_aggregation]
    genes = [gene for gene in row["usable_genes"].split(";") if gene]

    subunit_values_by_celltype = count_df_norm_needed.loc[genes]
    per_celltype_production = aggregator(subunit_values_by_celltype)

    if timepoint0_weight > 0:
        timepoint0_expression = per_celltype_production.xs("TimePoint-0", level="cytokine")
        timepoint0_lookup = pd.MultiIndex.from_arrays(
            [per_celltype_production.index.get_level_values("cell_type"), per_celltype_production.index.get_level_values("donor")],
            names=["cell_type", "donor"],
        )
        timepoint0_offset = timepoint0_expression.reindex(timepoint0_lookup).fillna(0)
        timepoint0_offset.index = per_celltype_production.index
        per_celltype_production = per_celltype_production + timepoint0_weight * timepoint0_offset
    # OLD
    # subunit_values_by_celltype = count_df_norm_needed.loc[genes]
    # per_celltype_production = aggregator(subunit_values_by_celltype)

    # apply state sensitivity
    if cell_states is not None:
        state_gain = state_gain_for_protein(
            per_celltype_production.index, cell_states, state_sensitivity
        )
    else:
        state_gain = 1
    per_celltype_production = per_celltype_production * state_gain

    # multiply with cell fractions
    weighted_per_celltype_production = per_celltype_production * cell_fraction_vector

    # APPLY CENSORING (exclude conditions where the protein itself was added to the supernatant)
    if censored_treatments_all is not None:
        censored_treatments = censored_treatments_all.get(row['protein'], [])
        weighted_per_celltype_production = weighted_per_celltype_production[
            ~weighted_per_celltype_production.index.get_level_values('cytokine').isin(censored_treatments)
        ]
    
    # calculate simulated supernatant
    simulated = weighted_per_celltype_production.groupby(
        level=["cytokine", "donor"],
        observed=True,
    ).sum()
    tmp = simulated.rename("sim_supernatant").reset_index().rename(
        columns={"cytokine": "treatment"}
        )
    tmp["protein"] = row["protein"]
    tmp["target_genes"] = row["target_genes"]
    tmp["aggregation"] = subunit_aggregation
    tmp["n_subunits"] = len(genes)
    
    # calculate per cell type contributions
    contrib = weighted_per_celltype_production.rename("cell_type_contribution")
    contrib = contrib.reset_index()
    contrib = contrib.rename(columns={"cytokine": "treatment"})
    contrib["protein"] = row["protein"]
    contrib["target_genes"] = row["target_genes"]
    contrib["aggregation"] = subunit_aggregation
    contrib["n_subunits"] = len(genes)

    return tmp, contrib