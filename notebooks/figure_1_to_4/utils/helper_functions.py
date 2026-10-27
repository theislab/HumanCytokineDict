import numpy as np
import leidenalg
import igraph as ig
import pandas as pd

def get_common_index(series1, series2):
    series1 = series1.copy().dropna()
    series2 = series2.copy().dropna()
    common_index = series1.index.intersection(series2.index)
    return series1.loc[common_index], series2.loc[common_index]

def calculate_cell_fractions(cell_counts, target_columns=None, mean_umis_per_cell=None):
    """
    Cell-type fractions per (treatment, donor), optionally weighted by relative transcriptome size.
    Parameters
    ----------
    cell_counts : pd.Series
        Cell numbers with a 3-level MultiIndex (cell_type, treatment, donor), e.g. cell_count_agg.
    target_columns : pd.MultiIndex, optional
        Columns of the expression table (cell_type, cytokine, donor) the fractions are aligned to
        (e.g. count_df_norm_agg.columns). Missing combinations get fraction 0. If None, the fractions
        are returned in long format without alignment.
    mean_umis_per_cell : pd.Series, optional
        Relative UMIs per cell by cell type. If given, fractions are multiplied by it and renormalized
        to sum to one per (treatment, donor).

    Returns
    -------
    pd.Series
        Fractions (or transcriptome-weighted fractions) indexed like target_columns.
    """
    counts = cell_counts.copy()
    counts.index = counts.index.set_names(["cell_type", "treatment", "donor"])

    fractions = counts.unstack(["treatment", "donor"])
    fractions = fractions.div(fractions.sum(axis=0), axis=1)
    fractions = fractions.stack(["treatment", "donor"], future_stack=True).rename("relative_count")

    if target_columns is not None:
        fractions = fractions.reindex(target_columns).fillna(0)
    if mean_umis_per_cell is None:
        return fractions
    if fractions.index.get_level_values("cell_type").map(mean_umis_per_cell).isna().any():
        raise ValueError("mean_umis_per_cell is missing cell types present in the fractions")
    
    umis = pd.Series(fractions.index.get_level_values("cell_type").map(mean_umis_per_cell), index=fractions.index)
    weighted = fractions * umis
    condition_levels = [name for name in weighted.index.names if name != "cell_type"]
    weighted = weighted.div(weighted.groupby(level=condition_levels).transform("sum"))
    return weighted

def leiden_clustering(distance_matrix, resolution_parameter=0.5, print_communities=False, invert_distance=False, seed=0):
    """Leiden clustering expects an adjacency matrix. If a distance matrix is provided, it should be inverted."""
    # Convert the DataFrame to a list-of-lists
    if invert_distance:
        adjacency = (1-distance_matrix.values).tolist()
    else:
        adjacency = (distance_matrix.values).tolist()

    # Create an undirected graph. Use loops=False to avoid self-loops.
    g = ig.Graph.Weighted_Adjacency(adjacency, mode=ig.ADJ_UNDIRECTED, attr="weight", loops=False)

    # Set vertex names for easier interpretation.
    g.vs["name"] = list(distance_matrix.index)

    # Apply the Leiden algorithm.
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=resolution_parameter,
        seed=seed
    )

    all_communities = []
    for i, community in enumerate(partition):
        community_nodes = [g.vs[node]["name"] for node in community]
        if print_communities:
            print(f"Community {i+1}: {community_nodes}")
        all_communities.append(community_nodes)

    communities_flat = [item for sublist in all_communities for item in sublist]
    
    return all_communities, communities_flat

def get_upper_triangle(df, diagonal=False):
    """
    Extract the upper triangle of a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame (typically a correlation or distance matrix)
    diagonal : bool, default=False
        Whether to include the diagonal elements
        
    Returns:
    --------
    pandas.Series
        A series containing the upper triangle values with MultiIndex
    """
    # Get the upper triangle mask
    if diagonal:
        mask = np.triu(np.ones_like(df, dtype=bool))
    else:
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
    
    # Apply mask and stack to get a series
    upper_triangle = df.where(mask)
    
    return upper_triangle

def delete_diagonal(df):
    """
    Remove the diagonal elements from a DataFrame by setting them to NaN.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame (typically a correlation or distance matrix)
        
    Returns:
    --------
    pandas.DataFrame
        A copy of the DataFrame with diagonal elements set to NaN
    """
    df_copy = df.copy()
    arr = df_copy.to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    df_copy.iloc[:, :] = arr
    return df_copy