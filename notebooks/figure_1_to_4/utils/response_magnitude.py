import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import mstats

def _euclidean_distance(arr1, arr2, dropna=False, min_genes=None, print_min_gene_warning=False, length_norm=False, cell_type=None, cytokine=None):
    if dropna:
        arr1 = arr1.dropna()
        arr2 = arr2.dropna()

    common_index = arr1.index.intersection(arr2.index)
    arr1 = arr1.loc[common_index]
    arr2 = arr2.loc[common_index]
    
    # compute distance
    if min_genes is not None:
        if len(common_index) < min_genes:
            if print_min_gene_warning:
                print(f"Not enough genes for {cell_type} {cytokine}")
            return np.nan
    
    distance = euclidean(arr1.values.flatten(), arr2.values.flatten())
    if length_norm:
        distance = distance / np.sqrt(len(arr1))
        
    return distance

def compute_response_euclidean_distance(
    df, cell_type, cytokine, control="PBS",
    dropna=False, min_genes=None, print_min_gene_warning=False,
    length_norm=False):
    """
    Compute the Euclidean distance between the treatment and control cytokine response for a given cell type.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing gene expression data with a MultiIndex of genes,(cell_type, cytokine).
    - cell_type (str): The specific cell type for which the distance is to be computed.
    - cytokine (str): The treatment cytokine whose response is being compared.
    - control (str, optional): The control cytokine to compare against, default is "PBS".

    Returns:
    - float: The Euclidean distance between the treatment and control responses for the specified cell type and cytokine.
    """
    df_treatment = df.loc[:, (cell_type, cytokine)]
    df_control = df.loc[:, (cell_type, control)]
    
    distance = _euclidean_distance(
        df_treatment,
        df_control,
        dropna=dropna,
        min_genes=min_genes,
        print_min_gene_warning=print_min_gene_warning,
        length_norm=length_norm,
        cell_type=cell_type,
        cytokine=cytokine
        )
        
    return distance

def normalize_distances(distances: pd.Series):
    """
    Normalize the distances to a scale of 0 to 1.

    Parameters:
    - distances (pd.Series): A pandas Series containing the distances to be normalized.

    Returns:
    - pd.Series: A pandas Series of the same index as the input, containing the normalized distances.
                 The distances are scaled to a range of 0 to 1 based on the minimum and maximum values in the input Series.
    """
    min_val, max_val = distances.min(), distances.max()
    normalized_distances = (distances - min_val) / (max_val - min_val)
    
    return normalized_distances

def winsorize_and_normalize_distances(distances: pd.Series, winsorize=True, normalize=True):
    """
    Winsorize and normalize the distances to a scale of 0 to 1.

    Parameters:
    - distances (pd.Series): A pandas Series containing the distances to be winsorized and normalized.

    Returns:
    - pd.Series: A pandas Series of the same index as the input, containing the normalized distances.
                 The distances are first winsorized to limit extreme values, then scaled to a range of 0 to 100.
    """
    if winsorize:
        # winsorize only non-nan values; mstats.winsorize sorts NaN as the largest value and would overwrite it
        finite = distances.dropna()
        winsorized_distances = pd.Series(mstats.winsorize(finite.values, limits=[0, 0.05]), index=finite.index)
        winsorized_distances = winsorized_distances.reindex(distances.index)
    else:
        winsorized_distances = distances
        
    if normalize:
        min_val, max_val = winsorized_distances.min(), winsorized_distances.max()
        normalized_distances = (winsorized_distances - min_val) / (max_val - min_val)
    else:
        normalized_distances = winsorized_distances
    
    return normalized_distances