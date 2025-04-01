import pandas as pd

def convert_to_two_level_multiindex(df):
    """
    Converts a DataFrame with a single-level index to a two-level MultiIndex.
    
    The function assumes that the index of the DataFrame is a string with 
    an underscore separating 'cell_type' and 'cytokine'. It splits the index 
    into these two components and sets them as a MultiIndex. The DataFrame 
    is then transposed.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with a single-level index.
    
    Returns:
    pd.DataFrame: A DataFrame with a two-level MultiIndex ('cell_type', 'cytokine').
    """
    df = df.copy()
    df["cell_type"] = df.index.str.split("_").str[0]
    df["cytokine"] = df.index.str.split("_").str[1]
    df = df.reset_index().set_index(["cell_type", "cytokine"])
    df.drop(columns=["index"], inplace=True)
    df = df.T
    return df

def normalize_df_to_rpm(df, along_axis=1):
    """
    Normalizes a DataFrame such that the sum along the specified axis is 10^6.
    
    This function first divides each element by the sum of its row or column 
    (depending on the specified axis), multiplies by 10^6, adds 1 to each 
    element, and then normalizes again to ensure the sum is 10^6.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to be normalized.
    along_axis (int): The axis along which to normalize (0 for columns, 1 for rows).
    
    Returns:
    pd.DataFrame: A normalized DataFrame where the sum along the specified axis is 10^6.
    
    Raises:
    Exception: If an invalid axis is provided.
    """
    if along_axis == 1:
        other_axis = 0
    elif along_axis == 0:
        other_axis = 1
    else:
        raise Exception("Invalid axis!")
    
    df = df.copy()
    df = df.div(df.sum(axis=along_axis), axis=other_axis) * 1e6
    df = df + 1
    df = df.div(df.sum(axis=along_axis), axis=other_axis) * 1e6
    
    return df

def generate_mean_df(df, donor_names):
    """Generates the mean and standard deviation across different donors"""
    df = pd.concat(df, keys=donor_names, axis=0)
    
    # drop cell type and cytokine columns if they exist
    if "cell_type" in df.columns:
        df = df.drop(columns=["cell_type"])
    if "cytokine" in df.columns:
        df = df.drop(columns=["cytokine"])
    
    mean_df = df.groupby(level=1).mean()
    std_df = df.groupby(level=1).std()
    
    return mean_df, std_df

def generate_cytokine_mean_df(aggregate_dfs):
    """Generates the mean across different cytokine stimulations for each cell type"""
    dfs = []
    
    for df in aggregate_dfs:    
        df["cell_type"] = df.index.str.split("_").str[0]
        df["cytokine"] = df.index.str.split("_").str[1]
        df.drop(columns=["cytokine"], inplace=True)
        
        df = df.groupby(["cell_type"]).mean()
        dfs.append(df)
        
    mean_cytokine_df = pd.concat(dfs, keys=donor_names, axis=0)
    return mean_cytokine_df


