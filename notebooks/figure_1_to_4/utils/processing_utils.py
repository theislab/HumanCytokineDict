import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pybiomart as pbm
from pybiomart import Server
from utils.plotting_utils import cm_to_inches

def convert_to_two_level_multiindex(df, flip_indices_first=False):
    """
    Converts a DataFrame with a single-level or two-level index to a two-level or three-level MultiIndex.
    
    If the DataFrame has a single-level index, the function assumes that the index is a string with 
    an underscore separating 'cell_type' and 'cytokine'. It splits the index 
    into these two components and sets them as a MultiIndex. The DataFrame 
    is then transposed.
    
    If the DataFrame has a two-level index, the function converts the first level 
    (assumed to be in 'cell_type_cytokine' format) into two separate levels,
    resulting in a three-level MultiIndex.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with a single-level or two-level index.
    
    Returns:
    pd.DataFrame: A DataFrame with a two-level or three-level MultiIndex.
    """
    df = df.copy()
    
    if isinstance(df.index, pd.MultiIndex):
        if flip_indices_first:
            # Flip the order of the two-level index
            df.index = df.index.swaplevel(0, 1)
        # Handle two-level index case - convert first level to two levels
        first_level_values = df.index.get_level_values(0)
        second_level_values = df.index.get_level_values(1)
        
        # Split the first level assuming 'cell_type_cytokine' format
        cell_types = first_level_values.str.split("_").str[0]
        cytokines = first_level_values.str.split("_").str[1]
        
        # Create new three-level MultiIndex
        new_index = pd.MultiIndex.from_arrays(
            [cell_types, cytokines, second_level_values],
            names=['cell_type', 'cytokine', df.index.names[1]]
        )
        df.index = new_index
        
    else:
        # Handle single-level index case (original functionality)
        df["cell_type"] = df.index.str.split("_").str[0]
        df["cytokine"] = df.index.str.split("_").str[1]
        df = df.reset_index().set_index(["cell_type", "cytokine"])
        df.drop(columns=["index"], inplace=True)
        df = df.T
        
    return df

def get_filter_by_counts(counts, low=10, very_low=5, fract=0.2, plot=True, cell_type_colors=None):
    # pseudobulking across donors could be skewed if there are very few cells in a given condition
    mask_low_count = counts < low
    very_low_count = counts < very_low
    mask_low_fraction = counts < fract * counts.groupby(level=0).transform("median")
    mask_average_calc = (mask_low_count & mask_low_fraction) | very_low_count # very low count is relevant for HSPCs
    
    if plot:
        g = mask_average_calc.groupby(level=[0,2]).sum()
        plt.figure(figsize=cm_to_inches((8, 4)))
        bottom = np.zeros(len(g.index.get_level_values(1).unique()))

        if pd.isna(cell_type_colors):
            tab10 = list(plt.get_cmap('tab10').colors)
            cell_type_colors = {}
            for cell_i, cell_type in enumerate(g.index.get_level_values(0).unique()):
                cell_type_colors[cell_type] = tab10[cell_i]

        for cell_type in g.index.get_level_values(0).unique():
            donor_vals = g.xs(cell_type, level=0)
            plt.bar(donor_vals.index, donor_vals, bottom=bottom, color=cell_type_colors[cell_type], label=cell_type)
            bottom += donor_vals
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.xticks(rotation=90)
        plt.xlabel("Donor")
        plt.ylabel("Low cell counts")
        plt.show()
    
    return mask_average_calc

def normalize_df_to_rpm(df, along_axis=1, pseudocount_initial=0.5, pseudocount_after=1):
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
    df = df + pseudocount_initial
    df = df.div(df.sum(axis=along_axis), axis=other_axis) * 1e6
    df = df + pseudocount_after
    df = df.div(df.sum(axis=along_axis), axis=other_axis) * 1e6
    
    return df

def get_raw_log2fc_values(
    count_df,
    count_df_raw,
    baseline_label="PBS",
    donors=None,
    min_counts=10,
    require_both_counts=False
    ):
    required = {"cell_type", "cytokine", "donor"}
    missing = required - set(count_df.columns.names)
    if missing:
        raise ValueError(f"count_df columns missing levels: {missing}")
    missing_raw = required - set(count_df_raw.columns.names)
    if missing_raw:
        raise ValueError(f"count_df_raw columns missing levels: {missing_raw}")

    # --- get donors ---
    if donors is not None:
        donors = pd.Index(donors)
        present_donors = count_df.columns.get_level_values("donor")
        keep = present_donors.isin(donors)
        count_df = count_df.loc[:, keep]
        count_df_raw = count_df_raw.loc[:, keep]

    df = count_df.T
    df_raw = count_df_raw.T

    # Split baseline and non-baseline
    is_baseline = df.index.get_level_values("cytokine") == baseline_label
    if not is_baseline.any():
        raise ValueError(f"No baseline cytokine '{baseline_label}' found.")
    df_treat = df.loc[~is_baseline]
    raw_treat = df_raw.loc[~is_baseline]
    baseline = df.loc[is_baseline]
    baseline_raw = df_raw.loc[is_baseline]

    # Align indices
    baseline = baseline.droplevel('cytokine')
    baseline_raw = baseline_raw.droplevel('cytokine')
    treat_match = df_treat.index.droplevel('cytokine')
    baseline = baseline.reindex(treat_match)
    baseline.index = df_treat.index
    baseline_raw = baseline_raw.reindex(treat_match)
    baseline_raw.index = df_treat.index

    # Calc log2fcs
    log2fc_by_donor = np.log2(df_treat) - np.log2(baseline)

    # Filtering
    mask_treat = raw_treat >= min_counts
    mask_baseline = baseline_raw >= min_counts

    if not require_both_counts: # OR
        mask = mask_treat | mask_baseline
    else: # AND
        mask = mask_treat & mask_baseline
    log2fc_by_donor = log2fc_by_donor.where(mask)

    # Transpose
    log2fc_by_donor = log2fc_by_donor.T
    log2fc_by_donor.index.name = "gene"

    # Mean and STD
    log2fc_values_mean = (
        log2fc_by_donor.T
        .groupby(level=["cell_type", "cytokine"], group_keys=False)
        .mean()
        .T
    )
    log2fc_values_std = (
        log2fc_by_donor.T
        .groupby(level=["cell_type", "cytokine"], group_keys=False)
        .std()
        .T
    )
    log2fc_values_mean.index.name = "gene"
    log2fc_values_std.index.name = "gene"

    return log2fc_by_donor, log2fc_values_mean, log2fc_values_std

def get_raw_log2fc_to_condition_median(
    count_df,
    count_df_raw,
    donors=None,
    min_counts=10,
    require_donor_or_median=False,
):
    required = {"cell_type", "cytokine", "donor"}

    missing = required - set(count_df.columns.names)
    if missing:
        raise ValueError(f"count_df columns missing levels: {missing}")

    missing_raw = required - set(count_df_raw.columns.names)
    if missing_raw:
        raise ValueError(f"count_df_raw columns missing levels: {missing_raw}")

    # Align normalized and raw columns first
    shared_cols = count_df.columns.intersection(count_df_raw.columns)
    count_df = count_df.loc[:, shared_cols]
    count_df_raw = count_df_raw.loc[:, shared_cols]

    # Restrict donors
    if donors is not None:
        donors = pd.Index(donors)
        keep = count_df.columns.get_level_values("donor").isin(donors)
        count_df = count_df.loc[:, keep]
        count_df_raw = count_df_raw.loc[:, keep]

    if count_df.shape[1] == 0:
        raise ValueError("No columns left after donor/shared-column filtering.")

    # rows = condition metadata, columns = genes
    df = count_df.T
    df_raw = count_df_raw.T

    # Median over donors within each condition: (cell_type, cytokine)
    median = (
        df
        .groupby(level=["cell_type", "cytokine"])
        .median()
    )

    median_raw = (
        df_raw
        .groupby(level=["cell_type", "cytokine"])
        .median()
    )

    # Match each donor row to its corresponding condition median
    condition_key = df.index.droplevel("donor")

    median_aligned = median.reindex(condition_key)
    median_aligned.index = df.index

    median_raw_aligned = median_raw.reindex(condition_key)
    median_raw_aligned.index = df.index

    # log2FC relative to condition median
    log2fc_by_donor = (
        np.log2(df.replace(0, np.nan))
        - np.log2(median_aligned.replace(0, np.nan))
    )

    # Filtering
    mask_donor = df_raw >= min_counts
    mask_median = median_raw_aligned >= min_counts

    if require_donor_or_median:  # OR
        mask = mask_donor | mask_median
    else:  # AND
        mask = mask_donor & mask_median

    log2fc_by_donor = log2fc_by_donor.where(mask)

    # Back to genes x (cell_type, cytokine, donor)
    log2fc_by_donor = log2fc_by_donor.T
    log2fc_by_donor.index.name = "gene"

    return log2fc_by_donor

def create_edger_val_df(result_df, val_type):
    """
    Creates a DataFrame from EdgeR results with a MultiIndex of (cell_type, cytokine) columns.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        The EdgeR results DataFrame containing gene, cell_type, cytokine, and value columns.
    val_type : str
        The column name in result_df to use as values (e.g., 'adj_p_value', 'log_fc', 'logCPM').
        
    Returns:
    --------
    pandas.DataFrame
        A pivoted DataFrame with genes as index and MultiIndex columns of (cell_type, cytokine),
        containing the specified values. Missing combinations are filled with NaN.
    """
    # rename 
    df = result_df[[val_type, "gene", "cell_type", "cytokine"]]
    df = df.pivot(index="gene",columns=["cell_type", "cytokine"], values=val_type)
    # Create tuples of (cell_type, cytokine) for all combinations
    column_tuples = [(cell_type, cytokine) 
                    for cell_type in df.columns.get_level_values(0).unique() 
                    for cytokine in df.columns.get_level_values(1).unique()]
    df = df.reindex(pd.MultiIndex.from_tuples(column_tuples), axis=1)
    return df


def get_protein_coding_genes(annotation_folder, exclude_ribosomal=True):
    filename = "protein_coding_genes.txt" if exclude_ribosomal else "protein_coding_genes_with_ribosomal.txt"
    path = os.path.join(annotation_folder, filename)

    if not os.path.exists(path):
        # hg38
        dataset_human = pbm.Dataset(name='hsapiens_gene_ensembl',  host='http://www.ensembl.org')
        annot = dataset_human.query(attributes=['chromosome_name', 'transcription_start_site', 'strand', 'external_gene_name', 'transcript_biotype'])

        # remove mitochondrial genes
        annot['Chromosome/scaffold name'] = annot['Chromosome/scaffold name'].astype(str)
        annot = annot[~annot['Chromosome/scaffold name'].str.contains('MT')]

        coding_biotypes = {
            "protein_coding",
            "IG_C_gene",
            "IG_D_gene",
            "IG_J_gene",
            "IG_V_gene",
            "TR_C_gene",
            "TR_D_gene",
            "TR_J_gene",
            "TR_V_gene",
        }

        transcript_filter = annot['Transcript type'].isin(coding_biotypes)
        annot_filtered = annot[transcript_filter]
        protein_coding_genes = set(annot_filtered['Gene name'].unique())
        len(protein_coding_genes)

        if exclude_ribosomal:
            # ribosomal genes were taken from https://www.genenames.org/data/genegroup/#!/group/1054
            with open("../../annotations/ribosomal_genes.json", "r") as f:
                ribosomal_gene_info = json.load(f)    
            ribosomal_gene_info = pd.DataFrame(ribosomal_gene_info)
            ribosomal_genes = set(ribosomal_gene_info["approvedSymbol"].unique())
            protein_coding_genes = protein_coding_genes.difference(ribosomal_genes)

        protein_coding_genes = list(protein_coding_genes)
        protein_coding_genes = [str(gene) for gene in protein_coding_genes]
        with open(path, "w") as f:
            for gene in sorted(protein_coding_genes):
                f.write(gene + "\n")
    else:
        with open(path, "r") as f:
            protein_coding_genes = [line.strip() for line in f.readlines()]
            
    return protein_coding_genes

def get_sex_gene_list(input_data_folder):
    if not os.path.exists(os.path.join(input_data_folder, "sex_genes.txt")):
        server = Server(host="http://www.ensembl.org")
        dataset = server["ENSEMBL_MART_ENSEMBL"]["hsapiens_gene_ensembl"]

        genes = dataset.query(
            attributes=["chromosome_name", "external_gene_name"],
            filters={"chromosome_name": ["X", "Y"]}
        )
        sex_gene_list = set(genes["Gene name"].dropna().unique().tolist())
        with open(os.path.join(input_data_folder, "sex_genes.txt"), "w") as f:
            for gene in sex_gene_list:
                f.write(gene + "\n")
    else:
        with open(os.path.join(input_data_folder, "sex_genes.txt"), "r") as f:
            sex_gene_list = set([line.strip() for line in f.readlines()])
    return sex_gene_list
