import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import pybiomart
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import warnings
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from typing import Literal
from anndata import AnnData
from anndata import ImplicitModificationWarning        
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.sparse import csr_matrix, issparse, SparseEfficiencyWarning
from tqdm.auto import tqdm


def get_mouse_dict(
    cytokine_name_map: dict,
    celltype: str | None = None, 
    cytokine: str | None = None, 
    mouse_to_human_dict: dict | None = None,
    subset_to_homologs: bool = True,
) -> pd.DataFrame:
    
    mcd = pd.read_excel("/Users/soeren.becker/repos/HumanCytokineDict/data/mouse_cytokine_dict/41586_2023_6816_MOESM5_ESM.xlsx", sheet_name=celltype)
    mcd.loc[:, "cytokine"]  = mcd.Cytokine_Str.map(cytokine_name_map)
    mcd = mcd.loc[~mcd.cytokine.isna()]
    if cytokine is not None:
        mcd = mcd.loc[mcd.cytokine == cytokine]
    mcd.rename({"Gene": "gene", "Celltype_Str": "celltype", "Avg_log2FC": "log_fc"}, axis=1, inplace=True)
    mcd.loc[:, "gene_mouse"] = mcd.gene.copy()
    
    if mouse_to_human_dict is not None:
        mcd.loc[:, "gene"] = mcd.loc[:, "gene"].map(mouse_to_human_dict)
        if subset_to_homologs:
            mcd = mcd.loc[~mcd.gene.isna()]
    return mcd


def get_gene_conversion_dict():
    # Human to mouse gene name conversion
    human_dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    h2m = human_dataset.query(attributes=[
        'ensembl_gene_id', 
        'external_gene_name', 
        'mmusculus_homolog_ensembl_gene', 
        'mmusculus_homolog_associated_gene_name',
        "mmusculus_homolog_ensembl_gene"
    ])
    human_to_mouse = h2m.copy()
    human_to_mouse = human_to_mouse.dropna()
    human_to_mouse = human_to_mouse[~human_to_mouse.duplicated(subset=["Gene name"])]
    human_to_mouse = human_to_mouse.rename(columns={
        "Gene name": "genesymbol_human",
        "Mouse gene name": "genesymbol_mouse",
    })
    human_to_mouse_dict = human_to_mouse.set_index("genesymbol_human")["genesymbol_mouse"].to_dict()
    mouse_to_human_dict = {v: k for k, v in human_to_mouse_dict.items()}
    return human_to_mouse_dict, mouse_to_human_dict


def map_gene_names(adata, conversion_dict):
    adata.var.loc[:, "gene_name"] = adata.var.index.map(conversion_dict)
    adata = adata[:, ~adata.var.gene_name.isna()]
    adata.var.set_index("gene_name", inplace=True)
    return adata


def compute_weighted_mean(adata):
    weights = adata.obs["psbulk_n_cells"].to_numpy()
    weights = weights / weights.sum()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    weighted_mean = np.average(X, axis=0, weights=weights)
    return weighted_mean.reshape(1, -1)


def pool_adata(adata):
    # X = adata.X.mean(axis=0).reshape(1, -1)
    X = compute_weighted_mean(adata)
    obs = adata.obs.iloc[[0]] # arbitrarily choose obs of first row
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        obs.loc[:, "donor"] = "mean"
    obs.loc[:, "psbulk_n_cells"] = adata.obs.psbulk_n_cells.sum()
    obs = obs.loc[:, ["donor", "celltype", "cytokine", "psbulk_n_cells"]]
    adata = AnnData(X=X, obs=obs, var=adata.var)
    return adata


def compute_fold_change(
    adata_cyto, 
    adata_pbs, 
    threshold_expression=10,
    threshold_min_num_cells=20,
    pool_pbs_across_donors: bool = False,
    pool_cyto_across_donors: bool = False,
    apply_log2: bool = False,
    subset_expressed_genes_by: Literal["union", "intersection"] = "union",
    epsilon: float = 1e-6,
):    

    if pool_pbs_across_donors:
        adata_pbs = pool_adata(adata_pbs)

    if pool_cyto_across_donors:
        adata_cyto = pool_adata(adata_cyto)
    
    # apply min num cells filter
    adata_cyto = adata_cyto[adata_cyto.obs.psbulk_n_cells >= threshold_min_num_cells]
    adata_pbs = adata_pbs[adata_pbs.obs.psbulk_n_cells >= threshold_min_num_cells]
    
    if (not pool_pbs_across_donors) and (not pool_cyto_across_donors):
        # if neither is pooled, then subset to shared donors
        shared_donors = np.intersect1d(adata_cyto.obs.donor.unique(), adata_pbs.obs.donor.unique())
        adata_pbs = adata_pbs[adata_pbs.obs.donor.isin(shared_donors)]
        adata_cyto = adata_cyto[adata_cyto.obs.donor.isin(shared_donors)]
        # ensure same donor order
        adata_pbs = adata_pbs[adata_pbs.obs.donor.argsort()]
        adata_cyto = adata_cyto[adata_cyto.obs.donor.argsort()]
        assert all(adata_pbs.obs.donor.values == adata_cyto.obs.donor.values)
        
    if (adata_pbs.shape[0] == 0) or (adata_cyto.shape[0] == 0):
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        # filter out genes that are not above expression threshold in all samples
        expressed_genes_pbs = adata_pbs.var_names[np.asarray((adata_pbs.X >= threshold_expression).sum(axis=0) == adata_pbs.shape[0]).squeeze()]
        expressed_genes_cyto = adata_cyto.var_names[np.asarray((adata_cyto.X >= threshold_expression).sum(axis=0) == adata_cyto.shape[0]).squeeze()]
    if subset_expressed_genes_by == "intersection":
        expressed_genes = np.intersect1d(expressed_genes_pbs, expressed_genes_cyto)
    elif subset_expressed_genes_by == "union":
        expressed_genes = np.union1d(expressed_genes_pbs, expressed_genes_cyto)
    adata_pbs = adata_pbs[:, expressed_genes]
    adata_cyto = adata_cyto[:, expressed_genes]

    if issparse(adata_cyto.X):
        X_cyto = np.asarray(adata_cyto.X.todense())
    else:
        X_cyto = np.asarray(adata_cyto.X)
        
    if issparse(adata_pbs.X):
        X_pbs = np.asarray(adata_pbs.X.todense())
    else:
        X_pbs = np.asarray(adata_pbs.X)
    
    fcs = (X_cyto + epsilon) / (X_pbs + epsilon)
    if apply_log2:
        # with warnings.catch_warnings():
        #    warnings.filterwarnings("ignore", category=RuntimeWarning) # RuntimeWarning: divide by zero encountered in log2
        fcs = np.log2(fcs)
    
    adata_lfc = AnnData(
        X=fcs,
        var=adata_pbs.var,
        obs=adata_cyto.obs.loc[:, ["donor", "cytokine", "celltype", "psbulk_n_cells"]],
    )
    
    return adata_lfc


def compute_correlations(adata, method):
    X = adata.X
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, index=adata.obs_names)
    
    # Compute correlations between rows (samples)
    corr_df = X.T.corr(method=method)
    return corr_df


def compute_correlations_pearson_only(adata):
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    
    corr = np.corrcoef(X)
    
    # corr is now an (n_obs × n_obs) array of Pearson correlations
    corr_df = pd.DataFrame(corr, index=adata.obs_names, columns=adata.obs_names)
    return corr_df


def clean_ticks(x):
    if isinstance(x, (list, np.ndarray, pd.Index)):
        return [clean_ticks(_) for _ in x]

    if not isinstance(x, str):
        x = x.get_text()

    x = x.replace("Donor", "D")
    x = x.replace("rep", "M")
    return x


def replace_greek(x):
    x = x.replace("alpha", "α")
    x = x.replace("beta", "β")
    x = x.replace("gamma", "γ")
    x = x.replace("lambda", "λ")
    x = x.replace("epsilon", "ε")    
    x = x.replace("omega", "ω")
    return x


def inverse_replace_greek(x):
    x = x.replace("α", "alpha")
    x = x.replace("β", "beta")
    x = x.replace("γ", "gamma")
    x = x.replace("λ", "lambda")
    x = x.replace("ε", "epsilon")    
    x = x.replace("ω", "omega")
    return x


def set_label_color(ax, label_color_map, axis: Literal["x", "y"]):
    if axis == "x":
        ticklabels = ax.get_xticklabels()
    elif axis == "y":
        ticklabels = ax.get_yticklabels()
    else:
        raise ValueError()
    for label in ticklabels:
        label_text = label.get_text()
        label.set_color(label_color_map["mouse"] if (label_text.lower().startswith("rep") | label_text.lower().startswith("m")) else label_color_map["human"])


def create_heatmap(
    corrs_human_and_mouse, 
    fontsize: int = 6, 
    figsize: tuple[float, float] | None = None, 
    show: bool = True,
    path: str | None = None,
    annot: bool = True,
    title: str | None = None,
    label_color_map: dict | None = None,
):

    if figsize is None:
        figsize = (5,5)
    # Convert correlation to distance (make sure diagonal is 0)
    dist = 1 - corrs_human_and_mouse.fillna(0)
    np.fill_diagonal(dist.values, 0)
    condensed_dist = squareform(dist.values, checks=False)
    Z = linkage(condensed_dist, method='average')
    leaf_order = leaves_list(Z)
    corrs_human_and_mouse = corrs_human_and_mouse.iloc[leaf_order, leaf_order]
    
    clustered_index = corrs_human_and_mouse.index#[leaf_order]
    m_indices = [idx for idx in clustered_index if idx.startswith('rep')]
    d_indices = [idx for idx in clustered_index if idx.startswith('D')]
    custom_order = m_indices + d_indices
    
    corrs_human_and_mouse = corrs_human_and_mouse.loc[custom_order, custom_order]
    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax = sns.heatmap(corrs_human_and_mouse, cmap="RdBu_r", vmin=-1, center=0, vmax=1, annot=annot, fmt=".2g", annot_kws={"fontsize": fontsize, "family": "Arial"}, cbar=False, square=True)
    xticks = clean_ticks(corrs_human_and_mouse.columns)
    ax.set_xticks(0.5 + np.arange(corrs_human_and_mouse.shape[1]), xticks, fontsize=fontsize, family="Arial")
    
    yticks = clean_ticks(corrs_human_and_mouse.index)
    ax.set_yticks(0.5 + np.arange(corrs_human_and_mouse.shape[0]), yticks, fontsize=fontsize, family="Arial")
    ax.tick_params(length=0)
    
    if label_color_map is not None:
        set_label_color(ax, label_color_map, "x")
        set_label_color(ax, label_color_map, "y")
    
    plt.xlabel("")
    plt.ylabel("")
    if title is not None:
        plt.title(title, fontsize=fontsize, family="Arial")
    if path is not None:
        plt.savefig(path, bbox_inches="tight", pad_inches=0, transparent=True)
    if show:
        plt.show()
    
    return ax


def load_data_human(celltype_h, cytokine, geneset_restriction=None):
    adata_pbs_h = sc.read_h5ad(f"/Users/soeren.becker/repos/HumanCytokineDict/data/pseudobulks_mean_10k/pbulk_{celltype_h}_PBS_mean_10000.h5ad")
    adata_cyto_h = sc.read_h5ad(f"/Users/soeren.becker/repos/HumanCytokineDict/data/pseudobulks_mean_10k/pbulk_{celltype_h}_{cytokine}_mean_10000.h5ad")

    if geneset_restriction is not None:
        assert np.all(adata_pbs_h.var_names == adata_cyto_h.var_names)
        valid_genes = adata_pbs_h.var_names[adata_pbs_h.var_names.isin(geneset_restriction)]
        adata_pbs_h = adata_pbs_h[:, valid_genes]
        adata_cyto_h = adata_cyto_h[:, valid_genes]
    
    if "cell_type_refined" in adata_pbs_h.obs.columns:
        adata_pbs_h.obs.rename({"cell_type_refined": "celltype"}, axis=1, inplace=True)
        adata_cyto_h.obs.rename({"cell_type_refined": "celltype"}, axis=1, inplace=True)
    else:
        adata_pbs_h.obs.rename({"cell_type_refined_coarse": "celltype"}, axis=1, inplace=True)
        adata_cyto_h.obs.rename({"cell_type_refined_coarse": "celltype"}, axis=1, inplace=True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
        adata_pbs_h.obs.index = adata_pbs_h.obs.donor
        adata_cyto_h.obs.index = adata_cyto_h.obs.donor
        adata_cyto_h.obs.loc[:, "cytokine"] = adata_cyto_h.obs.loc[:, "cytokine"].apply(inverse_replace_greek)
        adata_pbs_h.obs.loc[:, "cytokine"] = adata_pbs_h.obs.loc[:, "cytokine"].apply(inverse_replace_greek)
    return adata_pbs_h, adata_cyto_h


def load_data_mouse(celltype_m, cytokine, mouse_to_human_dict, geneset_restriction=None):
    adata_m = sc.read_h5ad(f"/Users/soeren.becker/repos/HumanCytokineDict/data/mouse_cytokine_dict/mouse_cytokine_dict_all_normalized_10k_pbulk_{celltype_m}.h5ad")
    adata_m.obs.rename({"rep": "donor"}, axis=1, inplace=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        if mouse_to_human_dict is not None:
            adata_m = map_gene_names(adata_m, mouse_to_human_dict)
        if geneset_restriction is not None:
            adata_m = adata_m[:, adata_m.var_names[adata_m.var_names.isin(geneset_restriction)]]
        adata_m.obs.loc[:, "cytokine"] = adata_m.obs.loc[:, "cytokine"].apply(inverse_replace_greek)
        adata_cyto_m = adata_m[adata_m.obs.cytokine == cytokine]
        adata_pbs_m = adata_m[adata_m.obs.cytokine == "PBS"]    
        adata_pbs_m.obs.index = adata_pbs_m.obs.donor
        adata_cyto_m.obs.index = adata_cyto_m.obs.donor
    return adata_pbs_m, adata_cyto_m


def load_data(
    celltype_combination, 
    cytokine, 
    mouse_to_human_dict: None | dict = None, 
    geneset_restriction=None,
    load_human: bool = True,
    load_mouse: bool = True,
) -> dict:
    celltype_m, celltype_h = celltype_combination
    if load_mouse:    
        adata_pbs_m, adata_cyto_m = load_data_mouse(celltype_m, cytokine, mouse_to_human_dict, geneset_restriction=geneset_restriction)
    else: 
        adata_pbs_m, adata_cyto_m = None, None
    if load_human:
        adata_pbs_h, adata_cyto_h = load_data_human(celltype_h, cytokine, geneset_restriction=geneset_restriction)
    else:
        adata_pbs_h, adata_cyto_h = None, None
    return {"human_pbs": adata_pbs_h, "human_cyto": adata_cyto_h, "mouse_pbs": adata_pbs_m, "mouse_cyto": adata_cyto_m}


def lower_triangle_to_long(corr):
    corr.values[np.triu_indices_from(corr, k=0)] = np.nan
    corr = corr.reset_index().melt(
        id_vars='donor',
        var_name='donor2',
        value_name='correlation'
    ).rename(columns={'donor': 'donor1'})
    corr = corr.loc[~corr.correlation.isna()]
    return corr


def to_long(corr):
    corr = corr.reset_index().melt(
        id_vars='donor',
        var_name='donor2',
        value_name='correlation'
    ).rename(columns={'donor': 'donor1'})
    return corr


def compare_fcs(
    celltype_combination: tuple[str, str],
    cytokine: str,
    threshold_expression: int,
    threshold_min_num_cells: int,
    mouse_to_human_dict: dict,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    apply_log2: bool = False,
    mean_per_species: bool = False,
    geneset_restriction: tuple[str, list] | None = None,
    subset_expressed_genes_by: Literal["union", "intersection"] = "union",
):
    if geneset_restriction is not None:
        geneset_restriction_name, geneset_restriction = geneset_restriction
    else:
        geneset_restriction_name = None
        
    data = load_data(celltype_combination, cytokine, mouse_to_human_dict, geneset_restriction=geneset_restriction)
    for key in data:
        if data[key].shape[1] < 2: 
            return None # need at least 2 genes to compute correlations

    if mean_per_species:
        with warnings.catch_warnings():    
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
            data["human_cyto"].obs.loc[:, "donor"] = "mean"
            data["human_pbs"].obs.loc[:, "donor"] = "mean"
            data["mouse_cyto"].obs.loc[:, "donor"] = "mean"
            data["mouse_pbs"].obs.loc[:, "donor"] = "mean"
            
        fcs_human = compute_fold_change(
            data["human_cyto"], 
            data["human_pbs"], 
            threshold_expression=threshold_expression, 
            threshold_min_num_cells=threshold_min_num_cells, 
            apply_log2=apply_log2, 
            pool_cyto_across_donors=True, 
            pool_pbs_across_donors=True,
            subset_expressed_genes_by=subset_expressed_genes_by,
        )
        fcs_mouse = compute_fold_change(
            data["mouse_cyto"], 
            data["mouse_pbs"], 
            threshold_expression=threshold_expression, 
            threshold_min_num_cells=threshold_min_num_cells, 
            apply_log2=apply_log2, 
            pool_cyto_across_donors=True, 
            pool_pbs_across_donors=True,
            subset_expressed_genes_by=subset_expressed_genes_by,
        )
    else:
        fcs_human = compute_fold_change(
            data["human_cyto"], 
            data["human_pbs"], 
            threshold_expression=threshold_expression, 
            threshold_min_num_cells=threshold_min_num_cells, 
            apply_log2=apply_log2,
            subset_expressed_genes_by=subset_expressed_genes_by,
        )
        fcs_mouse = compute_fold_change(
            data["mouse_cyto"], 
            data["mouse_pbs"], 
            threshold_expression=threshold_expression, 
            threshold_min_num_cells=threshold_min_num_cells, 
            apply_log2=apply_log2, 
            pool_pbs_across_donors=True, # necessary as every mouse received only one treatment, 
            # i.e., donor specific pseudobulks are not really possible here. 
            # We hence mouse-specific cytokine treated cells by the average across all PBS treated mice
            subset_expressed_genes_by=subset_expressed_genes_by,
        )
    
    if (fcs_human is None) or (fcs_mouse is None) or (fcs_human.shape[0] == 0) or (fcs_mouse.shape[0] == 0):
        return None
    assert fcs_human.shape[0] > 0, fcs_human.shape
    assert fcs_mouse.shape[0] > 0, fcs_mouse.shape
    shared_genes = np.intersect1d(fcs_mouse.var_names, fcs_human.var_names)

    def _fc_correlation(fcs, method, comparison, geneset):
        corr = compute_correlations(fcs, method=method)
        # corr = lower_triangle_to_long(corr)
        corr = to_long(corr)
    
        donor_to_num_cells = dict(zip(fcs.obs.loc[:, "donor"].values, fcs.obs.loc[:, "psbulk_n_cells"].values))
        corr.loc[:, "num_cells_donor1"] = corr.donor1.map(donor_to_num_cells)
        corr.loc[:, "num_cells_donor2"] = corr.donor2.map(donor_to_num_cells)
        
        corr.loc[:, "comparison"] = comparison
        corr.loc[:, "geneset"] = geneset
        corr.loc[:, "num_genes"] = fcs.shape[1]
        return corr
    
    results = []
    
    if fcs_human.shape[0] > 1:
        results.append(_fc_correlation(fcs_human, method, "human_vs_human", "human"))
    if fcs_mouse.shape[0] > 1:
        results.append(_fc_correlation(fcs_mouse, method, "mouse_vs_mouse", "mouse"))        
    
    fcs_human_shared = fcs_human[:, shared_genes]
    fcs_mouse_shared = fcs_mouse[:, shared_genes]
    assert np.all(fcs_human_shared.var_names == fcs_mouse_shared.var_names)
    fcs_human_and_mouse = ad.concat([fcs_human_shared, fcs_mouse_shared], axis=0)
    results.append(_fc_correlation(fcs_human_and_mouse, method, "human_vs_mouse", "shared"))
    corrs = pd.concat(results, axis=0)
    corrs.loc[:, "threshold_expression"] = threshold_expression
    corrs.loc[:, "threshold_min_num_cells"] = threshold_min_num_cells
    corrs.loc[:, "cytokine"] = cytokine
    corrs.loc[:, "celltype_combination"] = "__".join(celltype_combination)
    corrs.loc[:, "method"] = method
    corrs.loc[:, "apply_log2"] = apply_log2
    corrs.loc[:, "geneset_restriction"] = geneset_restriction_name
    corrs.loc[:, "mean_per_species"] = mean_per_species
    corrs.loc[:, "subset_expressed_genes_by"] = subset_expressed_genes_by

    returns = {
        "corrs": corrs,
        "fcs_human": fcs_human,
        "fcs_mouse": fcs_mouse,
        "fcs_human_shared": fcs_human_shared,
        "fcs_mouse_shared": fcs_mouse_shared,
    }
    return returns


def count_unique_non_rep_donors(df, species: Literal["mouse", "human"]):
    df_long = df.melt(id_vars=['cytokine'], value_vars=['donor1', 'donor2'], value_name='num_donors')
    df_long = df_long.dropna(subset=['num_donors'])
    if species == "human":
        df_filtered = df_long[~df_long['num_donors'].str.lower().str.startswith('rep')].copy()
    elif species == "mouse":
        df_filtered = df_long[~df_long['num_donors'].str.lower().str.startswith('donor')].copy()
    else:
        raise ValueError()
    unique_donor_counts = df_filtered.groupby('cytokine')['num_donors'].nunique().sort_values(ascending=False).to_frame()
    
    return unique_donor_counts


def add_missing_values(df, idcs, value, index_name, value_name):
    missing_idcs = np.array(idcs)[~np.isin(idcs, df.index)]
    if len(missing_idcs) == 0:
        return df
    _df = pd.DataFrame()\
        .from_dict(
            zip(missing_idcs, np.ones_like(missing_idcs)*value)
        )\
        .rename({0: index_name, 1: value_name}, axis=1)\
        .set_index(index_name)
    return pd.concat([df, _df], axis=0)