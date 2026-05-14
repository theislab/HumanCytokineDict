from typing import Tuple

import os
import h5py
import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import decoupler as dc
import dask.distributed as dd

from hydra.core.hydra_config import HydraConfig
from logging import getLogger
from omegaconf import OmegaConf

LOGGER = getLogger()

def load_and_subset_data(cfg, logger) -> ad.AnnData:

    logger.info(f"Load data with dask: {cfg.dataset.input_path}")
    with h5py.File(cfg.dataset.input_path, "r") as f:
        adata = ad.AnnData(
            obs=ad.experimental.read_elem(f["obs"]),
            var=ad.experimental.read_elem(f["var"]),
        )
        adata.X = ad.experimental.read_elem_as_dask(
            f["X"], chunks=[cfg.dataset.chunk_size, adata.shape[1]]
        )

    logger.info(f"Load annotations: {cfg.dataset.annotation_path}")
    all_obs = pd.read_csv(cfg.dataset.annotation_path, index_col=0)
    adata.obs.loc[:, "cell_type_refined"] = adata.obs.index.map(all_obs.loc[:, "cell_type_new"])
    adata.obs.loc[:, "cell_type_refined_coarse"] = adata.obs.index.map(all_obs.loc[:, "cell_type_new_coarse"])
    logger.info(f"adata: {adata}")

    if not cfg.dataset.celltype is None:
        celltype_subset = [cfg.dataset.celltype]
        logger.info(f"Subset data to: {celltype_subset}.")
        adata = adata[adata.obs.cell_type_refined.isin(celltype_subset)]
        selected_cell_type = cfg.dataset.celltype
    else:
        assert cfg.dataset.celltype_coarse is not None
        celltype_subset = [cfg.dataset.celltype_coarse]
        logger.info(f"Subset data to coarse cell type: {celltype_subset}.")
        adata = adata[adata.obs.cell_type_refined_coarse.isin(celltype_subset)]
        selected_cell_type = cfg.dataset.celltype_coarse
    logger.info(f"adata: {adata}")

    if cfg.dataset.normalize_target_sum is None:
        cpm_str = ""
    else:
        LOGGER.info(f"Run normalize_total() with target_sum = {cfg.dataset.normalize_target_sum}")
        sc.pp.normalize_total(adata, target_sum=cfg.dataset.normalize_target_sum)
        cpm_str = f"_{cfg.dataset.normalize_target_sum}"

    logger.info("Transfer data to memory.")
    adata = adata.to_memory(copy=False)
    adata.layers["counts"] = adata.X
    logger.info("load_and_subset(): done.")
    return adata, selected_cell_type, cpm_str


def _get_expression_stats(adata) -> Tuple[int, float]:
    LOGGER.info("Running _get_expression_stats()")
    num_expressing_cells = np.asarray((adata.X > 0).sum(axis=0)).squeeze()
    frac_expressing_cells = num_expressing_cells / adata.shape[0]
    return num_expressing_cells, frac_expressing_cells


def add_expression_stats(adata) -> None:
    LOGGER.info("Running add_expression_stats()")
    num_expressing_cells, frac_expressing_cells = _get_expression_stats(adata)
    adata.var.loc[:, "num_expressing_cells_total"] = num_expressing_cells
    adata.var.loc[:, "frac_expressing_cells_total"] = frac_expressing_cells
    for donor in adata.obs.donor.unique():
        adata_donor = adata[adata.obs.donor == donor]
        if adata_donor.shape[0] == 0:
            LOGGER.info(f"No samples found for {donor}, skipping.")
            continue
        num_expressing_cells, frac_expressing_cells = _get_expression_stats(adata_donor)
        adata.var.loc[:, f"num_expressing_cells_{donor}"] = num_expressing_cells
        adata.var.loc[:, f"frac_expressing_cells_{donor}"] = frac_expressing_cells


@hydra.main(config_path="conf", config_name="conf_pseudobulk_subsampled", version_base="1.1")
def main(cfg):
    LOGGER.info(OmegaConf.to_yaml(cfg))
    output_dir = HydraConfig.get()["runtime"]["output_dir"]
    LOGGER.info(f"output_dir: {output_dir}")
    adata, cell_type, cpm_str = load_and_subset_data(cfg, LOGGER)
    all_cytokines = adata.obs.cytokine.unique()
    LOGGER.info(f"There are {len(all_cytokines)} unique cytokines.")
    for cytokine in all_cytokines:

        if not cytokine == "PBS": 
            continue

        LOGGER.info(f"Subset data to {cytokine}.")
        adata_sub = adata[adata.obs.cytokine == cytokine].copy()
        LOGGER.info(f"adata: {adata_sub}")
        # compute how many cells express a gene
        add_expression_stats(adata_sub) # in-place op
        pdata = dc.get_pseudobulk(
            adata_sub,
            sample_col='donor',
            groups_col=['bc1_well'],
            layer='counts',
            mode=cfg.dataset.get_pseudobulk.mode,
            min_cells=0,
            min_counts=0,
            remove_empty=False,
        ) # beware: genes are automatically sorted by name
        assert np.all(adata_sub.var.sort_index() == pdata.var)

        # remove empty rows (which are created by dc.get_pseudobulk())
        pdata = pdata[~pdata.obs.donor.isna()]

        # for some reason some of the columns in obs are converted to object
        # which needs to be undone in order to be able to save as h5ad file later
        for column in pdata.obs.columns:
            if pdata.obs.loc[:, column].dtype == object:
                pdata.obs[column] = pdata.obs[column].astype(
                    adata_sub.obs.loc[:, column].dtype
                )
        pdata.uns["adata_subsampled_obs"] = adata_sub.obs
        LOGGER.info(f"pdata: {pdata}")
        # save results
        filepath = os.path.join(
            output_dir, 
            f"pbulk_{cell_type}_{cytokine}_{cfg.dataset.get_pseudobulk.mode}{cpm_str}.h5ad".replace(" ", "_").replace("/", "_")
        )
        LOGGER.info(f"Saving {filepath}.")
        pdata.write_h5ad(filepath)
    
    LOGGER.info(f"Done.")
    return 0.0

if __name__ == "__main__":
    LOGGER.info("Running compute_pseudobulks_subsampled().")
    main()