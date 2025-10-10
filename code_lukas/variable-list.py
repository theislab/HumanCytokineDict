"""
INFO ON CYTOKINES
cytokine_info: Info on all cytokines. (Dataframe)
cytokine_info_parse: Info on parse cytokines. (Dataframe)
CYTOKINE_ORDER: Order of cytokines. (List)
CYTOKINE_ORDER_WITH_PBS: Order of cytokines including PBS. (List)
ALL_CELL_TYPES: Order of cell types. (List)
ALL_CELL_TYPES_EXCLUDE: Order of cell types minus ILC and pDC. (List)
ALL_CELL_TYPES_EXCLUDE_ALSO_HSPC: Order of cell types minus ILC, pDC, and HSPC. (List)

INFO ON CYTOKINE FAMILIES
cytokine_name_to_level_1_family = cytokine_info.reset_index().set_index("name")["family_level_1"].to_dict()
cytokine_name_to_level_2_family = cytokine_info.reset_index().set_index("name")["family_level_2"].to_dict()
cytokine_level_1_order = list(dict.fromkeys(cytokine_name_to_level_1_family.values()))
cytokine_level_2_order = list(dict.fromkeys(cytokine_name_to_level_2_family.values()))
level2_to_level1 = cytokine_info.set_index("family_level_2")["family_level_1"].to_dict()

PSEUDOBULK BY INDIVIDUAL DONOR
sum_gene_count_df_raw_by_donor: Sum of gene counts for each cytokine and cell type per donor. Format: index = genes, columns = (cell_type, cytokine, donor)
sum_gene_count_df_norm_by_donor: Normalized sum of gene counts for each cytokine and cell type per donor. Format: index = genes, columns = (cell_type, cytokine, donor)
all_counts_by_condition = Cell counts for each cytokine and cell type per donor. Format: index = (cell_type, cytokine, donor)
sum_gene_count_df_norm_by_donor_pbs: Normalized sum of gene counts for PBS only per donor. Format: index = genes, columns = cell_type
sum_gene_count_df_norm_pbs_median: Median of normalized sum of gene counts for PBS only across donors. Format: index = genes, columns = cell_type

donor_pbs_ratio_to_median_df: aka 'baseline FC'. Ratio of each donor's PBS expression to the median PBS expression across donors for each cell type. Format: index = genes, columns = (cell_type, donor)
all_ratios_to_PBS: aka 'raw FC'. Ratios of cytokine expression to PBS expression for each cytokine, cell type, and donor. Format: index = genes, columns = (donor, cell_type, cytokine)

PSEUDOBULK AVERAGED ACROSS DONORS
average_norm_sum_gene_count_df: Average normalized sum of gene counts for each cytokine and cell type across donors. Format: index = genes, columns = (cell_type, cytokine)
average_sum_gene_count_df: Average sum of gene counts for each cytokine and cell type across donors. Format: index = genes, columns = (cell_type, cytokine)
control_norm_gene_count_df: Average normalized sum of gene counts for PBS only. Format: index = genes, columns = cell_type

GENE SETS AND FILTERS
unfiltered_genes: All human gene names that were aligned.
inclusive_filters_by_cell_type: Permissive gene filters for each cell type (average non-normalized sum of counts in at least one condition > 20)
questionable_genes: Gens flagged by an edge effect analysis as potentially problematic.
protein_coding_genes: Genes that code for non-mitochondrial, non-ribosomal proteins.
expressed_genes_by_cell_type_dict: More stringent filter with
1) a mean expression > 4
2) a max expression > 20
3) a total count > 20
for a gene in a specific cell type calculated over all stimulation conditions.

EdgeR
edger_results: EdgeR results for each cytokine and cell type.
edger_df_padj: P-values for each cytokine and cell type. Format: index = genes, columns = (cell_type, cytokine)
edger_df_logCPM: Log2-CPM for each cytokine and cell type. Format: index = genes, columns = (cell_type, cytokine)
edger_df_log2FoldChange: Log2-Fold changes for each cytokine and cell type. Format: index = genes, columns = (cell_type, cytokine)
significant_results: Number of significant results for each cytokine and cell type. Format: index =  cell_type, columns = cytokine

RESPONSE MAGNITUDES (distances is a synonym here)
all_normalized_distances_mean: Mean normalized distances for each cytokine and cell type. index = cytokine, columns = cell_type
all_relative_distances_mean: Mean relative distances (divided by max distance for each cytokine) for each cytokine and cell type. index = cytokine, columns = cell_type
all_normalized_distances_percentile: Percentile of normalized distances for each cytokine and cell type (used for plotting). index = cytokine, columns = cell_type
strong_impact_by_cell_type: strong impact by cell type. (calulated using the distances). index = cytokine, columns = cell_type

all_raw_euclidean_distances_by_donor = Raw (non-normalized and winsorized) response magnitudes for each cytokine, cell type, and donor. Format: index = cytokine, columns = (cell_type, donor)
all_normalized_euclidean_distances_by_donor = Normalized response magnitudes for each cytokine, cell type, and donor. Format: index = cytokine, columns = (cell_type, donor)
all_abs_sums_by_donor = Response magnitudes based on absolute sum method for each cytokine, cell type, and donor. Format: index = cytokine, columns = (cell_type, donor)
"""