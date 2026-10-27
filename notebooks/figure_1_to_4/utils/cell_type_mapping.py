def get_relevant_cell_types(alt_flag=False):
    if not alt_flag:
        relevant_cell_types = [
            "B cell",
            "CD4 T cell",
            "CD8 T cell",
            "Treg",
            "MAIT",
            "NKT",
            "NK CD56hi",
            "NK CD56low",
            "CD14 Mono",
            "CD16 Mono", 
            "cDC",
            "pDC",
            "granulocyte",
            "ILC",
            "HSPC",
            "plasmablast",
        ]
    else:
        relevant_cell_types = [
            "B Naive",
            "B Intermediate",
            "CD4 Naive",
            "CD4 Memory",
            "CD8 Naive",
            "CD8 Memory",
            "Treg",
            "MAIT",
            "CD14 Mono",
            "CD16 Mono",
            "NK CD56hi",
            "NK CD56low",
            "NKT",
            "cDC",
            "pDC",
            "granulocyte",
            "ILC",
            "HSPC",
            "plasmablast",
        ]
        
    relevant_cell_type_conversion = {
            "B_cell": "B cell",
            "Naive_B_cell": "B Naive",
            "Intermediate_B_cell": "B Intermediate",
            "CD4_T_cell": "CD4 T cell",
            "CD4_Naive_T_cell": "CD4 Naive",
            "CD4_Memory_T_cell": "CD4 Memory",
            "CD8_T_cell": "CD8 T cell",
            "CD8_Naive_T_cell": "CD8 Naive",
            "CD8_Memory_T_cell": "CD8 Memory",
            "Treg": "Treg",
            "MAIT": "MAIT",
            "NKT": "NKT",
            "cDC": "cDC",
            "pDC": "pDC",
            "NK": "Natural Killer",
            "NK_CD56hi": "NK CD56hi",
            "NK_CD56low": "NK CD56low",
            "Mono": "Monocyte",
            "CD14_Mono": "CD14 Mono",
            "CD16_Mono": "CD16 Mono",
            "HSPC": "HSPC",
            "Granulocyte": "granulocyte",
            "ILC": "ILC",
            "Plasmablast": "plasmablast",
        }

    assert len(set(relevant_cell_type_conversion.values())) == len(relevant_cell_type_conversion.values())
    relevant_cell_to_pseudobulk = {v: k for k, v in relevant_cell_type_conversion.items()}
    # relevant_cells_pseudobulk = [relevant_cell_to_pseudobulk[c] for c in relevant_cell_types]

    assert all([c in relevant_cell_type_conversion.values() for c in relevant_cell_types])

    # add some additional conversions
    relevant_cell_type_conversion["CD56-dim_NK"] = "NK CD56low"
    relevant_cell_type_conversion["CD56-dim NK"] = "NK CD56low"
    relevant_cell_type_conversion["CD56-bright_NK"] = "NK CD56hi"
    relevant_cell_type_conversion["CD56-bright NK"] = "NK CD56hi"
    relevant_cell_type_conversion["other"] = "granulocyte"
    
    return relevant_cell_types, relevant_cell_type_conversion, relevant_cell_to_pseudobulk