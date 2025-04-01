import matplotlib.pyplot as plt

tab10 = list(plt.get_cmap('tab10').colors)
set2 = list(plt.get_cmap('Set2').colors)
tab10 = tab10 + set2

# from lib.info_dicts import cell_type_colors
CELL_TYPE_COLORS_orig = {
    'B cell': tab10[0],
    'CD4 T cell': tab10[1], 
    'CD8 T cell': tab10[2],
    'Treg': tab10[9],
    'MAIT': tab10[4],
    'NKT': tab10[5],
    'NK CD56hi': tab10[6],
    'NK CD56low': tab10[7],
    #'ILC': tab10[10],
    'pDC': tab10[9],
    'cDC': tab10[8],
    'CD14 Mono': tab10[3],
    'CD16 Mono': tab10[16],
    #'Granulocyte': tab10[12],
    'HSPC': tab10[14],
    #'Platelet': tab10[15],
}


CELL_TYPE_COLORS_a = {
    'CD4 T cell': '#4f5d91ff', 
    'CD4 Naive':'#5f73beff',
    'CD4 Memory': '#1f3792ff',
    'CD8 T cell': '#6b8159ff',
    'CD8 Naive':'#8ea47bff',
    'CD8 Memory':'#376313ff',
    'Treg': '#98b1d0ff',
    'MAIT': '#45a784ff',
    'NKT': '#8f6fb5ff',
    'CD56-bright NK': '#f8967bff',
    'CD56-dim NK': '#d76552ff',
    'ILC': '#f0a9b1ff',
    'B cell': '#b09292ff',
    'B Naive': '#72554eff' ,
    'B Intermediate/Memory': '#bb6c6cff',
    'Plasmablast':'#874949ff',
    'pDC': '#f9c671ff',
    'cDC': '#cea668ff',
    'CD14 Mono': '#f79d51ff',
    'CD16 Mono': '#e3874cff',
    'HPSC': '#6e5046ff',
    'Platelet':'#767476ff',
    'other': '#afababff',
}


CELL_TYPE_COLORS = {
    'B cell':  tab10[0],
    'CD4 T cell': tab10[1], 
    'CD8 T cell': tab10[2],
    'Treg': 'blueviolet',
    'MAIT': 'dodgerblue',
    'NKT': 'red',
    'NK CD56hi': 'pink',
    'NK CD56low': 'saddlebrown',
    'ILC': 'navy',
    'pDC': 'deeppink',
    'cDC': 'lime',
    'CD14 Mono': 'purple',
    'CD16 Mono': tab10[16],
    'HSPC': 'dimgray',
    #'Platelet':'white',
    
    
    #'CD4 Naive': tab10[2],
    #'CD4 Memory': tab10[1],
    #'CD8 Naive':'turquoise',
    #'CD8 Memory': 'yellow', 
    #'ILC': 'navy',
    #'B Naive': 'burlywood',
    #'B Intermediate/Memory': 'blue',
    #'Plasmablast':'darkolivegreen',
    
    #'other': 'gold',
}

ALL_CELL_TYPES = list(CELL_TYPE_COLORS.keys())

if __name__ == "__main__":
    # Create vertical bars for each cell type
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot vertical bars
    x_positions = range(len(CELL_TYPE_COLORS_a))
    for i, (cell_type, color) in enumerate(CELL_TYPE_COLORS_a.items()):
        ax.bar(i, 1, color=color, label=cell_type)

    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(CELL_TYPE_COLORS_a.keys(), rotation=90)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.title('similar cells, similar colors')
    plt.tight_layout()
    plt.show()


    # Create vertical bars for each cell type
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CELL_TYPE_COLORS_b.keys(), rotation=90)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.title('similar cells, disimilar colors')
    plt.tight_layout()
    plt.show()