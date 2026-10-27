import matplotlib.pyplot as plt

tab10 = list(plt.get_cmap('tab10').colors)
set2 = list(plt.get_cmap('Set2').colors)
tab10 = tab10 + set2

CELL_TYPE_COLORS = {
    'B cell':  tab10[0],
    'CD4 T cell': tab10[1], 
    'CD8 T cell': tab10[2],
    'Treg': 'blueviolet',
    'MAIT': 'dodgerblue',
    'NKT': 'red',
    'NK CD56hi': 'pink',
    'NK CD56low': 'saddlebrown',
    #'pDC': 'deeppink',
    'cDC': 'lime',
    'CD14 Mono': 'purple',
    'CD16 Mono': tab10[16],
    'HSPC': 'dimgray',
}

CELL_TYPE_COLORS_ALT = {
    'B Naive': 'burlywood',
    'B Intermediate': 'blue',
    'CD4 Naive': tab10[2],
    'CD4 Memory': tab10[1],
    'CD8 Naive':'turquoise',
    'CD8 Memory': 'yellow', 
    'Treg': 'blueviolet',
    'MAIT': 'dodgerblue',
    'NKT': 'red',
    'NK CD56hi': 'pink',
    'NK CD56low': 'saddlebrown',
#    'pDC': 'deeppink',
    'cDC': 'lime',
    'CD14 Mono': 'purple',
    'CD16 Mono': tab10[16],
    'HSPC': 'dimgray',
}

ALL_CELL_TYPES = list(CELL_TYPE_COLORS.keys())
ALL_CELL_TYPES_ALT = list(CELL_TYPE_COLORS_ALT.keys())

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))
    x_positions = range(len(CELL_TYPE_COLORS_ALT))
    for i, (cell_type, color) in enumerate(CELL_TYPE_COLORS_ALT.items()):
        ax.bar(i, 1, color=color, label=cell_type)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CELL_TYPE_COLORS_ALT.keys(), rotation=90)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.title('similar cells, similar colors')
    plt.tight_layout()
    plt.show()

    x_positions = range(len(CELL_TYPE_COLORS))
    for i, (cell_type, color) in enumerate(CELL_TYPE_COLORS.items()):
        ax.bar(i, 1, color=color, label=cell_type)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(CELL_TYPE_COLORS.keys(), rotation=90)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.title('similar cells, disimilar colors')
    plt.tight_layout()
    plt.show()