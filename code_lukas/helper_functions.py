import numpy as np
import leidenalg
import igraph as ig

def get_common_index(series1, series2):
    series1 = series1.copy().dropna()
    series2 = series2.copy().dropna()
    common_index = series1.index.intersection(series2.index)
    return series1.loc[common_index], series2.loc[common_index]

def leiden_clustering(distance_matrix, resolution_parameter=0.5, print_communities=False, invert_distance=False):
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
        resolution_parameter=resolution_parameter
    )

    all_communities = []
    for i, community in enumerate(partition):
        community_nodes = [g.vs[node]["name"] for node in community]
        if print_communities:
            print(f"Community {i+1}: {community_nodes}")
        all_communities.append(community_nodes)

    communities_flat = [item for sublist in all_communities for item in sublist]
    
    return all_communities, communities_flat