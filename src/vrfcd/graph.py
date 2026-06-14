import numpy as np
import networkx as nx


def build_functional_graph(
    A,
    node_names=None,
    threshold=0.0,
):
    """
    Build a weighted NetworkX graph from a functional adjacency matrix.

    Parameters
    ----------
    A : ndarray
        Functional adjacency matrix.
    node_names : list, optional
        Node labels.
    threshold : float
        Only edges with weight > threshold are added.

    Returns
    -------
    G : networkx.Graph
        Weighted graph.
    """

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]

    if node_names is None:
        node_names = list(range(n))

    if len(node_names) != n:
        raise ValueError("node_names must have length equal to A.shape[0].")

    G = nx.Graph()
    G.add_nodes_from(node_names)

    for i in range(n):
        for j in range(i + 1, n):
            weight = A[i, j]

            if weight > threshold:
                G.add_edge(
                    node_names[i],
                    node_names[j],
                    weight=float(weight),
                )

    return G
