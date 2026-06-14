from collections import Counter

import networkx as nx
from networkx.algorithms.community import louvain_communities


def detect_communities(
    G,
    method="louvain",
    seed=42,
    resolution=1.0,
    threshold=1e-10,
):
    """
    Detect communities in a weighted functional graph.

    Parameters
    ----------
    G : networkx.Graph
        Weighted graph.
    method : {"louvain", "leiden"}
        Community detection method.
    seed : int
        Random seed.
    resolution : float
        Louvain resolution parameter.
    threshold : float
        Louvain threshold parameter.

    Returns
    -------
    partition : dict
        Mapping node -> community label.
    """

    if method == "louvain":
        communities = louvain_communities(
            G,
            weight="weight",
            seed=seed,
            resolution=resolution,
            threshold=threshold,
        )

        partition = {node: cid for cid, comm in enumerate(communities) for node in comm}

    elif method == "leiden":
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError(
                "Leiden requires optional dependencies. "
                "Install using: pip install -e '.[leiden]'"
            ) from exc

        nodes = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

        weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]

        g_ig = ig.Graph()
        g_ig.add_vertices(len(nodes))
        g_ig.add_edges(edges)
        g_ig.es["weight"] = weights
        g_ig.vs["name"] = nodes

        part = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            seed=seed,
        )

        partition = {}

        for cid, comm in enumerate(part):
            for idx in comm:
                partition[g_ig.vs[idx]["name"]] = cid

    else:
        raise ValueError("method must be 'louvain' or 'leiden'.")

    return partition


def add_community_attributes(G, partition, attribute="community"):
    """
    Add community labels as node attributes.
    """

    G_out = G.copy()
    nx.set_node_attributes(G_out, partition, attribute)
    return G_out


def community_counts(partition):
    """
    Count nodes in each community.
    """

    return Counter(partition.values())


def make_partition_weighted_graph(
    G,
    partition,
    within_scale=2.0,
    between_scale=0.5,
):
    """
    Modify edge weights for plotting.

    Within-community edges are strengthened and between-community edges
    are weakened.
    """

    G_comm = G.copy()

    for u, v, data in G_comm.edges(data=True):
        w = data.get("weight", 1.0)

        if partition[u] == partition[v]:
            data["weight"] = w * within_scale
        else:
            data["weight"] = w * between_scale

    return G_comm
