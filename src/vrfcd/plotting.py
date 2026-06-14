import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_matrix(
    M,
    title=None,
    colorbar_title=None,
    cmap="inferno",
    figsize=(7, 6),
    xlabel="Neuron index $j$",
    ylabel="Neuron index $i$",
    savepath=None,
    fontsize=18,
):
    """
    Plot a distance or functional matrix.
    """

    M = np.asarray(M)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        M,
        cmap=cmap,
        vmin=np.min(M),
        vmax=np.max(M),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.03)

    if colorbar_title is not None:
        cbar.ax.set_title(colorbar_title, fontsize=fontsize, pad=10)

    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.tick_params(axis="both", labelsize=fontsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_functional_network(
    G,
    partition=None,
    pos=None,
    weight_attr="weight",
    layout_seed=42,
    figsize=(5, 4),
    node_size=500,
    title=r"Network representation of $A(t_R)$",
    savepath=None,
    fontsize=18,
):
    """
    Plot weighted functional network.

    If partition is supplied, nodes are coloured by community.
    """

    if pos is None:
        pos = nx.spring_layout(G, weight=weight_attr, seed=layout_seed)

    fig, ax = plt.subplots(figsize=figsize)

    if partition is None:
        node_colors = "lightgray"
    else:
        node_colors = [partition[n] for n in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.2,
        cmap=plt.cm.tab10,
        ax=ax,
    )

    widths = [max(0.2, G[u][v].get(weight_attr, 1.0)) for u, v in G.edges()]

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=0.8,
        edge_color="black",
        ax=ax,
    )

    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax, pos
