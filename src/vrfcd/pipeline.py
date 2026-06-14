from dataclasses import dataclass

from .distance import compute_van_rossum_distance
from .kernels import distance_to_functional_matrix
from .graph import build_functional_graph
from .communities import (
    detect_communities,
    add_community_attributes,
    make_partition_weighted_graph,
    community_counts,
)


@dataclass
class VRFCDResult:
    D: object
    A: object
    G: object
    partition: dict | None = None
    G_partitioned: object | None = None
    counts: object | None = None
    waveforms: object | None = None


class VRFCDPipeline:
    """
    End-to-end pipeline:

        spike matrix -> van Rossum distance -> functional matrix
        -> weighted graph -> community labels
    """

    def __init__(
        self,
        t_R,
        kernel="minmax",
        beta=0.1,
        q_low=0.0,
        q_high=1.0,
        graph_threshold=0.0,
        community_method="louvain",
        seed=42,
        n_jobs=6,
        traces=False,
        normalise=True,
        verbose=10,
    ):
        self.t_R = t_R
        self.kernel = kernel
        self.beta = beta
        self.q_low = q_low
        self.q_high = q_high
        self.graph_threshold = graph_threshold
        self.community_method = community_method
        self.seed = seed
        self.n_jobs = n_jobs
        self.traces = traces
        self.normalise = normalise
        self.verbose = verbose

    def fit(self, spike_matrix, t, node_names=None, detect=True):
        """
        Run the full VRFCD pipeline.
        """

        if self.traces:
            D, waveforms = compute_van_rossum_distance(
                spike_matrix,
                t,
                self.t_R,
                traces=True,
                n_jobs=self.n_jobs,
                normalise=self.normalise,
                verbose=self.verbose,
            )
        else:
            D = compute_van_rossum_distance(
                spike_matrix,
                t,
                self.t_R,
                traces=False,
                n_jobs=self.n_jobs,
                normalise=self.normalise,
                verbose=self.verbose,
            )
            waveforms = None

        A = distance_to_functional_matrix(
            D,
            kernel=self.kernel,
            beta=self.beta,
            q_low=self.q_low,
            q_high=self.q_high,
        )

        G = build_functional_graph(
            A,
            node_names=node_names,
            threshold=self.graph_threshold,
        )

        if detect:
            partition = detect_communities(
                G,
                method=self.community_method,
                seed=self.seed,
            )

            G_with_partition = add_community_attributes(G, partition)

            G_partitioned = make_partition_weighted_graph(
                G_with_partition,
                partition,
            )

            counts = community_counts(partition)

        else:
            partition = None
            G_partitioned = None
            counts = None

        return VRFCDResult(
            D=D,
            A=A,
            G=G,
            partition=partition,
            G_partitioned=G_partitioned,
            counts=counts,
            waveforms=waveforms,
        )
