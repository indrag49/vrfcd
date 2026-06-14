import numpy as np

from vrfcd import VRFCDPipeline
from vrfcd.plotting import plot_matrix, plot_functional_network


t = np.linspace(0, 1, 1000)

spike_matrix = np.zeros((6, len(t)))

spike_matrix[0, [100, 300, 500]] = 1
spike_matrix[1, [105, 305, 505]] = 1
spike_matrix[2, [110, 310, 510]] = 1

spike_matrix[3, [600, 750, 900]] = 1
spike_matrix[4, [605, 755, 905]] = 1
spike_matrix[5, [610, 760, 910]] = 1


pipeline = VRFCDPipeline(
    t_R=0.01,
    kernel="minmax",
    community_method="louvain",
    n_jobs=2,
    traces=True,
)

result = pipeline.fit(spike_matrix, t)

print("Partition:", result.partition)
print("Counts:", result.counts)

plot_matrix(
    result.D,
    colorbar_title=r"$\tilde{D}(t_R)$",
    savepath="ExampleD.pdf",
)

plot_matrix(
    result.A,
    colorbar_title=r"$A(t_R)$",
    savepath="ExampleA.pdf",
)

plot_functional_network(
    result.G_partitioned,
    partition=result.partition,
    savepath="ExampleNetwork.pdf",
)
