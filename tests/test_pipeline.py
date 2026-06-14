import numpy as np

from vrfcd import VRFCDPipeline


def test_pipeline_runs():
    t = np.linspace(0, 1, 500)

    spike_matrix = np.zeros((4, len(t)))
    spike_matrix[0, [100, 200, 300]] = 1
    spike_matrix[1, [105, 205, 305]] = 1
    spike_matrix[2, [350, 400, 450]] = 1
    spike_matrix[3, [355, 405, 455]] = 1

    pipeline = VRFCDPipeline(
        t_R=0.01,
        kernel="minmax",
        community_method="louvain",
        n_jobs=1,
        verbose=0,
    )

    result = pipeline.fit(spike_matrix, t)

    assert result.D.shape == (4, 4)
    assert result.A.shape == (4, 4)
    assert len(result.G.nodes()) == 4
    assert result.partition is not None
