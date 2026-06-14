import numpy as np
import pytest

from vrfcd.kernels import distance_to_functional_matrix


def test_clipping_kernel():
    D = np.array(
        [
            [0.0, 0.5, 1.5],
            [0.5, 0.0, 0.2],
            [1.5, 0.2, 0.0],
        ]
    )

    A = distance_to_functional_matrix(D, kernel="clipping")

    assert A.shape == D.shape
    assert np.allclose(np.diag(A), 0.0)
    assert np.all(A >= 0.0)
    assert np.all(A <= 1.0)


def test_exponential_kernel():
    D = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    A = distance_to_functional_matrix(D, kernel="exponential", beta=0.5)

    assert A.shape == D.shape
    assert np.allclose(np.diag(A), 0.0)
    assert A[0, 1] > 0.0


def test_invalid_kernel():
    D = np.eye(3)

    with pytest.raises(ValueError):
        distance_to_functional_matrix(D, kernel="wrong")
