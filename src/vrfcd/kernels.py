import numpy as np


def distance_to_functional_matrix(
    D,
    kernel="minmax",
    beta=0.1,
    q_low=0.0,
    q_high=1.0,
    zero_diagonal=True,
):
    """
    Convert a van Rossum distance matrix into a functional matrix.

    Parameters
    ----------
    D : ndarray
        Distance matrix.
    kernel : {"clipping", "exponential", "minmax"}
        Similarity transformation.
    beta : float
        Scale parameter for the exponential kernel.
    q_low : float
        Lower quantile for minmax scaling.
    q_high : float
        Upper quantile for minmax scaling.
    zero_diagonal : bool
        Whether to set the diagonal of A to zero.

    Returns
    -------
    A : ndarray
        Functional adjacency matrix.
    """

    D = np.asarray(D, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix.")

    if kernel not in {"clipping", "exponential", "minmax"}:
        raise ValueError("kernel must be one of: 'clipping', 'exponential', 'minmax'.")

    if kernel == "clipping":
        D_cap = np.clip(D, 0.0, 1.0)
        A = 1.0 - D_cap

    elif kernel == "exponential":
        if beta <= 0:
            raise ValueError("beta must be positive.")
        A = np.exp(-D / beta)

    elif kernel == "minmax":
        dvals = D[np.triu_indices_from(D, k=1)]

        lo = np.quantile(dvals, q_low)
        hi = np.quantile(dvals, q_high)

        if np.isclose(hi, lo):
            A = np.zeros_like(D)
        else:
            Dnorm = (D - lo) / (hi - lo)
            Dnorm = np.clip(Dnorm, 0.0, 1.0)
            A = 1.0 - Dnorm

    A = 0.5 * (A + A.T)

    if zero_diagonal:
        np.fill_diagonal(A, 0.0)

    return A
