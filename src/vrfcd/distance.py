import numpy as np
from scipy import signal
from joblib import Parallel, delayed


def compute_van_rossum_distance(
    spike_matrix,
    t,
    t_R,
    traces=False,
    n_jobs=6,
    normalise=True,
    verbose=10,
):
    """
    Compute the normalised van Rossum distance matrix.

    Parameters
    ----------
    spike_matrix : array-like, shape (n_neurons, n_time_bins)
        Binary or count spike matrix.
    t : array-like, shape (n_time_bins,)
        Time axis.
    t_R : float
        van Rossum kernel time constant.
    traces : bool, default=False
        If True, also return convolved spike traces.
    n_jobs : int, default=6
        Number of parallel workers.
    normalise : bool, default=True
        If True, divide distances by sqrt average spike count.
    verbose : int, default=10
        Joblib verbosity.

    Returns
    -------
    D : ndarray
        van Rossum distance matrix.
    waveforms : ndarray, optional
        Convolved spike traces.
    """

    t = np.asarray(t, dtype=float)

    if not isinstance(spike_matrix, np.ndarray):
        spike_matrix = np.asarray(spike_matrix.todense())
    else:
        spike_matrix = np.asarray(spike_matrix)

    if spike_matrix.ndim != 2:
        raise ValueError("spike_matrix must have shape (n_neurons, n_time_bins).")

    if spike_matrix.shape[1] != len(t):
        raise ValueError("Length of t must match the number of time bins.")

    if t_R <= 0:
        raise ValueError("t_R must be positive.")

    dt = np.mean(np.diff(t))
    n_neurons, n_time = spike_matrix.shape

    kernel = np.exp(-t / t_R)

    waveforms = np.zeros((n_neurons, n_time))

    for j in range(n_neurons):
        waveforms[j, :] = signal.convolve(
            spike_matrix[j, :],
            kernel,
            mode="full",
        )[:n_time]

    spike_counts = (spike_matrix > 0).sum(axis=1)

    def compute_row(j):
        waveform_difference = waveforms - waveforms[j]

        raw = np.sqrt(dt * np.sum(waveform_difference**2, axis=1) / t_R)

        if normalise:
            avg_spikes = np.sqrt((spike_counts + spike_counts[j]) / 2)

            row = np.divide(
                raw,
                avg_spikes,
                out=np.zeros_like(raw),
                where=avg_spikes > 0,
            )
        else:
            row = raw

        return j, row

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_row)(j) for j in range(n_neurons)
    )

    D = np.zeros((n_neurons, n_neurons))

    for j, row in results:
        D[j, :] = row

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    if traces:
        return D, waveforms

    return D
