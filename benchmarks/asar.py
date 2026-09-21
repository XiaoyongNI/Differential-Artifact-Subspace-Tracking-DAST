"""Floating-point ASAR using the supplied MATLAB NLMS update convention.

Reference: Basir-Kazeruni et al., NER 2017, doi:10.1109/NER.2017.8008322,
equations (1)-(5). Uses neighboring references with the supplied MATLAB
201-tap defaults, rather than reproducing the paper's 16-tap hardware.
"""
import numpy as np


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def calibrate_asar(X, n_samples=500):
    """Return per-channel mean and sample std from the first N calibration samples.

    X has shape (channels, samples). Supply a preceding artifact-free interval
    for causal deployment. Statistics stay fixed during subsequent adaptation.
    """
    n_samples = _positive_integer(n_samples, "n_samples", 2)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < n_samples:
        raise ValueError("Calibration requires (channels, samples) with at least n_samples samples")
    prefix = X[:, :n_samples]
    if not np.isfinite(prefix).all():
        raise ValueError("Calibration samples must be finite")
    # Centered variance avoids cancellation in T - N * avg**2; ddof=1
    # retains the MATLAB sample-standard-deviation convention.
    return prefix.mean(axis=1), prefix.std(axis=1, ddof=1)


class OnlineASAR:
    """One multichannel sample per call, with C independent adaptive FIR filters.

    mean/std describe all input channels; reference_channels[c] chooses the
    simultaneous reference for target c (default: next channel, or previous
    for the last channel). Only the gated
    current and past references are stored. No stimulation markers are needed.
    """

    def __init__(self, mean, std, filter_length=201, mu=0.1, threshold=5.,
                 epsilon=1e-4, reference_channels=None, weights=None):
        mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        if mean.ndim != 1 or mean.size == 0 or std.shape != mean.shape:
            raise ValueError("mean and std must have the same nonempty (channels,) shape")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("Statistics must be finite with nonnegative std")
        length = _positive_integer(filter_length, "filter_length")
        if not np.isfinite([mu, threshold, epsilon]).all() or not 0 < mu < 2 or threshold < 0 or epsilon <= 0:
            raise ValueError("Require 0 < mu < 2, threshold >= 0, epsilon > 0, all finite")
        if reference_channels is None:
            if mean.size < 2:
                raise ValueError("Neighbor-reference ASAR requires at least two channels")
            refs = np.minimum(np.arange(mean.size) + 1, mean.size - 1)
            refs[-1] = mean.size - 2
        else:
            refs = np.asarray(reference_channels)
        if (refs.shape != mean.shape or refs.dtype.kind not in 'iu'
                or np.any(refs < 0) or np.any(refs >= mean.size)):
            raise ValueError("reference_channels must contain one valid integer channel index per target")
        self._refs = refs.astype(int, copy=True)
        self._mean = mean[self._refs].copy()
        self._limit = threshold * std[self._refs]
        self.mu, self.epsilon = float(mu), float(epsilon)
        self._initial_weights = np.zeros((mean.size, length))
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != self._initial_weights.shape or not np.isfinite(weights).all():
                raise ValueError("weights must be finite with shape (channels, filter_length)")
            self._initial_weights[:] = weights
        self.reset()

    @property
    def reference_channels(self):
        """Snapshot of the resolved zero-based reference mapping."""
        return self._refs.copy()

    @property
    def weights(self):
        """Snapshot of current adaptive coefficients."""
        return self._weights.copy()

    def reset(self):
        """Clear reference history and restore initial coefficients; retain calibration."""
        self._weights = self._initial_weights.copy()
        self._history = np.zeros_like(self._weights)

    def process_sample(self, x_t, *, adapt=True):
        """Return the post-update residual; adapt=False is for frozen rest evaluation."""
        x_t = np.asarray(x_t, dtype=float)
        if x_t.shape != (self._weights.shape[0],) or not np.isfinite(x_t).all():
            raise ValueError(f"x_t must be a finite ({self._weights.shape[0]},) sample")
        reference = x_t[self._refs]
        self._history[:, 1:] = self._history[:, :-1]
        self._history[:, 0] = np.where(np.abs(reference-self._mean) >= self._limit, reference, 0.)
        u = self._history
        if adapt:
            error = x_t - np.einsum('ij,ij->i', u, self._weights)
            energy = np.einsum('ij,ij->i', u, u)
            self._weights += (self.mu * error / (energy + self.epsilon))[:, None] * u
        # Intentionally recompute with updated weights (MATLAB Dout_clean,
        # paper Eq. 5), rather than returning the pre-update error.
        return x_t - np.einsum('ij,ij->i', u, self._weights)
