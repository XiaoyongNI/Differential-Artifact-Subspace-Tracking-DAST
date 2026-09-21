"""Fixed-weight linear regression reference (Young et al., J Neural Eng 2018).

DOI: 10.1088/1741-2552/aa9ee8. Training is offline, without an intercept;
inference uses only simultaneous channels of the current sample.
"""
import numpy as np


def _signal(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or any(size == 0 for size in X.shape):
        raise ValueError("X must be nonempty (channels, samples)")
    if not np.isfinite(X).all():
        raise ValueError("X must be finite")
    return X


def _weights(W):
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a nonempty square matrix")
    if not np.isfinite(W).all() or np.any(np.diag(W) != 0):
        raise ValueError("W must be finite with an exactly zero diagonal")
    return W


def fit_lrr(X_train, artifact_mask):
    """Fit each channel from all other channels at masked training samples.

    No centering, intercept, regularization, or clean reference is used.
    Rank-deficient fits use NumPy's minimum-norm least-squares solution.
    A single channel has no predictors and returns a zero (1, 1) matrix.
    """
    X_train = _signal(X_train)
    mask = np.asarray(artifact_mask)
    if mask.dtype != np.bool_ or mask.shape != (X_train.shape[1],):
        raise ValueError("artifact_mask must be boolean with shape (n_samples,)")
    if not mask.any():
        raise ValueError("artifact_mask must select at least one training sample")
    Xa = X_train[:, mask]
    channels = np.arange(Xa.shape[0])
    W = np.zeros((len(channels), len(channels)))
    for c in channels:
        others = channels != c
        W[c, others] = np.linalg.lstsq(Xa[others].T, Xa[c], rcond=None)[0]
    return W


class OnlineLRR:
    """Stateless sample-by-sample cancellation with owned, fixed weights.

    Persistent storage is O(C**2); each call costs O(C**2) and uses O(C)
    temporary storage. No history, triggers, fitting, or lookahead is needed.
    """

    def __init__(self, W):
        self._W = _weights(W).copy()
        self._W.flags.writeable = False

    @property
    def W(self):
        """A snapshot; modifying it cannot change the inference weights."""
        return self._W.copy()

    def process_sample(self, x_t):
        """Immediately return x_t - W @ x_t for exactly one (C,) sample."""
        x_t = np.asarray(x_t, dtype=float)
        if x_t.shape != (self._W.shape[0],):
            raise ValueError(f"x_t must have shape ({self._W.shape[0]},)")
        if not np.isfinite(x_t).all():
            raise ValueError("x_t must be finite")
        return x_t - self._W @ x_t

    def reset(self):
        """No-op for API consistency: fixed-weight LRR has no temporal state."""


def apply_lrr_batch(X, W):
    """Offline verification only; online benchmarks use process_sample instead."""
    X, W = _signal(X), _weights(W)
    if X.shape[0] != W.shape[0]:
        raise ValueError("X and W channel counts must match")
    return X - W @ X
