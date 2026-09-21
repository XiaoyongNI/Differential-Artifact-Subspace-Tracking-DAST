"""Core artifact-cancellation algorithm blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype="low", analog=False)


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    if not 0 < lowcut < highcut < nyq:
        raise ValueError(f"Expected 0 < lowcut < highcut < Nyquist ({nyq:g} Hz).")
    return butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass", analog=False)


def apply_lpf_3d(X: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    b, a = butter_lowpass(min(cutoff, 0.45 * fs), fs, order=order)
    return lfilter(b, a, np.asarray(X, dtype=float), axis=-1)


def apply_bpf_3d(X: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, np.asarray(X, dtype=float), axis=-1)


def derivative_time(X: np.ndarray, order: int = 1) -> np.ndarray:
    out = np.asarray(X, dtype=float)
    for _ in range(order):
        out = np.diff(out, axis=-1)
    return out


def nth_derivative_time(X: np.ndarray, n: int = 1) -> np.ndarray:
    """Compute nth-order finite differences along the last (time) axis.

    Recursively applies delta^n x[t] = delta^(n-1) x[t] - delta^(n-1) x[t-1].
    Each step shortens the time axis by one, down to zero; n=0 returns
    the input as a floating-point array.
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    return derivative_time(X, order=n)


def integrate_time(dX: np.ndarray, x0: np.ndarray) -> np.ndarray:
    dX = np.asarray(dX, dtype=float)
    y = np.zeros((dX.shape[0], dX.shape[1], dX.shape[2] + 1), dtype=float)
    y[:, :, 0] = x0
    y[:, :, 1:] = x0[..., None] + np.cumsum(dX, axis=-1)
    return y


@dataclass
class NormalizationStats:
    mode: str
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    debug: Optional[dict] = None


def normalize_channels(
    X: np.ndarray,
    mode: str = "none",
    eps: float = 1e-12,
    alpha_mean: float = 1.0 / 512.0,
    alpha_scale: float = 1.0 / 512.0,
    init_scale: float = 1.0,
) -> tuple[np.ndarray, NormalizationStats]:
    X = np.asarray(X, dtype=float)
    if mode in (None, "none"):
        return X.copy(), NormalizationStats(mode="none")
    if mode == "global_per_channel":
        mean = X.mean(axis=(0, 2), keepdims=True)
        scale = X.std(axis=(0, 2), keepdims=True).clip(min=eps)
        return (X - mean) / scale, NormalizationStats(mode=mode, mean=mean, scale=scale)
    if mode == "per_trial":
        mean = X.mean(axis=(1, 2), keepdims=True)
        scale = X.std(axis=(1, 2), keepdims=True).clip(min=eps)
        return (X - mean) / scale, NormalizationStats(mode=mode, mean=mean, scale=scale)
    if mode == "streaming_causal":
        return streaming_channel_normalization_3d(
            X,
            alpha_mean=alpha_mean,
            alpha_scale=alpha_scale,
            eps=eps,
            init_scale=init_scale,
        )
    raise ValueError(f"Unknown normalization mode '{mode}'.")


def apply_normalization_stats(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    if stats.mode == "none":
        return np.asarray(X, dtype=float).copy()
    if stats.mode in ("global_per_channel", "per_trial"):
        return (np.asarray(X, dtype=float) - stats.mean) / stats.scale
    raise ValueError(f"Cannot replay normalization stats for mode '{stats.mode}'.")


def denormalize_channels(
    X: np.ndarray, stats: NormalizationStats, time_offset: int = 0,
) -> np.ndarray:
    """Restore input units, aligning streaming histories after time cropping."""
    X = np.asarray(X, dtype=float)
    if stats.mode == "none":
        return X.copy()
    if stats.mode in ("global_per_channel", "per_trial"):
        return X * stats.scale + stats.mean
    if stats.mode == "streaming_causal":
        stop = time_offset + X.shape[-1]
        mean = stats.debug["mu_hist"][..., time_offset:stop]
        scale = stats.debug["scale_hist"][..., time_offset:stop]
        if time_offset < 0 or mean.shape != X.shape or scale.shape != X.shape:
            raise ValueError("Streaming normalization histories must align with X.")
        return X * scale + mean
    raise ValueError(f"Unknown normalization mode '{stats.mode}'.")


def streaming_channel_normalization_3d(
    X: np.ndarray,
    alpha_mean: float = 1.0 / 512.0,
    alpha_scale: float = 1.0 / 512.0,
    eps: float = 1e-6,
    init_scale: float = 1.0,
) -> tuple[np.ndarray, NormalizationStats]:
    X = np.asarray(X, dtype=float)
    n_trials, n_channels, n_time = X.shape
    Y = np.zeros_like(X, dtype=float)
    mu_hist = np.zeros_like(X, dtype=float)
    scale_hist = np.zeros_like(X, dtype=float)
    for i in range(n_trials):
        mu = np.zeros(n_channels, dtype=float)
        scale = np.full(n_channels, float(init_scale), dtype=float)
        for t in range(n_time):
            x_t = X[i, :, t]
            mu = mu + alpha_mean * (x_t - mu)
            e_t = x_t - mu
            scale = scale + alpha_scale * (np.abs(e_t) - scale)
            scale_safe = np.maximum(scale, eps)
            Y[i, :, t] = e_t / scale_safe
            mu_hist[i, :, t] = mu
            scale_hist[i, :, t] = scale_safe
    stats = NormalizationStats(
        mode="streaming_causal",
        debug={
            "mu_hist": mu_hist,
            "scale_hist": scale_hist,
            "alpha_mean": alpha_mean,
            "alpha_scale": alpha_scale,
            "eps": eps,
            "init_scale": init_scale,
        },
    )
    return Y, stats


class PASTd:
    def __init__(self, n_channels: int, rank: int, beta: float = 0.999, reorth_interval: int = 10**9):
        self.n_channels = int(n_channels)
        self.rank = int(rank)
        self.beta = float(beta)
        self.reorth_interval = int(reorth_interval)
        self.reset()

    def reset(self):
        self.t = 0
        self.U = np.eye(self.n_channels, self.rank)
        self.d = np.ones(self.rank)
        self.delta_u = np.zeros(self.rank)
        self.delta_u_ema = np.zeros(self.rank)

    def warm_start(self, X_init: np.ndarray):
        U, S, _ = np.linalg.svd(X_init, full_matrices=False)
        self.U = U[:, :self.rank]
        self.d = (S[:self.rank] ** 2) / max(X_init.shape[1], 1)

    def update(self, update_x: np.ndarray, project_x: Optional[np.ndarray] = None, track_delta: bool = True):
        update_x = np.asarray(update_x, dtype=float).ravel()
        project_x = update_x if project_x is None else np.asarray(project_x, dtype=float).ravel()
        residual_update = update_x.copy()
        y = np.zeros(self.rank)
        for j in range(self.rank):
            u_old = self.U[:, j].copy()
            y_update = u_old @ residual_update
            y[j] = u_old @ project_x
            e_update = residual_update - u_old * y_update
            self.d[j] = self.beta * self.d[j] + y_update**2
            if self.d[j] != 0:
                self.U[:, j] = u_old + e_update * (y_update / self.d[j])
                nrm = np.linalg.norm(self.U[:, j])
                if nrm > 0:
                    self.U[:, j] /= nrm
            if track_delta:
                self.delta_u[j] = np.linalg.norm(self.U[:, j] - u_old)
                self.delta_u_ema[j] = self.beta * self.delta_u_ema[j] + (1.0 - self.beta) * self.delta_u[j]
            residual_update = e_update

        self.t += 1
        if self.reorth_interval > 0 and self.t % self.reorth_interval == 0:
            self.U, _ = np.linalg.qr(self.U)
        return y

    def get_components(self) -> np.ndarray:
        return self.U


class MCC_OPAST:
    def __init__(self, n_channels: int, rank: int, beta: float = 0.999, sigma: float = 1.0, reorth_interval: int = 100):
        self.n_channels = int(n_channels)
        self.rank = int(rank)
        self.beta = float(beta)
        self.sigma2 = float(sigma) ** 2
        self.reorth_interval = int(reorth_interval)
        self.reset()

    def reset(self):
        self.t = 0
        self.U = np.eye(self.n_channels, self.rank)
        self.P = np.eye(self.rank)

    def warm_start(self, X_init: np.ndarray):
        U, _, _ = np.linalg.svd(X_init, full_matrices=False)
        self.U = U[:, :self.rank]

    def update(self, update_x: np.ndarray, project_x: Optional[np.ndarray] = None):
        x = np.asarray(update_x, dtype=float).reshape(-1, 1)
        y = self.U.T @ x
        residual = x - self.U @ y
        weight = np.exp(-float(np.sum(residual**2)) / (2.0 * self.sigma2))
        if weight >= 1e-10:
            Py = self.P @ y
            gain = Py / (self.beta + weight * float(y.T @ Py))
            self.P = (self.P - weight * (gain @ Py.T)) / self.beta
            self.U = self.U + weight * (residual @ gain.T)
            self.t += 1
            if self.reorth_interval > 0 and self.t % self.reorth_interval == 0:
                self.U, _ = np.linalg.qr(self.U)
        px = x if project_x is None else np.asarray(project_x, dtype=float).reshape(-1, 1)
        return (self.U.T @ px).ravel()

    def get_components(self) -> np.ndarray:
        return self.U


def choose_k_from_delta_ema(delta_u_ema: np.ndarray, beta: float, threshold: float, min_k: int = 1) -> int:
    delta_u_ema = np.asarray(delta_u_ema, dtype=float).ravel()
    if delta_u_ema.size == 0:
        return int(min_k)
    excess = (delta_u_ema - delta_u_ema[0]) / max(1.0 - beta, 1e-12)
    chosen = int(min_k)
    for j in range(max(min_k, 1), delta_u_ema.size):
        if excess[j] < threshold:
            chosen = j + 1
        else:
            break
    return chosen


def project_and_remove(x: np.ndarray, U: np.ndarray, rank: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    U = np.asarray(U, dtype=float)
    rank = U.shape[1] if rank is None else int(rank)
    U_active = U[:, :rank]
    return x - U_active @ (U_active.T @ x)


def covariance_aware_subspace_suppression(
    x: np.ndarray, U: np.ndarray, Rn: np.ndarray, Rx: np.ndarray,
    lambda_reg: float = 1.0, eps: float = 1e-6,
) -> np.ndarray:
    """Suppress excess artifact power using W = Rn @ solve(A, I).

    x is a channel vector or channels-by-samples matrix in the same units as
    Rn/Rx. Only the supplied (currently active) columns of U are suppressed.
    Artifact powers are max(diag(U.T @ (Rx-Rn) @ U), 0), not U.T @ x.
    Tiny negative covariance eigenvalues from rounding are clipped; materially
    indefinite matrices are rejected. No orthogonalization of U is performed.
    """
    x, U = np.asarray(x, dtype=float), np.asarray(U, dtype=float)
    if x.ndim not in (1, 2) or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a finite channel vector or channels-by-samples matrix")
    channels = x.shape[0]
    if U.ndim != 2 or U.shape[0] != channels or U.shape[1] > channels or not np.isfinite(U).all():
        raise ValueError("U must be finite with shape (channels, active_components)")
    if not np.isfinite(lambda_reg) or lambda_reg < 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("lambda_reg must be nonnegative and eps strictly positive and finite")

    def checked_covariance(matrix, name):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (channels, channels) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite channels-by-channels covariance")
        matrix = (matrix + matrix.T) * 0.5
        eigenvalues, vectors = np.linalg.eigh(matrix)
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if eigenvalues[0] < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite")
        if eigenvalues[0] < 0:
            matrix = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
        return matrix

    Rn = checked_covariance(Rn, "Rn")
    Rx = checked_covariance(Rx, "Rx")
    powers = np.maximum(np.sum(U * ((Rx - Rn) @ U), axis=0), 0.0)
    A = Rn + lambda_reg * (U * powers) @ U.T + eps * np.eye(channels)
    A = (A + A.T) * 0.5
    # A is symmetric positive definite; solve A z = x, then Rn z = W x.
    try:
        output = Rn @ cho_solve(cho_factor(A, lower=True), x)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance solve failed; increase covariance-eps for this signal scale") from exc
    if not np.isfinite(output).all():
        raise ValueError("Covariance suppression produced nonfinite output")
    return output


def build_harmonic_freqs(stim_rate: float, fs: float, max_harmonics: Optional[int] = None) -> np.ndarray:
    n_max = int(np.floor((fs / 2.0) / stim_rate))
    if max_harmonics is not None:
        n_max = min(n_max, int(max_harmonics))
    return np.array([k * stim_rate for k in range(1, n_max + 1)], dtype=float)


def harmonic_rls(x: np.ndarray, fs: float, stim_rate: float, max_harmonics: int = 10, settling_time: float = 1e-2) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]
    C, N = x.shape
    x0 = x - x.mean(axis=1, keepdims=True)
    M = int(max_harmonics)
    kappa_f = np.cos(2 * np.pi * stim_rate / fs)
    kappa_k = np.zeros(M + 2)
    kappa_k[1] = 1.0
    kappa_k[0] = kappa_f
    for k in range(M):
        kappa_k[k + 2] = 2 * kappa_f * kappa_k[k + 1] - kappa_k[k]
    lam = np.exp(np.log(0.05) / (settling_time * fs + 1))
    u_kp = np.ones(M)
    u_k = np.ones(M)
    r1 = 10.0 * np.ones((C, M))
    r4 = 10.0 * np.ones((C, M))
    a = np.zeros((C, M))
    b = np.zeros((C, M))
    y = np.zeros((C, N))
    for n in range(N):
        e = x0[:, n].copy()
        for k in range(M):
            tmp = kappa_k[k + 2] * (u_kp[k] + u_k[k])
            tmp2 = u_kp[k]
            u_kp[k] = tmp - u_k[k]
            u_k[k] = tmp + tmp2
            denom = kappa_k[k + 2] + 1
            G = 1.0 if denom == 0 else 1.5 - (u_kp[k] ** 2 - (kappa_k[k + 2] - 1) / denom * u_k[k] ** 2)
            if G <= 0:
                G = 1.0
            u_kp[k] *= G
            u_k[k] *= G
            e -= a[:, k] * u_k[k] + b[:, k] * u_kp[k]
            r1[:, k] = lam * r1[:, k] + u_k[k] ** 2
            r4[:, k] = lam * r4[:, k] + u_kp[k] ** 2
            a[:, k] += u_k[k] * e / r1[:, k]
            b[:, k] += u_kp[k] * e / r4[:, k]
        y[:, n] = e
    return y[0] if single else y


def harmonic_rls_3d(X: np.ndarray, fs: float, stim_rate: float, max_harmonics: int = 10, settling_time: float = 1e-2) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.zeros_like(X)
    for tr in range(X.shape[0]):
        Y[tr] = harmonic_rls(X[tr], fs=fs, stim_rate=stim_rate, max_harmonics=max_harmonics, settling_time=settling_time)
    return Y

