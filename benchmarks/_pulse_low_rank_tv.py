"""Vendored PULSE/SPARC LowRankTV, MIT copyright 2025 Han Bui.

Source commit: 295d0c9fe1195efacb6454fe303109bb83f8c4a4.
Only the external BaseSACMethod import is replaced by a minimal local shim.
The upstream solver and its TV routine are retained for reproducibility.
See LICENSE-PULSE and README.md.
"""
import numpy as np
from typing import Optional
class BaseSACMethod:
    """Minimal upstream base shim; the TBME adapter supplies shape validation."""
    def __init__(self, **kwargs):
        self.is_fitted = False


def _svt(M: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    if np.all(s_thresh == 0):
        return np.zeros_like(M)
    return (U * s_thresh) @ Vt


def _tv1d_denoise(y: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0:
        return y
    n = y.size
    x = np.empty(n, dtype=y.dtype)
    k = 0
    k0 = 0
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam
    for i in range(1, n):
        val = y[i]
        umin += val - vmin
        umax += val - vmax
        if umin < -lam:
            while k <= i - 1:
                x[k] = vmin
                k += 1
            k0 = k = i
            vmin = val - lam
            vmax = val + lam
            umin = lam
            umax = -lam
        elif umax > lam:
            while k <= i - 1:
                x[k] = vmax
                k += 1
            k0 = k = i
            vmin = val - lam
            vmax = val + lam
            umin = lam
            umax = -lam
        else:
            if umin >= lam:
                vmin += (umin - lam) / (i - k0 + 1)
                umin = lam
            if umax <= -lam:
                vmax += (umax + lam) / (i - k0 + 1)
                umax = -lam
    v = vmin + umin / (n - k0 + 0)
    w = vmax + umax / (n - k0 + 0)
    vout = v if v <= w else (vmin + vmax) / 2
    while k <= n - 1:
        x[k] = vout
        k += 1
    return x


def _tv1d_rows(Z: np.ndarray, lam: float) -> np.ndarray:
    C, T = Z.shape
    out = np.empty_like(Z)
    for c in range(C):
        out[c] = _tv1d_denoise(Z[c], lam)
    return out


class LowRankTV(BaseSACMethod):
    def __init__(self,
                 lambda_tv: float = 1.0,
                 rho: float = 1.0,
                 max_iters: int = 200,
                 tol: float = 1e-4,
                 features_axis: Optional[int] = 1,
                 time_axis: Optional[int] = -1,
                 verbose: bool = True,
                 print_every: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.lambda_tv = float(lambda_tv)
        self.rho = float(rho)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self._features_axis = features_axis
        self._time_axis = time_axis
        self.is_fitted = False
        self.verbose = bool(verbose)
        self.print_every = int(print_every)

    def fit(self, data: np.ndarray, artifact_markers: Optional[np.ndarray] = None, **kwargs) -> 'LowRankTV':
        self.is_fitted = True
        return self

    def _solve_trial(self, X_tc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C, T = X_tc.shape
        L = np.zeros((C, T), dtype=X_tc.dtype)
        S = np.zeros((C, T), dtype=X_tc.dtype)
        U = np.zeros((C, T), dtype=X_tc.dtype)
        inv_rho = 1.0 / self.rho
        prev_obj = np.inf
        it = 0
        for it in range(1, self.max_iters + 1):
            Z_L = X_tc - S - U
            # L = _svt(Z_L, inv_rho
            L = _svt(Z_L.T, inv_rho).T # calculate the smoothness in the frequency domain
            Z_S = X_tc - L - U
            S = _tv1d_rows(Z_S, self.lambda_tv * inv_rho)
            R = L + S - X_tc
            U = U + R
            fro = np.linalg.norm(R, 'fro')
            obj = np.sum(np.linalg.svd(L, compute_uv=False)) + self.lambda_tv * np.sum(np.abs(np.diff(S, axis=1)))
            if self.verbose and (it == 1 or it % self.print_every == 0):
                print(f"  iter {it:4d}  fro={fro:.3e}  obj={obj:.6e}")
            if fro < self.tol and abs(prev_obj - obj) / (obj + 1e-8) < 1e-5:
                if self.verbose:
                    print(f"  converged at iter {it}  fro={fro:.3e}  obj={obj:.6e}")
                break
            prev_obj = obj
        return L, S

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transforming data.")
        if data.ndim != 3:
            raise ValueError("Data must be 3D (trials, channels, timesteps)")
        axes = list(range(3))
        ch_axis = self._features_axis if self._features_axis is not None else 1
        t_axis = self._time_axis if self._time_axis is not None else 2
        if ch_axis == t_axis:
            raise ValueError("features_axis and time_axis must be different")
        if axes[1] != ch_axis or axes[2] != t_axis:
            perm = [0, ch_axis, t_axis]
            X = np.transpose(data, perm)
        else:
            X = data
        n_trials, C, T = X.shape
        S_out = np.empty_like(X)
        for i in range(n_trials):
            if self.verbose:
                print(f"trial {i+1}/{n_trials}  C={C} T={T}")
            _, S = self._solve_trial(X[i])
            S_out[i] = S
            if self.verbose:
                print(f"done trial {i+1}/{n_trials}")
        if axes[1] != ch_axis or axes[2] != t_axis:
            inv_perm = [0, 0, 0]
            inv_perm[0] = 0
            inv_perm[ch_axis] = 1
            inv_perm[t_axis] = 2
            S_out = np.transpose(S_out, inv_perm)
        return S_out


