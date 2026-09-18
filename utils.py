"""Utility functions shared by loaders, pipeline, metrics, and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def concatenate_trials(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    return np.concatenate([X[i] for i in range(X.shape[0])], axis=1)


def svd_components(X: np.ndarray, rank: int) -> np.ndarray:
    U, _, _ = np.linalg.svd(np.asarray(X, dtype=float), full_matrices=False)
    return U[:, :rank]


def ideal_u1s_from_stim_geometry(
    stim_channels: Sequence[int],
    num_channels: int,
    rank: Optional[int] = None,
    k1: float = -0.201,
    strip_size: int = 8,
    strip_distance: float = 1e-1,
    same_strip_distance: float = 1e-3,
) -> np.ndarray:
    ideals = np.zeros((num_channels, len(stim_channels)), dtype=float)
    for j, stim_channel in enumerate(stim_channels):
        stim_strip = int(stim_channel) // strip_size
        distances = np.zeros(num_channels, dtype=float)
        for ch in range(num_channels):
            ch_strip = ch // strip_size
            if ch_strip == stim_strip:
                distances[ch] = abs(ch - int(stim_channel)) * same_strip_distance
            else:
                distances[ch] = abs(ch_strip - stim_strip) * strip_distance
        amps = 10 ** (k1 * distances)
        ideals[:, j] = amps / (np.linalg.norm(amps) + 1e-12)
    rank = len(stim_channels) if rank is None else int(rank)
    U, _, _ = np.linalg.svd(ideals, full_matrices=False)
    return U[:, :rank]


def align_vector_sign(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return -u if np.dot(u, ref) < 0 else u


def subspace_angle_deg(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    Uq, _ = np.linalg.qr(U)
    Vq, _ = np.linalg.qr(V)
    s = np.linalg.svd(Uq.T @ Vq, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def one_sided_spectrum(x: np.ndarray, fs: float, remove_mean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    if remove_mean:
        x = x - x.mean()
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x))
    return freqs, mag


def harmonic_energy_ratio(X: np.ndarray, fs: float, stim_rate: float, max_harmonics: int = 10, bw_hz: float = 3.0) -> float:
    X = np.asarray(X, dtype=float)
    freqs = np.fft.rfftfreq(X.shape[-1], d=1.0 / fs)
    psd = np.abs(np.fft.rfft(X - X.mean(axis=-1, keepdims=True), axis=-1)) ** 2
    mask = np.zeros_like(freqs, dtype=bool)
    for h in range(1, max_harmonics + 1):
        center = h * stim_rate
        if center >= fs / 2:
            break
        mask |= np.abs(freqs - center) <= bw_hz
    harmonic = float(psd[..., mask].sum())
    total = float(psd.sum()) + 1e-12
    return harmonic / total


def rms(x: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2, axis=axis))

