"""RGB conversion matching Chad A. Greene's rgb2hex.m (2014)."""
import numpy as np


def rgb2hex(rgb):
    """Return a hex string for (3,) input, or a list for (N, 3) input.

    If any component exceeds 1, the entire input is interpreted as 0..255.
    Rounding matches MATLAB (half away from zero), not Python's round().
    """
    values = np.asarray(rgb)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError('RGB input must contain real numbers')
    single = values.ndim == 1
    if single:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != 3 or not values.size:
        raise ValueError('RGB input must have shape (3,) or (N, 3)')
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
        raise ValueError('RGB values must be finite and between 0 and 255')
    values = values.astype(float)
    if values.max() <= 1:
        values *= 255
    values = np.floor(values + 0.5).astype(int)
    result = ['#{:02X}{:02X}{:02X}'.format(*row) for row in values]
    return result[0] if single else result
