"""
Pure numpy/scipy signal-processing helpers for previewing a curve's
main peak: Savitzky-Golay smoothing (to suppress point-to-point noise)
followed by scipy.signal.find_peaks (to locate candidate peaks) — used
by ProcessTab's "Automatic" toggle (see gui/process_window.py) to
overlay a smoothed curve and its tallest peak. No delta-energy
matching happens here yet; this only identifies one curve's own peak.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter

# savgol_filter requires window_length <= len(y) (and > polyorder) --
# below this many points, smoothing wouldn't be meaningful anyway.
MIN_POINTS_FOR_SMOOTHING = 7


def smooth_savgol(y: np.ndarray, window_frac: float = 0.075, polyorder: int = 2) -> np.ndarray:
    """
    Savitzky-Golay-smooth `y`, with window_length scaled off the
    data's own length (a fixed fraction of it, floored at
    MIN_POINTS_FOR_SMOOTHING and forced odd) rather than a constant —
    so it adapts to how densely the scan was sampled without needing
    to know the peak width up front. `y` is returned unchanged if
    there are too few points to smooth meaningfully in the first place
    (see MIN_POINTS_FOR_SMOOTHING).
    """
    n = len(y)
    if n < MIN_POINTS_FOR_SMOOTHING:
        return y

    window = max(MIN_POINTS_FOR_SMOOTHING, round(n * window_frac))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)  # savgol_filter requires window_length <= len(y) and odd
    if window <= polyorder:
        return y

    return savgol_filter(y, window_length=window, polyorder=polyorder)


def find_main_peak(
    x: np.ndarray, y: np.ndarray, prominence_frac: float = 0.3
) -> tuple[float, float] | None:
    """
    The tallest (by y value, not necessarily by prominence) of the
    peaks found in `y` with prominence at least `prominence_frac` of
    y's own dynamic range (max - min) — scaled off the range rather
    than the raw max so a nonzero baseline (e.g. a survey scan's
    secondary-electron background) doesn't throw off the threshold.
    The true tallest peak's own prominence is nearly always close to
    its full height above baseline (nothing higher flanks it on either
    side), so a fairly strict fraction still reliably keeps it while
    filtering out noise-induced bumps.

    Returns (x, y) at that peak, or None if no peak clears the
    threshold or the range is degenerate (e.g. a flat curve).
    """
    if len(y) == 0:
        return None
    y_range = float(np.max(y) - np.min(y))
    if y_range <= 0:
        return None

    peak_indices, _ = find_peaks(y, prominence=prominence_frac * y_range)
    if len(peak_indices) == 0:
        return None

    best = peak_indices[np.argmax(y[peak_indices])]
    return float(x[best]), float(y[best])
