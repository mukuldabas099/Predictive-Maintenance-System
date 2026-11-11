"""
calibration.py
----------------
Small, dependency-light helpers (numpy/scipy only -- no TensorFlow) shared by
the model, the live bridge, the demo and the tests, so that training and
serving can never drift apart on HOW features are normalised or HOW the
alarm threshold is derived.

1. Robust scaling:   z = clip((x - median) / scale, -clip, +clip)
     - median / IQR-based scale instead of mean / std, so a few outliers in
       the healthy data don't distort the scale.
     - `scale` has a floor (relative to |median| plus a tiny absolute floor).
       This fixes the bug found earlier where a near-constant healthy
       temperature (std ~0.007 C) turned a normal 25 C reading into a z-score
       of ~ -1350 and saturated the LSTM.
     - clipping at +/-clip bounds the damage any single wild feature can do.

2. Threshold calibration from healthy validation errors:
     - "sigma":      mean + k*std            (the original method; assumes
                                              normally distributed errors)
     - "percentile": q-th percentile * margin (distribution-free)
     - "pot":        peaks-over-threshold (extreme value theory, generalised
                      Pareto tail fit). Falls back to "percentile" if there
                      are too few samples for a meaningful tail fit.

3. Alert persistence: raise an alert only if >= k of the last n windows
   exceeded the threshold (cuts nuisance alerts from isolated spikes).
"""

from collections import deque

import numpy as np

DEFAULT_CLIP = 5.0


# ----------------------------------------------------------------- scaling
def fit_scaler(X_flat, method="robust", rel_floor=0.05, abs_floor=1e-6):
    """Return (center, scale) fitted on (n_samples, n_features) healthy data."""
    X_flat = np.asarray(X_flat, dtype=np.float64)
    if method == "zscore":
        center = X_flat.mean(axis=0)
        scale = X_flat.std(axis=0) + 1e-8
    elif method == "robust":
        center = np.median(X_flat, axis=0)
        q1, q3 = np.percentile(X_flat, [25, 75], axis=0)
        scale = (q3 - q1) / 1.349            # IQR -> std-equivalent for a normal
        # If the IQR is degenerate (e.g. a quantised sensor), fall back to std.
        scale = np.where(scale < abs_floor, X_flat.std(axis=0), scale)
        scale = np.maximum(scale, rel_floor * np.abs(center))
        scale = np.maximum(scale, abs_floor)
    else:
        raise ValueError(f"unknown scaling method: {method}")
    return center.astype(np.float32), scale.astype(np.float32)


def apply_norm(x, center, scale, clip=DEFAULT_CLIP):
    """Normalise (and clip) raw features. clip=None/inf disables clipping."""
    z = (np.asarray(x, dtype=np.float32) - center) / scale
    if clip is not None and np.isfinite(clip):
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)


def load_calibration(path):
    """Load threshold_and_norm.npz. Understands both the new format
    (center/scale/clip) and the original one (mean/std, no clipping)."""
    c = np.load(path)
    if "center" in c.files:
        center, scale = c["center"], c["scale"]
        clip = float(c["clip"]) if "clip" in c.files else None
    else:                                    # legacy file written by old main.py
        center, scale, clip = c["mean"], c["std"], None
    return {
        "threshold": float(c["threshold"]),
        "center": center,
        "scale": scale,
        "clip": clip,
        "seq_len": int(c["seq_len"]),
    }


# -------------------------------------------------------------- thresholds
def compute_threshold(errors, method="percentile", sigma=3.0, q=99.5,
                      margin=1.0, pot_p=1e-3):
    """Alarm threshold from healthy-validation reconstruction errors."""
    errors = np.asarray(errors, dtype=np.float64)
    if method == "sigma":
        return float(errors.mean() + sigma * errors.std())
    if method == "percentile":
        return float(np.percentile(errors, q) * margin)
    if method == "pot":
        thr = _pot_threshold(errors, pot_p)
        if thr is None:                      # too little data for a tail fit
            return float(np.percentile(errors, q) * margin)
        return float(thr * margin)
    raise ValueError(f"unknown threshold method: {method}")


def _pot_threshold(errors, p, tail_frac=0.10, min_excess=15):
    """Peaks-over-threshold: fit a generalised Pareto to the top `tail_frac`
    of errors and return the level exceeded with probability `p`."""
    from scipy.stats import genpareto
    n = len(errors)
    u = np.quantile(errors, 1 - tail_frac)
    excess = errors[errors > u] - u
    if len(excess) < min_excess:
        return None
    try:
        xi, _, beta = genpareto.fit(excess, floc=0.0)
    except Exception:
        return None
    nu = len(excess)
    if abs(xi) < 1e-6:
        return u + beta * np.log(nu / (n * p))
    return u + (beta / xi) * ((n * p / nu) ** (-xi) - 1.0)


# ------------------------------------------------------------- persistence
def persistent_flags(flags, k=3, n=5):
    """Causal k-of-n filter over a boolean sequence: output[i] is True when at
    least k of flags[max(0,i-n+1) : i+1] are True."""
    flags = np.asarray(flags, dtype=bool)
    out = np.zeros(len(flags), dtype=bool)
    for i in range(len(flags)):
        out[i] = flags[max(0, i - n + 1): i + 1].sum() >= k
    return out


class PersistenceFilter:
    """Streaming version of persistent_flags() for the live bridge."""

    def __init__(self, k=3, n=5):
        if not (1 <= k <= n):
            raise ValueError("need 1 <= k <= n")
        self.k, self.n = k, n
        self._buf = deque(maxlen=n)

    def push(self, flag):
        self._buf.append(bool(flag))
        return sum(self._buf) >= self.k

    def reset(self):
        self._buf.clear()


def ewma(scores, alpha=0.3):
    """Exponentially weighted moving average of a score series."""
    scores = np.asarray(scores, dtype=np.float64)
    out = np.empty_like(scores)
    acc = scores[0]
    for i, s in enumerate(scores):
        acc = alpha * s + (1 - alpha) * acc
        out[i] = acc
    return out
