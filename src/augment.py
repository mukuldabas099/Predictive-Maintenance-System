"""
augment.py -- signal-level augmentation of HEALTHY recordings.

Applied to the raw vibration (then features are recomputed), so every
augmented window is a physically plausible healthy signal:
  - time warp (resample by +/- warp): all frequencies scale by the factor, which
    mimics a small speed change (shaft and fault frequencies move together)
  - amplitude scaling: mimics load / mounting / gain differences
  - additive noise: mimics sensor and background noise
This targets exactly what broke thresholds in the leave-one-load-out tests:
an operating condition the model has not seen.
"""
import numpy as np


def augment_signal(x, rng, scale=(0.9, 1.1), noise_rel=0.05, warp=(0.97, 1.03)):
    x = np.asarray(x, dtype=np.float64)
    f = rng.uniform(*warp)
    xw = np.interp(np.arange(0, len(x) - 1, f), np.arange(len(x)), x)
    xw = xw * rng.uniform(*scale)
    xw = xw + rng.normal(0.0, noise_rel * (xw.std() + 1e-12), len(xw))
    return xw.astype(np.float32)


def augment_run(run, rng, **kw):
    """Same dict shape as the loaders; vibration augmented, temperature kept."""
    v = augment_signal(run["vibration"], rng, **kw)
    n = len(v)
    return {**run, "vibration": v, "time": np.arange(n) / run["sample_rate"],
            "temperature": np.resize(run["temperature"], n), "state": np.resize(run["state"], n)}
