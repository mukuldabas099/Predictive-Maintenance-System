"""
drift.py -- detect when the machine's "healthy" behaviour has moved away from
the data the model was calibrated on (new load/speed, sensor re-mounted,
season), so you know when the threshold needs recalibrating.

  psi()                    population stability index of one feature
                           (<0.1 stable, 0.1-0.25 moderate, >0.25 major shift)
  FeatureDriftMonitor      per-feature PSI + two-sample KS test of recent
                           windows vs. the healthy reference
  PageHinkley              streaming change detector for the anomaly score

IMPORTANT: drift is not a fault. A fault also changes the features. Use
FeatureDriftMonitor on windows the model itself calls NORMAL (score below the
threshold); if those drift, the baseline is stale. If the score is high, it's
an alarm, not drift. The monitor's `update(features, is_normal)` enforces that.
"""
from collections import deque

import numpy as np
from scipy.stats import ks_2samp


def psi(reference, current, bins=10, eps=1e-4):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # (near-)constant reference feature
        return 0.0 if np.allclose(current, reference.mean(), atol=1e-9) else np.inf
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


class FeatureDriftMonitor:
    def __init__(self, reference_features, feature_names, window=120,
                 psi_threshold=0.25, ks_alpha=0.001, min_features=2):
        self.ref = np.asarray(reference_features, dtype=float)
        self.names = list(feature_names)
        self.psi_threshold, self.ks_alpha, self.min_features = psi_threshold, ks_alpha, min_features
        self.buf = deque(maxlen=window)
        self.window = window

    def update(self, features, is_normal=True):
        """Feed one feature vector. Windows the detector flags as anomalous are
        ignored (they are faults, not drift)."""
        if is_normal:
            self.buf.append(np.asarray(features, dtype=float))

    def check(self):
        """Returns dict(drifted, features=[...], details) or None until the buffer is full."""
        if len(self.buf) < self.window:
            return None
        cur = np.vstack(self.buf)
        details, drifted = {}, []
        for j, name in enumerate(self.names):
            p = psi(self.ref[:, j], cur[:, j])
            ks = ks_2samp(self.ref[:, j], cur[:, j])
            details[name] = {"psi": p, "ks_p": float(ks.pvalue)}
            if p > self.psi_threshold and ks.pvalue < self.ks_alpha:
                drifted.append(name)
        return {"drifted": len(drifted) >= self.min_features,
                "features": drifted, "details": details}


class PageHinkley:
    """Detects a sustained upward shift in a stream (e.g. the anomaly score)."""

    def __init__(self, delta=0.005, lam=50.0, alpha=0.999):
        self.delta, self.lam, self.alpha = delta, lam, alpha
        self.reset()

    def reset(self):
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0

    def update(self, x):
        """Returns True when a change is detected (and resets)."""
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum = self.alpha * self.cum + (x - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        if self.cum - self.min_cum > self.lam:
            self.reset()
            return True
        return False
