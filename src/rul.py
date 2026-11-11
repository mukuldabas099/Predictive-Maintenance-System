"""
rul.py -- Remaining Useful Life (time to failure) estimation.

Anomaly detection says "something is wrong"; maintenance planning needs "how
long do I have?". This module turns the anomaly score into a HEALTH INDICATOR
(HI = EWMA-smoothed score / alarm threshold, so HI = 1 means "alarm") and
extrapolates its recent trend to a failure level.

Model: HI grows roughly exponentially once a fault is established
(log-linear fit, robust to noise via a Theil-Sen slope), which matches the
accelerating degradation seen in run-to-failure bearing data (IMS, FEMTO).

    RULEstimator(failure_hi=...).estimate(times_s, scores, threshold)
        -> {"rul_s", "rul_low_s", "rul_high_s", "hi_now", "growth_per_s", "status"}

The failure level `failure_hi` should be learned from run-to-failure data
(median HI at the moment of failure); the default is a placeholder.
status: "healthy" (HI < 1), "degrading" (HI >= 1, trend reliable),
        "uncertain" (too few points / flat trend), "imminent" (HI >= failure_hi).
The interval is an 80% moving-block-bootstrap interval of the log-linear fit
(it accounts for noise in both the slope and the current health level). The raw
bootstrap interval was too narrow (~58% coverage for a nominal 80%), so it is
widened by `interval_scale`=1.7, which was calibrated on SYNTHETIC exponential
degradation (80% +/- 3% there, noise 5-30%). Real machines will not follow an
exponential exactly: treat the interval as indicative until you have calibrated
it on run-to-failure data of your own equipment.
"""
import numpy as np
from scipy.stats import theilslopes

from calibration import ewma


class RULEstimator:
    def __init__(self, failure_hi=8.0, alpha=0.3, min_points=10, fit_points=60,
                 n_boot=200, block=10, seed=0, interval_scale=1.7):
        self.failure_hi, self.alpha = failure_hi, alpha
        self.min_points, self.fit_points = min_points, fit_points
        self.n_boot, self.block, self.seed = n_boot, block, seed
        self.interval_scale = interval_scale

    def health_indicator(self, scores, threshold):
        return ewma(np.asarray(scores, dtype=float) / threshold, self.alpha)

    def estimate(self, times_s, scores, threshold):
        t = np.asarray(times_s, dtype=float)
        hi = self.health_indicator(scores, threshold)
        hi_now = float(hi[-1])
        out = {"hi_now": hi_now, "rul_s": None, "rul_low_s": None, "rul_high_s": None,
               "growth_per_s": None}
        if hi_now >= self.failure_hi:
            return {**out, "rul_s": 0.0, "rul_low_s": 0.0, "rul_high_s": 0.0, "status": "imminent"}
        if hi_now < 1.0:
            return {**out, "status": "healthy"}
        # fit only the recent part of the run that is above the alarm level
        above = np.where(hi >= 1.0)[0]
        idx = above[-self.fit_points:]
        if len(idx) < self.min_points:
            return {**out, "status": "uncertain"}
        tt, y = t[idx], np.log(hi[idx])
        slope, intercept, _, _ = theilslopes(y, tt)
        out["growth_per_s"] = float(slope)
        if slope <= 1e-9:
            return {**out, "status": "uncertain"}
        t_now = tt[-1]
        log_fail = np.log(self.failure_hi)

        def rul_from(sl, ic):
            level_now = ic + sl * t_now          # fitted (de-noised) level, not the raw last point
            return (log_fail - level_now) / sl if sl > 1e-9 else np.inf

        rul = rul_from(slope, intercept)
        resid = y - (intercept + slope * tt)
        rng = np.random.default_rng(self.seed)
        n, b = len(y), min(self.block, len(y))
        boots = []
        for _ in range(self.n_boot):
            idxs = np.concatenate([np.arange(st, st + b) for st in
                                   rng.integers(0, n - b + 1, size=int(np.ceil(n / b)))])[:n]
            yb = intercept + slope * tt + resid[idxs]
            sl_b, ic_b, _, _ = theilslopes(yb, tt)
            boots.append(rul_from(sl_b, ic_b))
        lo, hi_ = np.percentile(boots, [10, 90])
        k = self.interval_scale
        low = rul - k * (rul - min(lo, rul))
        high = rul + k * (max(hi_, rul) - rul)
        return {**out, "rul_s": float(max(rul, 0.0)), "rul_low_s": float(max(low, 0.0)),
                "rul_high_s": float(high), "status": "degrading"}
