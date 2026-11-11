"""
ims_rul.py -- run-to-failure pipeline for the NASA IMS bearing data (and any
dataset of periodic snapshots): ONE feature vector per snapshot against real
elapsed time, then anomaly score -> health indicator -> remaining useful life.

Why not reuse the windowed pipeline: IMS snapshots are 1 s long, 10 min apart,
so stitching them into one signal (ims_loader.load_ims_run) makes windows that
straddle gaps. Here every snapshot is its own window.

STATUS: tested on generated files in the exact IMS text format (parsing,
features, timestamps, the RUL maths). NOT run on the real IMS download (not
reachable from the dev sandbox). Real-data accuracy is therefore unknown; the
RUL interval calibration in rul.py was done on synthetic exponential trends.

    python3 ims_rul.py /path/to/IMS/2nd_test 0       # bearing 1 channel = failed bearing in test 2

IMS facts used (Rexnord ZA-2115 bearings, 2000 rpm, 20 kHz, 20480 samples/snapshot):
BPFO 236.4 Hz, BPFI 296.9 Hz, BSF 139.9 Hz -> ratios 7.092 / 8.907 / 4.197 of shaft rate.
Check the dataset's Readme for which channel belongs to the bearing that failed.
"""
import glob
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import extract_single_window_features

IMS_SHAFT_HZ = 2000.0 / 60.0
IMS_RATIOS = {"bpfo": 7.092, "bpfi": 8.907, "bsf": 4.197}


def snapshot_features(test_dir, channel=0, sample_rate=20000, limit=None):
    """Returns (elapsed_seconds[n], features[n, 23]) -- 14 base + 9 advanced."""
    files = sorted(f for f in glob.glob(os.path.join(test_dir, "*")) if os.path.isfile(f))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no snapshot files in {test_dir}")
    stamps = [datetime.strptime(os.path.basename(f), "%Y.%m.%d.%H.%M.%S") for f in files]
    t0 = stamps[0]
    times, rows = [], []
    for f, ts in zip(files, stamps):
        data = np.loadtxt(f)
        x = (data[:, channel] if data.ndim > 1 else data).astype(np.float64)
        x = x - x.mean()
        rows.append(extract_single_window_features(x, sample_rate, 0.0, extended=True, advanced=True,
                                                   shaft_hz=IMS_SHAFT_HZ, bearing_ratios=IMS_RATIOS))
        times.append((ts - t0).total_seconds())
    return np.array(times), np.vstack(rows)


def snapshot_health(times, X, train_frac=0.25, seq_len=4, epochs=60, seed=0, model="cnn"):
    """Train on the first `train_frac` of snapshots (assumed healthy) and return
    (t_seq, hi, threshold_hi) where hi = sqrt(reconstruction error): linear in the
    deviation from healthy, so exponentially growing damage gives a log-linear
    trend (raw error grows with the SQUARE of the deviation).
    The autoencoder is deliberately UNCLIPPED here: with the +/-5 clipping used
    for alarms the score saturates (observed at ~30x threshold) and stops
    carrying degradation information, which breaks RUL extrapolation."""
    import tensorflow as tf
    from models import MODELS
    from lstm_autoencoder import make_sequences
    tf.keras.utils.set_random_seed(seed)
    n_tr = int(len(X) * train_frac)
    ae = MODELS[model](seq_len, X.shape[1], threshold_method="percentile", clip=None)
    ae.fit(make_sequences(X[:n_tr], seq_len), epochs=epochs, batch_size=8)
    err = ae.reconstruction_error(make_sequences(X, seq_len))
    return times[seq_len - 1:], np.sqrt(err), float(np.sqrt(ae.threshold_))


def learn_failure_hi(times, X, last_k=3, **kw):
    """Failure level = median health indicator (in alarm-threshold units) over the
    last `last_k` snapshots of a COMPLETED run-to-failure run. Learn it from
    runs OTHER than the one you are predicting."""
    _, hi, thr = snapshot_health(times, X, **kw)
    return float(np.median(hi[-last_k:]) / thr)


def rul_from_snapshots(times, X, failure_hi, checkpoints=(0.6, 0.75, 0.9), **kw):
    """Estimate RUL at fractions of the run; truth = (end of run) - now."""
    from rul import RULEstimator
    t_seq, hi, thr = snapshot_health(times, X, **kw)
    est = RULEstimator(failure_hi=failure_hi)
    out = []
    for frac in checkpoints:
        k = int(len(hi) * frac)
        r = est.estimate(t_seq[:k], hi[:k], thr)
        truth = float(times[-1] - t_seq[k - 1])
        err = None if r["rul_s"] is None else (r["rul_s"] - truth) / max(truth, 1.0)
        out.append({"fraction": frac, "status": r["status"], "rul_s": r["rul_s"],
                    "interval_s": (r["rul_low_s"], r["rul_high_s"]), "true_rul_s": truth,
                    "rel_error": err})
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    # usage: ims_rul.py <test_dir_to_predict> <channel> <completed_run_dir_for_failure_level> <channel>
    t, X = snapshot_features(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    if len(sys.argv) < 5:
        sys.exit("need a SECOND, completed run to learn the failure level from (no hindsight): "
                 "ims_rul.py <run> <ch> <other_run> <other_ch>")
    t2, X2 = snapshot_features(sys.argv[3], int(sys.argv[4]))
    fh = learn_failure_hi(t2, X2)
    print(f"{len(t)} snapshots over {t[-1]/3600:.1f} h; failure level learned elsewhere = {fh:.1f}x threshold")
    for r in rul_from_snapshots(t, X, fh):
        print(r)
