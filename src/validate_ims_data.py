"""
validate_ims_data.py
-----------------------
*** WRITTEN AND READY BUT UNTESTED -- see README's "NASA IMS validation"
section for why (network access to the dataset isn't available in the
sandbox this was built in), and for exactly what you need to do to run it. ***

Computes the ONE metric CWRU couldn't: detection lag on a real continuous
run-to-failure trace. Re-uses every other piece of this project unchanged
(feature_engineering, lstm_autoencoder, baselines, evaluate) -- only
ims_loader.py is new, precisely so this file could be this short and so
you're benchmarking the SAME pipeline as the rest of the report, not a
different one.

Usage (after downloading and unzipping the dataset -- see README):
    python3 validate_ims_data.py /path/to/IMS/1st_test/1st_test --channel 0
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ims_loader import load_ims_run, estimate_degradation_onset
from feature_engineering import extract_windowed_features
from lstm_autoencoder import LSTMAutoencoder, make_sequences
from baselines import IsolationForestDetector, OneClassSVMDetector, StatisticalThresholdDetector
from evaluate import detection_lag, false_positive_rate, print_table

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 15
WINDOW_S = 1.024  # = 20480 samples @ 20 kHz = EXACTLY one IMS snapshot. Must not be 1.0 s:
                  # a 20000-sample window drifts out of alignment and ~every window then
                  # straddles a seam between two snapshots taken 10 minutes apart.
OVERLAP = 0.0    # can't overlap: there's a 10-minute gap between snapshots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_dir", help="Path to e.g. IMS/1st_test/1st_test/")
    ap.add_argument("--channel", type=int, default=0,
                     help="Accelerometer column for the bearing documented to fail "
                          "in this test -- check IMS's Readme Document.txt")
    ap.add_argument("--healthy-frac", type=float, default=0.3,
                     help="Fraction of the run (from the start) to treat as healthy "
                          "training data. IMS gives no ground-truth onset label, so "
                          "this is a conservative manual assumption -- cross-check it "
                          "against estimate_degradation_onset()'s output below before "
                          "trusting the detection-lag number.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Loading real IMS run-to-failure data from {args.test_dir}")
    print("=" * 70)
    run = load_ims_run(args.test_dir, bearing_channel=args.channel)
    print(f"  {len(run['vibration'])} samples spanning {run['time'][-1]/3600:.1f} hours")

    onset_time, rms, snapshot_times = estimate_degradation_onset(run)
    if onset_time is not None:
        print(f"  RMS-based estimated degradation onset: {onset_time/3600:.2f}h "
              f"({onset_time/run['time'][-1]*100:.1f}% through the run)")
    else:
        print("  WARNING: RMS proxy found no clear onset -- double check --channel "
              "points at the bearing that actually failed in this test.")

    print("\n" + "=" * 70)
    print("Extracting features + building ground-truth labels from the RMS-onset proxy")
    print("=" * 70)
    X, times, _ = extract_windowed_features(run, window_s=WINDOW_S, overlap=OVERLAP)

    # Ground truth for detection-lag scoring: healthy until the RMS-proxy onset
    # (or the manual --healthy-frac cutoff, whichever is more conservative/earlier).
    manual_cutoff_time = run["time"][-1] * args.healthy_frac
    effective_onset = min(onset_time, manual_cutoff_time) if onset_time is not None else manual_cutoff_time
    labels = (times >= effective_onset).astype(int)  # 0=healthy, 1=degrading+ (IMS has no 3-way split)
    print(f"  Using onset={effective_onset/3600:.2f}h for ground truth "
          f"({(labels==0).sum()} healthy windows, {(labels==1).sum()} post-onset windows)")

    X_healthy = X[labels == 0]
    split = int(len(X_healthy) * 0.8)
    X_train, X_val = X_healthy[:split], X_healthy[split:]

    print("\n" + "=" * 70)
    print("Training LSTM Autoencoder + baselines on real healthy IMS data")
    print("=" * 70)
    seqs_all = make_sequences(X, seq_len=SEQ_LEN)
    seq_labels = np.array([int(np.round(np.mean(labels[i:i+SEQ_LEN])))
                            for i in range(0, len(labels) - SEQ_LEN + 1)])
    seq_times = np.array([times[i + SEQ_LEN - 1] for i in range(0, len(times) - SEQ_LEN + 1)])
    healthy_seqs = seqs_all[seq_labels == 0]
    split_seq = int(len(healthy_seqs) * 0.8)
    if split_seq < 10:
        print(f"\nERROR: only {len(healthy_seqs)} healthy sequences available "
              f"(need at least ~10-15 to train anything meaningful with "
              f"seq_len={SEQ_LEN}). This usually means --healthy-frac is too small, "
              f"or the RMS-based onset estimate fired too early (check the "
              f"'RMS-based estimated degradation onset' line above against the "
              f"actual run duration -- if it looks implausibly early, try a "
              f"different --channel, since not every column corresponds to the "
              f"bearing that actually failed in this test).")
        sys.exit(1)
    ae = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=X.shape[1])
    ae.fit(healthy_seqs[:split_seq], epochs=40, verbose=0)
    ae_scores, ae_flags = ae.predict_anomaly(seqs_all)
    print(f"  LSTM threshold: {ae.threshold_:.5f}")

    iforest = IsolationForestDetector().fit(X_train)
    ocsvm = OneClassSVMDetector().fit(X_train)
    stat = StatisticalThresholdDetector().fit(X_train)

    align_idx = np.clip(np.searchsorted(times, seq_times), 0, len(X) - 1)
    labels_aligned = labels[align_idx]
    X_aligned = X[align_idx]

    results = {
        "LSTM Autoencoder": (ae_scores, ae_flags),
        "Isolation Forest": (iforest.score(X_aligned), iforest.predict_anomaly(X_aligned)),
        "One-Class SVM": (ocsvm.score(X_aligned), ocsvm.predict_anomaly(X_aligned)),
        "Statistical Threshold": (stat.score(X_aligned), stat.predict_anomaly(X_aligned)),
    }

    print("\n" + "=" * 70)
    print("Detection lag + false positive rate on REAL run-to-failure data")
    print("=" * 70)
    rows = []
    for name, (scores, flags) in results.items():
        lag = detection_lag(seq_times, flags, labels_aligned)
        fpr = false_positive_rate(flags, labels_aligned)
        rows.append({
            "method": name,
            "false_positive_rate_%": round(float(fpr) * 100, 2),
            "detection_lag_hours": round(float(lag) / 3600, 2) if lag is not None else None,
        })
    for r in rows:
        print(f"  {r['method']:24s} FPR={r['false_positive_rate_%']:5.1f}%  "
              f"lag={r['detection_lag_hours']}h")

    with open(os.path.join(OUT_DIR, "ims_benchmark_results.json"), "w") as f:
        json.dump({
            "results": rows,
            "estimated_onset_hours": onset_time / 3600 if onset_time is not None else None,
            "note": (
                "Ground truth onset is an RMS-based statistical PROXY, not a "
                "documented IMS label (IMS only documents which bearing failed "
                "by the end of the test). Report detection lag with that caveat."
            ),
        }, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/ims_benchmark_results.json")


if __name__ == "__main__":
    main()
