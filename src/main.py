"""
main.py
--------
End-to-end pipeline:
  1. Generate synthetic bearing run (healthy -> degrading -> failed)
  2. Extract time+frequency domain features per window
  3. Train LSTM Autoencoder on healthy-only data; calibrate threshold
  4. Train 3 baselines on the same healthy-only data
  5. Score the FULL run with all 4 methods
  6. Benchmark: false positive rate + detection lag per method
  7. Save comparison plot + results table

Run: python3 main.py
Outputs land in ../outputs/
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_data import generate_run
from pdm_config import SAMPLE_RATE_HZ, WINDOW_S, WINDOW_OVERLAP
from feature_engineering import extract_windowed_features, FEATURE_NAMES
from lstm_autoencoder import LSTMAutoencoder, make_sequences
from baselines import IsolationForestDetector, OneClassSVMDetector, StatisticalThresholdDetector
from evaluate import run_benchmark, plot_comparison, print_table

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 0
SEQ_LEN = 15  # LSTM sequence length (in windows); 15 x 1 s x (1-0.5 overlap) ≈ 7.5 s of context


def main():
    print("=" * 70)
    print("STEP 1: Generating synthetic bearing degradation run")
    print("=" * 70)
    run = generate_run(duration_s=3600, sample_rate=SAMPLE_RATE_HZ, seed=7)
    print(f"  Duration: {run['time'][-1]:.0f}s, sample rate: {run['sample_rate']} Hz")

    print("\n" + "=" * 70)
    print("STEP 2: Feature engineering (time-domain + FFT frequency-domain)")
    print("=" * 70)
    X, times, labels = extract_windowed_features(run, window_s=WINDOW_S, overlap=WINDOW_OVERLAP)
    print(f"  Feature matrix: {X.shape[0]} windows x {X.shape[1]} features")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"  Label split -> healthy: {(labels==0).sum()}, degrading: {(labels==1).sum()}, failed: {(labels==2).sum()}")

    X_healthy = X[labels == 0]
    split = int(len(X_healthy) * 0.8)
    X_train_flat, X_val_flat = X_healthy[:split], X_healthy[split:]

    print("\n" + "=" * 70)
    print("STEP 3: Training LSTM Autoencoder (healthy data ONLY)")
    print("=" * 70)
    sequences_all = make_sequences(X, seq_len=SEQ_LEN)
    seq_labels = np.array([
        int(np.round(np.mean(labels[i:i + SEQ_LEN]))) for i in range(0, len(labels) - SEQ_LEN + 1)
    ])
    seq_times = np.array([
        times[i + SEQ_LEN - 1] for i in range(0, len(times) - SEQ_LEN + 1)
    ])  # label/timestamp = end of sequence (point of detection)

    healthy_seq_mask = seq_labels == 0
    healthy_seqs = sequences_all[healthy_seq_mask]
    split_seq = int(len(healthy_seqs) * 0.8)
    train_seqs, val_seqs = healthy_seqs[:split_seq], healthy_seqs[split_seq:]

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)   # reproducible training (results otherwise vary run to run)
    ae = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=X.shape[1])
    ae.fit(train_seqs, epochs=40, verbose=0)
    print(f"  Calibrated threshold ({ae.threshold_method}, healthy validation block "
          f"separated from training by a {SEQ_LEN}-sequence gap): {ae.threshold_:.5f}")

    ae_scores, ae_flags = ae.predict_anomaly(sequences_all)

    print("\n" + "=" * 70)
    print("STEP 4: Training baseline detectors (same healthy-only data)")
    print("=" * 70)
    iforest = IsolationForestDetector().fit(X_train_flat)
    ocsvm = OneClassSVMDetector().fit(X_train_flat)
    stat = StatisticalThresholdDetector().fit(X_train_flat)
    print("  Trained: Isolation Forest, One-Class SVM, Statistical Threshold")

    # Align baseline (per-window) scores to the same timestamps as the LSTM
    # (per-sequence, timestamped at sequence end) for a fair, same-x-axis comparison.
    align_idx = np.searchsorted(times, seq_times)
    align_idx = np.clip(align_idx, 0, len(X) - 1)
    X_aligned = X[align_idx]
    labels_aligned = labels[align_idx]

    results = {
        "LSTM Autoencoder": (ae_scores, ae_flags),
        "Isolation Forest": (iforest.score(X_aligned), iforest.predict_anomaly(X_aligned)),
        "One-Class SVM": (ocsvm.score(X_aligned), ocsvm.predict_anomaly(X_aligned)),
        "Statistical Threshold": (stat.score(X_aligned), stat.predict_anomaly(X_aligned)),
    }

    print("\n" + "=" * 70)
    print("STEP 5: Benchmarking (false positive rate + detection lag)")
    print("=" * 70)
    rows = run_benchmark(results, seq_times, labels_aligned)
    print_table(rows)

    with open(os.path.join(OUT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 70)
    print("STEP 6: Saving comparison plot")
    print("=" * 70)
    plot_path = plot_comparison(seq_times, results, labels_aligned,
                                 os.path.join(OUT_DIR, "anomaly_comparison.png"))
    print(f"  Saved: {plot_path}")

    ae.model.save(os.path.join(OUT_DIR, "lstm_autoencoder.keras"))
    ae.save_calibration(os.path.join(OUT_DIR, "threshold_and_norm.npz"))
    print(f"  Saved trained model + calibration to {OUT_DIR}/")

    # Reference data for the live bridge: healthy feature distribution (drift
    # monitoring) and the health-indicator level at failure (RUL estimation).
    np.save(os.path.join(OUT_DIR, "healthy_reference_features.npy"), X_train_flat.astype(np.float32))
    failed = seq_labels == 2
    failure_hi = float(np.median(ae_scores[failed]) / ae.threshold_) if failed.any() else 8.0
    with open(os.path.join(OUT_DIR, "rul_config.json"), "w") as f:
        json.dump({"failure_hi": failure_hi,
                   "note": "median score/threshold during the SIMULATED failed state; "
                           "replace with a value learned from your own run-to-failure data"}, f, indent=2)
    print(f"  Saved drift reference ({len(X_train_flat)} healthy windows) and RUL failure level "
          f"(HI at failure = {failure_hi:.1f}x threshold)")
    from tracking import log_run
    log_run("train-synthetic",
            params={"sample_rate_hz": SAMPLE_RATE_HZ, "window_s": WINDOW_S, "seq_len": SEQ_LEN,
                    "n_features": X.shape[1], "scaling": ae.scaling, "clip": ae.clip,
                    "threshold_method": ae.threshold_method},
            metrics={"threshold": ae.threshold_, "failure_hi": failure_hi,
                     **{f"{r['method'].split()[0].lower()}_fpr_pct": r["false_positive_rate_%"] for r in rows
                        if "false_positive_rate_%" in r}},
            artifacts=[os.path.join(OUT_DIR, "benchmark_results.json"),
                       os.path.join(OUT_DIR, "anomaly_comparison.png")])
    try:
        from export_onnx import export
        export()
    except Exception as e:          # onnx tooling is optional for training
        print(f"  (ONNX export skipped: {e})")

    print("\nDone. See outputs/anomaly_comparison.png and outputs/benchmark_results.json")


if __name__ == "__main__":
    main()
