"""
validate_real_data.py
------------------------
Re-runs the same trained pipeline (LSTM Autoencoder + 3 baselines) on REAL
Case Western Reserve University bearing data, replacing the synthetic
sanity-check numbers with numbers you can actually cite.

Because CWRU is a set of short FIXED-CONDITION recordings (not a continuous
run-to-failure trace), "detection lag" doesn't apply here. Instead this
benchmarks:
  - False positive rate on held-out real healthy data
  - Sensitivity (recall): % of windows correctly flagged anomalous, PER REAL
    FAULT TYPE (Inner Race / Ball / Outer Race) and PER SEVERITY (0.007" /
    0.014" / 0.021" defect diameter) -- i.e. does the model generalize across
    different real fault mechanisms and different fault sizes, especially the
    smallest (hardest-to-detect, earliest-warning) 0.007" defects.

Run: python3 validate_real_data.py
(assumes ../../cwru_data/ contains the cloned CWRU dataset -- see README)
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwru_loader import concat_runs, load_cwru_run
from feature_engineering import extract_windowed_features
from lstm_autoencoder import LSTMAutoencoder, make_sequences
from baselines import IsolationForestDetector, OneClassSVMDetector, StatisticalThresholdDetector

CWRU_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cwru_data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = 48000
WINDOW_S = 0.25      # shorter windows: CWRU recordings are only ~5-10s each
OVERLAP = 0.5
SEQ_LEN = 8            # shorter sequences to match limited real-data volume
                        # (some CWRU fault files are as short as ~1.3s, i.e. ~9 windows)


def find_fault_files():
    """13 real fault files at 0 HP load: 3 fault types x 3 severities
    (+ extra outer-race clock-position variants), all from the 48kHz drive-end set."""
    base = os.path.join(CWRU_DIR, "48k_Drive_End_Bearing_Fault_Data")
    pattern_map = {
        "Inner Race / 0.007in": "IR/007/109_0.mat",
        "Inner Race / 0.014in": "IR/014/174_0.mat",
        "Inner Race / 0.021in": "IR/021/213_0.mat",
        "Ball / 0.007in": "B/007/122_0.mat",
        "Ball / 0.014in": "B/014/189_0.mat",
        "Ball / 0.021in": "B/021/226_0.mat",
        "Outer Race / 0.007in": "OR/007/@6/135_0.mat",
        "Outer Race / 0.014in": "OR/014/201@6_0.mat",
        "Outer Race / 0.021in": "OR/021/@6/238_0.mat",
    }
    return {name: os.path.join(base, rel) for name, rel in pattern_map.items()
            if os.path.exists(os.path.join(base, rel))}


def main():
    print("=" * 70)
    print("STEP 1: Loading real CWRU healthy baseline data")
    print("=" * 70)
    normal_files = sorted(glob.glob(os.path.join(CWRU_DIR, "Normal", "*.mat")))
    if not normal_files:
        print(f"ERROR: no CWRU data found under {CWRU_DIR}. "
              f"Clone it first: git clone https://github.com/s-whynot/CWRU-dataset.git")
        return
    print(f"  Found {len(normal_files)} healthy baseline files (different motor loads)")

    healthy_run = concat_runs(normal_files, sample_rate=SAMPLE_RATE)
    print(f"  Concatenated healthy signal: {healthy_run['time'][-1]:.1f}s @ {SAMPLE_RATE} Hz")

    X_h, t_h, _ = extract_windowed_features(healthy_run, window_s=WINDOW_S, overlap=OVERLAP)
    print(f"  Extracted {len(X_h)} healthy feature windows")

    split = int(len(X_h) * 0.7)
    X_train, X_val = X_h[:split], X_h[split:]
    print(f"  Train: {len(X_train)} windows, Validation (for FPR + threshold): {len(X_val)} windows")

    print("\n" + "=" * 70)
    print("STEP 2: Training LSTM Autoencoder on real healthy data only")
    print("=" * 70)
    seqs_train = make_sequences(X_train, seq_len=SEQ_LEN)
    seqs_val = make_sequences(X_val, seq_len=SEQ_LEN)
    print(f"  {len(seqs_train)} training sequences, {len(seqs_val)} validation sequences")

    if len(seqs_train) < 10:
        print("  ERROR: not enough real healthy data to form training sequences with "
              "this seq_len/window_s. Reduce SEQ_LEN or WINDOW_S.")
        return

    ae = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=X_h.shape[1])
    ae.fit(seqs_train, epochs=60, batch_size=8, val_split=0.2, verbose=0)
    print(f"  Calibrated threshold: {ae.threshold_:.5f}")

    print("\n" + "=" * 70)
    print("STEP 3: Training baselines on real healthy data only")
    print("=" * 70)
    iforest = IsolationForestDetector(contamination=0.05).fit(X_train)
    ocsvm = OneClassSVMDetector(nu=0.05).fit(X_train)
    stat = StatisticalThresholdDetector().fit(X_train)
    print("  Trained: Isolation Forest, One-Class SVM, Statistical Threshold")

    detectors = {
        "LSTM Autoencoder": ("sequence", ae),
        "Isolation Forest": ("window", iforest),
        "One-Class SVM": ("window", ocsvm),
        "Statistical Threshold": ("window", stat),
    }

    print("\n" + "=" * 70)
    print("STEP 4: False positive rate on held-out REAL healthy data")
    print("=" * 70)
    fpr_results = {}
    for name, (kind, det) in detectors.items():
        if kind == "sequence":
            _, flags = det.predict_anomaly(seqs_val)
        else:
            flags = det.predict_anomaly(X_val)
        fpr_results[name] = float(flags.mean())
        print(f"  {name:22s}: {fpr_results[name]*100:5.1f}% false positive rate "
              f"(n={len(flags)} healthy windows)")

    print("\n" + "=" * 70)
    print("STEP 5: Sensitivity per REAL fault type & severity")
    print("=" * 70)
    fault_files = find_fault_files()
    print(f"  Testing against {len(fault_files)} real fault recordings")

    sensitivity_results = {name: {} for name in detectors}
    for fault_name, path in fault_files.items():
        run = load_cwru_run(path, sample_rate=SAMPLE_RATE, state=2)
        X_f, t_f, _ = extract_windowed_features(run, window_s=WINDOW_S, overlap=OVERLAP)
        if len(X_f) == 0:
            continue
        row = f"  {fault_name:24s}"
        for name, (kind, det) in detectors.items():
            if kind == "sequence":
                if len(X_f) < SEQ_LEN:
                    continue  # too short to form even one full sequence at this seq_len
                seqs_f = make_sequences(X_f, seq_len=SEQ_LEN)
                _, flags = det.predict_anomaly(seqs_f)
            else:
                flags = det.predict_anomaly(X_f)
            sensitivity = float(flags.mean())
            sensitivity_results[name][fault_name] = sensitivity
            row += f" | {name.split()[0]}:{sensitivity*100:5.1f}%"
        print(row)

    print("\n" + "=" * 70)
    print("STEP 6: Saving results")
    print("=" * 70)
    summary = {
        "false_positive_rate": fpr_results,
        "sensitivity_by_fault": sensitivity_results,
        "notes": (
            "Trained on real CWRU healthy baseline data only (4 files, ~35s "
            "concatenated, 4 different motor loads). Tested against 9 real "
            "fault recordings spanning 3 fault mechanisms (Inner Race, Ball, "
            "Outer Race) x 3 defect severities (0.007in/0.014in/0.021in). "
            "No temperature channel in CWRU -> that feature was zero-filled. "
            "Small healthy training set (real dataset limitation, not a "
            "pipeline limitation) -- more baseline recordings would improve "
            "threshold stability."
        ),
    }
    with open(os.path.join(OUT_DIR, "real_data_benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUT_DIR}/real_data_benchmark.json")

    # Bar chart: sensitivity by fault severity, grouped by method
    fault_names = list(fault_files.keys())
    methods = list(detectors.keys())
    x = np.arange(len(fault_names))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(methods):
        vals = [sensitivity_results[name].get(f, 0) * 100 for f in fault_names]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(fault_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Sensitivity (% windows flagged anomalous)")
    ax.set_title("Real CWRU data: detection sensitivity by fault type & severity")
    ax.axhline(90, color="gray", linestyle=":", linewidth=1, label="90% reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "real_data_sensitivity.png"), dpi=140)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/real_data_sensitivity.png")


if __name__ == "__main__":
    main()
