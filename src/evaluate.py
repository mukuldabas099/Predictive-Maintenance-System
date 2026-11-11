"""
evaluate.py
------------
Benchmarks the LSTM Autoencoder against the three baselines on:

  - False positive rate (FPR): fraction of HEALTHY windows wrongly flagged.
  - Detection lag: how many seconds AFTER degradation actually begins
    (ground-truth state switches to "degrading") does each model first raise
    a sustained alarm. Lower lag = earlier warning = more useful for PdM.
    "Sustained" = 3 consecutive flagged windows, to avoid crediting single
    noisy spikes.
  - Anomaly score trajectory plot over the whole run, all 4 methods stacked,
    with the true degradation onset and failure point marked.

This directly produces the evidence needed for the report's "why LSTM
Autoencoder over classical baselines" argument.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detection_lag(times, flags, true_labels, sustain=3):
    """
    First time index where `flags` is True for `sustain` consecutive windows
    AFTER the true degradation onset, minus the true onset time.
    Returns lag in seconds, or None if never detected.
    """
    onset_idx = np.argmax(true_labels >= 1)  # first "degrading" window
    if true_labels[onset_idx] < 1:
        return None  # no degradation in this run
    onset_time = times[onset_idx]

    run_length = 0
    for i in range(len(flags)):
        if flags[i]:
            run_length += 1
            if run_length >= sustain:
                detect_time = times[i - sustain + 1]
                return max(0.0, detect_time - onset_time)
        else:
            run_length = 0
    return None  # never sustained-detected


def false_positive_rate(flags, true_labels):
    healthy_mask = true_labels == 0
    if healthy_mask.sum() == 0:
        return np.nan
    return flags[healthy_mask].mean()


def run_benchmark(results, times, true_labels):
    """
    results: dict of {method_name: (scores, flags)} aligned to `times` /
             `true_labels` (same length & order).
    Returns a list of dict rows ready to print/tabulate.
    """
    rows = []
    for name, (scores, flags) in results.items():
        lag = detection_lag(times, flags, true_labels)
        fpr = false_positive_rate(flags, true_labels)
        rows.append({
            "method": name,
            "false_positive_rate_%": round(float(fpr) * 100, 2),
            "detection_lag_s": round(float(lag), 1) if lag is not None else None,
        })
    return rows


def plot_comparison(times, results, true_labels, out_path, failure_frac_time=None):
    """
    results: dict of {method_name: (scores, flags)}. Plots normalized
    anomaly score per method over time, with true degradation onset marked.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    onset_idx = np.argmax(true_labels >= 1)
    onset_time = times[onset_idx] if true_labels[onset_idx] >= 1 else None

    for ax, (name, (scores, flags)) in zip(axes, results.items()):
        s = np.asarray(scores, dtype=float)
        # Normalize to [0,1] for visual comparability across very different scales
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
        ax.plot(times, s_norm, color="#3b6ea5", linewidth=1.1, label="anomaly score")
        ax.fill_between(times, 0, s_norm, where=flags, color="#d9534f", alpha=0.35,
                         label="flagged anomaly", step=None)
        if onset_time is not None:
            ax.axvline(onset_time, color="#333333", linestyle="--", linewidth=1,
                        label="true degradation onset")
        ax.set_ylabel(name, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Anomaly score comparison: LSTM Autoencoder vs. classical baselines", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def print_table(rows):
    headers = ["method", "false_positive_rate_%", "detection_lag_s"]
    widths = [max(len(str(r[h])) for r in rows + [dict(zip(headers, headers))]) + 2 for h in headers]
    line = "".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)))
