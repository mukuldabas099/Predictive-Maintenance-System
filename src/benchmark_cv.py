"""
benchmark_cv.py
------------------
Stricter real-data evaluation on CWRU than validate_real_data.py, plus an
ABLATION that adds each phase-2 improvement one at a time so you can see which
change actually helped (and which didn't).

Protocol (leave-one-LOAD-out cross-validation)
  CWRU has 4 healthy recordings, one per motor load (0-3 HP). For each fold one
  load is held out entirely:
    - train/threshold-calibration data = the other 3 healthy recordings
      (per file: first 70% of windows = train, then a gap, then the last part =
       validation block used for early stopping + threshold)
    - false-positive rate = flags on the held-out healthy load the model has
      NEVER seen (tests generalisation across operating conditions, which is
      what a real deployment faces)
    - sensitivity = flags on the 9 real fault recordings (IR/B/OR x 3 sizes)
    - AUROC / balanced PR-AUC = threshold-free separation of held-out healthy
      vs. fault windows
  Sequences are built per file (never across file boundaries).

Ablation configs (LSTM Autoencoder):
  A  legacy          basic 10 features, z-score, mean+3*std, validation tail
                     NOT separated from training (original behaviour)
  B  + gap split     same, but validation block separated by a gap
  C  + robust/pctl   robust scaling + clipping, 99.5th-percentile threshold
  D  + envelope      adds 4 envelope-analysis features (14 total)
  F  multi-condition: nested leave-one-condition-out errors pooled to set the
     threshold (LSTMAutoencoder.fit_multicondition) -- needs 3 train loads
  E  + cross-load    D, but early-stopping + threshold use a WHOLE different
     calibration      load (train on 2 loads, calibrate on the 3rd) so the
                      threshold reflects operating-condition shift, not just
                      within-recording noise
Every config is also reported with a 3-of-5 persistence filter on the flags.

Run:  python3 benchmark_cv.py            (expects ../../cwru_data, see README)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from cwru_loader import load_cwru_run
from feature_engineering import extract_windowed_features, BASIC_FEATURE_NAMES
from lstm_autoencoder import LSTMAutoencoder, make_sequences
from models import MODELS
from augment import augment_run
from baselines import (IsolationForestDetector, OneClassSVMDetector,
                       StatisticalThresholdDetector)
from calibration import persistent_flags
from validate_real_data import find_fault_files

CWRU_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cwru_data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
SR = 48000
WINDOW_S, OVERLAP, SEQ_LEN = 0.125, 0.5, 8
N_BASIC = len(BASIC_FEATURE_NAMES)
TRAIN_FRAC = 0.70

# Column layout of features_for(): DE channel extended(14) | DE advanced(9) | FE channel extended(14)
DE14 = list(range(14))
ADV9 = list(range(14, 23))
FE13 = [23 + i for i in range(14) if i != 9]        # fan-end features, minus its constant temperature column
_ROB = dict(scaling="robust", clip=5.0, threshold_method="percentile")
CONFIGS = {
    "A legacy (basic, z-score, mean+3sd, leaky val)":
        dict(cols=list(range(N_BASIC)), gap=False, kw=dict(scaling="zscore", clip=None, threshold_method="sigma")),
    "B + gap-separated validation":
        dict(cols=list(range(N_BASIC)), gap=True, kw=dict(scaling="zscore", clip=None, threshold_method="sigma")),
    "C + robust scaling & 99.5th pct threshold":
        dict(cols=list(range(N_BASIC)), gap=True, kw=_ROB),
    "D + envelope features (14)":
        dict(cols=DE14, gap=True, kw=_ROB),
    "E + cross-load threshold calibration":
        dict(cols=DE14, gap=True, cross=True, kw=_ROB),
    "F multi-condition (nested) calibration":
        dict(cols=DE14, gap=True, multi=True, kw=_ROB),
    "G F + advanced features (23)":
        dict(cols=DE14 + ADV9, gap=True, multi=True, kw=_ROB),
    "H F + fan-end channel fusion (27)":
        dict(cols=DE14 + FE13, gap=True, multi=True, kw=_ROB),
    "I F + signal augmentation":
        dict(cols=DE14, gap=True, multi=True, aug=3, kw=_ROB),
    "J F with 1D-CNN autoencoder":
        dict(cols=DE14, gap=True, multi=True, model="cnn", kw=_ROB),
    "K F with TCN autoencoder":
        dict(cols=DE14, gap=True, multi=True, model="tcn", kw=_ROB),
}
LABEL = {"lstm": "LSTM-AE", "cnn": "CNN-AE", "tcn": "TCN-AE"}


def shaft_hz_of(path, default_rpm=1797.0):
    import re
    import scipy.io as sio
    m = sio.loadmat(path)
    for k in m:
        if re.search(r"RPM$", k):
            return float(np.ravel(m[k])[0]) / 60.0
    return default_rpm / 60.0


def features_for(path, state):
    """(n_windows, 37): DE extended 14 | DE advanced 9 | FE extended 14."""
    sh = shaft_hz_of(path)
    de = load_cwru_run(path, sample_rate=SR, state=state, channel="DE")
    Xd, _, _ = extract_windowed_features(de, WINDOW_S, OVERLAP, extended=True,
                                         advanced=True, shaft_hz=sh)
    fe = load_cwru_run(path, sample_rate=SR, state=state, channel="FE")
    Xf, _, _ = extract_windowed_features(fe, WINDOW_S, OVERLAP, extended=True)
    n = min(len(Xd), len(Xf))
    return np.hstack([Xd[:n], Xf[:n]])


def aug_features_for(path, k, seed):
    """k augmented copies of a healthy file -> list of (n, 14) DE feature arrays."""
    rng = np.random.default_rng(seed)
    base = load_cwru_run(path, sample_rate=SR, state=0, channel="DE")
    out = []
    for _ in range(k):
        X, _, _ = extract_windowed_features(augment_run(base, rng), WINDOW_S, OVERLAP, extended=True)
        out.append(X)
    return out


def split_file(X):
    """chronological train | gap | validation windows for one healthy file."""
    n = len(X)
    n_tr = int(n * TRAIN_FRAC)
    return X[:n_tr], X[n_tr + SEQ_LEN:]


def seqs(X):
    return make_sequences(X, SEQ_LEN) if len(X) >= SEQ_LEN else np.empty((0, SEQ_LEN, X.shape[1]), np.float32)


def balanced_pr_auc(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    w = np.r_[np.full(len(neg), 0.5 / len(neg)), np.full(len(pos), 0.5 / len(pos))]
    return average_precision_score(y, s, sample_weight=w)


def evaluate(score_fn, flag_fn, healthy_X, fault_Xs, kind):
    """Return dict of metrics. kind='seq' scores sequences, 'win' scores windows."""
    prep = seqs if kind == "seq" else (lambda X: X)
    h = prep(healthy_X)
    h_scores, h_flags = score_fn(h), flag_fn(h)
    f_scores, f_flags = {}, {}
    for name, Xf in fault_Xs.items():
        p = prep(Xf)
        if len(p) == 0:
            continue
        f_scores[name], f_flags[name] = score_fn(p), flag_fn(p)
    all_f = np.concatenate(list(f_scores.values()))
    # threshold-free separation margin (AUROC saturates at 1.0 on CWRU): how many
    # decades the weakest fault's median score sits above the healthy 99.5th pct
    sep = float(np.log10(min(np.median(v) for v in f_scores.values())
                         / (np.percentile(h_scores, 99.5) + 1e-12)))
    sens = {n: float(f.mean()) for n, f in f_flags.items()}
    sens_p = {n: float(persistent_flags(f).mean()) for n, f in f_flags.items()}
    return {
        "fpr": float(h_flags.mean()),
        "fpr_persist": float(persistent_flags(h_flags).mean()),
        "sens_mean": float(np.mean(list(sens.values()))),
        "sens_min": float(np.min(list(sens.values()))),
        "sens_mean_persist": float(np.mean(list(sens_p.values()))),
        "sep_log10": sep,
        "auroc": float(roc_auc_score(np.r_[np.zeros(len(h_scores)), np.ones(len(all_f))],
                                     np.r_[h_scores, all_f])),
        "pr_auc_bal": float(balanced_pr_auc(h_scores, all_f)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated config letters to run, e.g. E")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--out", default="benchmark_cv.json")
    ap.add_argument("--fold", type=int, default=-1, help="run a single held-out load (0-3); -1 = all")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true",
                    help="only combine saved per-fold parts into the summary table")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    parts_dir = os.path.join(OUT_DIR, "cv_parts")
    if args.aggregate:
        return aggregate(parts_dir, args.out)

    import tensorflow as tf

    normal_files = sorted(glob.glob(os.path.join(CWRU_DIR, "Normal", "*.mat")),
                          key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
    if len(normal_files) < 4:
        sys.exit(f"CWRU data not found under {CWRU_DIR} (see README for the git clone command)")
    print("Extracting features (cached per run)...")
    healthy = [features_for(p, 0) for p in normal_files]            # index = motor load 0..3
    N14 = 14
    faults = {n: features_for(p, 2) for n, p in find_fault_files().items()}
    print(f"  healthy windows per load: {[len(h) for h in healthy]}; "
          f"fault recordings: {len(faults)}")

    results = {}   # (method) -> list of per-fold(+seed) metric dicts
    def add(name, m):
        results.setdefault(name, []).append(m)

    for held in ([args.fold] if args.fold >= 0 else range(4)):
        train_files = [i for i in range(4) if i != held]
        parts = [split_file(healthy[i]) for i in train_files]
        print(f"\n=== Fold: held-out load {held} HP "
              f"(train on loads {train_files}) ===")

        for cfg_name, cfg in CONFIGS.items():
            if only and cfg_name[0] not in only:
                continue
            cols = cfg["cols"]
            model_key = cfg.get("model", "lstm")
            for seed in range(args.seeds):
                tf.keras.utils.set_random_seed(seed + args.seed_offset)
                ae = MODELS[model_key](SEQ_LEN, len(cols), **cfg["kw"])
                if cfg.get("multi"):
                    groups = [seqs(healthy[i][:, cols]) for i in train_files]
                    aug_groups = None
                    if cfg.get("aug"):
                        aug_groups = [np.concatenate([seqs(a) for a in aug_features_for(
                            normal_files[i], cfg["aug"], seed + args.seed_offset + 17 * i)])
                            for i in train_files]
                    ae.fit_multicondition(groups, epochs=args.epochs, batch_size=8,
                                          aug_groups=aug_groups)
                elif cfg.get("cross"):
                    # calibrate on a whole different load; rotate which one per seed
                    cal = train_files[(seed + args.seed_offset) % len(train_files)]
                    fit_files = [i for i in train_files if i != cal]
                    tr = np.concatenate([seqs(healthy[i][:, cols]) for i in fit_files])
                    va = seqs(healthy[cal][:, cols])
                    ae.fit(tr, val_sequences=va, epochs=args.epochs, batch_size=8)
                elif cfg["gap"]:
                    tr = np.concatenate([seqs(p[0][:, cols]) for p in parts])
                    va = np.concatenate([seqs(p[1][:, cols]) for p in parts])
                    ae.fit(tr, val_sequences=va, epochs=args.epochs, batch_size=8)
                else:   # legacy: all healthy windows of the 3 files, tail of it used as "validation"
                    allw = [np.concatenate([p[0], p[1]])[:, cols] for p in parts]
                    tr = np.concatenate([seqs(w) for w in allw])
                    ae.fit(tr, epochs=args.epochs, batch_size=8, val_split=0.2, val_gap=False)
                m = evaluate(lambda s: ae.reconstruction_error(s),
                             lambda s: ae.reconstruction_error(s) > ae.threshold_,
                             healthy[held][:, cols],
                             {k: v[:, cols] for k, v in faults.items()}, "seq")
                m["threshold"] = float(ae.threshold_)
                add(f"{LABEL[model_key]} | " + cfg_name, m)
                print(f"  {LABEL[model_key]} {cfg_name[:40]:40s} FPR={m['fpr']*100:5.1f}% "
                      f"sens={m['sens_mean']*100:5.1f}%  AUROC={m['auroc']:.3f}  sep={m['sep_log10']:+.2f}")

        # classical baselines, once per feature set, trained on the same 3 healthy files
        for fs_name, nfb in ([] if args.no_baselines else
                            [("basic 10 feat", N_BASIC), ("extended 14 feat", N14)]):
            cols = list(range(nfb))
            Xtr = np.concatenate([np.concatenate([p[0], p[1]])[:, cols] for p in parts])
            for cls, kw in [(IsolationForestDetector, dict(contamination=0.05)),
                            (OneClassSVMDetector, dict(nu=0.05)),
                            (StatisticalThresholdDetector, dict())]:
                det = cls(**kw).fit(Xtr)
                m = evaluate(det.score, det.predict_anomaly, healthy[held][:, cols],
                             {k: v[:, cols] for k, v in faults.items()}, "win")
                add(f"{det.name} | {fs_name}", m)

    os.makedirs(parts_dir, exist_ok=True)
    tag = f"fold{args.fold if args.fold >= 0 else 'all'}_seed{args.seed_offset}_{'-'.join(sorted(only)) or 'ALL'}"
    with open(os.path.join(parts_dir, tag + ".json"), "w") as f:
        json.dump(results, f)
    print(f"Saved part {tag}.json  (combine with: python3 benchmark_cv.py --aggregate)")


def aggregate(parts_dir, out_name):
    results = {}
    for path in sorted(glob.glob(os.path.join(parts_dir, "*.json"))):
        for name, ms in json.load(open(path)).items():
            results.setdefault(name, []).extend(ms)
    if not results:
        sys.exit("no parts found; run the folds first")
    keys = ["fpr", "fpr_persist", "sens_mean", "sens_min", "auroc", "pr_auc_bal", "sep_log10"]
    def _stat(ms, k, fn):
        vals = [m[k] for m in ms if k in m]          # older parts lack newer metrics
        return float(fn(vals)) if vals else float("nan")
    summary = {name: {k: {"mean": _stat(ms, k, np.mean), "std": _stat(ms, k, np.std)} for k in keys}
               for name, ms in results.items()}
    n_runs = {name: len(ms) for name, ms in results.items()}
    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        json.dump({"protocol": "leave-one-load-out CV, CWRU 48kHz DE",
                   "window_s": WINDOW_S, "seq_len": SEQ_LEN,
                   "n_runs_per_method": n_runs, "summary": summary,
                   "per_run": results}, f, indent=2)
    print("=" * 122)
    print(f"{'method':58s}{'runs':>5s}{'FPR %':>9s}{'FPR+pers %':>12s}{'sens %':>9s}{'min sens %':>12s}{'AUROC':>8s}{'PR-AUC*':>9s}{'sep':>7s}")
    print("-" * 130)
    for name, s_ in summary.items():
        print(f"{name:58s}{n_runs[name]:5d}{s_['fpr']['mean']*100:9.1f}{s_['fpr_persist']['mean']*100:12.1f}"
              f"{s_['sens_mean']['mean']*100:9.1f}{s_['sens_min']['mean']*100:12.1f}"
              f"{s_['auroc']['mean']:8.3f}{s_['pr_auc_bal']['mean']:9.3f}{s_['sep_log10']['mean']:7.2f}")
    print("* PR-AUC weights classes equally (healthy vs. fault).")
    from tracking import log_run
    for name, s_ in summary.items():
        log_run("cv | " + name, params={"protocol": "leave-one-load-out", "runs": n_runs[name]},
                metrics={f"{k}_mean": s_[k]["mean"] for k in s_}, experiment="pdm-cwru-cv")
    print(f"Saved {os.path.join(OUT_DIR, out_name)}")


if __name__ == "__main__":
    main()
