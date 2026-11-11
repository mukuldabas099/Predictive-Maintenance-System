"""
train_multicondition.py -- train the production model on YOUR healthy
recordings from several operating conditions, with cross-condition threshold
calibration (LSTMAutoencoder.fit_multicondition).

Why this exists: in leave-one-load-out tests on CWRU, thresholds calibrated
inside the training recordings gave ~53% false alarms on an unseen motor load;
calibrating on cross-condition errors reduced that to ~1.4% (see README).

Usage (>= 3 healthy recordings, one per operating condition, 1-D float arrays
in g sampled at --sample-rate, saved as .npy or one-column .csv):

    python3 train_multicondition.py --sample-rate 1600 \
        --recording idle.npy --recording half_load.npy --recording full_load.npy \
        --temperature 38 --out-dir ../outputs

Writes lstm_autoencoder.keras/.onnx, threshold_and_norm.npz,
healthy_reference_features.npy -- the files mqtt_bridge.py loads.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import extract_windowed_features
from lstm_autoencoder import LSTMAutoencoder, make_sequences
from pdm_config import SAMPLE_RATE_HZ, WINDOW_S, WINDOW_OVERLAP
from augment import augment_run
from models import MODELS

SEQ_LEN = 15


def load_recording(path):
    x = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",", usecols=0)
    return np.asarray(x, dtype=np.float32).ravel()


def train(recordings, sample_rate=SAMPLE_RATE_HZ, temperature=25.0, out_dir=None,
          epochs=40, seq_len=SEQ_LEN, window_s=WINDOW_S, overlap=WINDOW_OVERLAP, verbose=1,
          augment=3, model="lstm", seed=0):
    """augment: number of augmented copies of each recording added to TRAINING only
    (time-warp/amplitude/noise). On CWRU leave-one-load-out it gave the best
    separation margin and never hurt; 0 disables it."""
    rng = np.random.default_rng(seed)
    groups, aug_groups, ref = [], [], []
    for i, x in enumerate(recordings):
        run = {"time": np.arange(len(x)) / sample_rate, "vibration": x,
               "temperature": np.full(len(x), temperature, np.float32),
               "state": np.zeros(len(x), int), "sample_rate": sample_rate}
        X, _, _ = extract_windowed_features(run, window_s, overlap)
        if len(X) < seq_len + 20:
            raise ValueError(f"recording {i} is too short: {len(X)} windows "
                             f"(need >= {seq_len + 20}; record at least a few minutes)")
        groups.append(make_sequences(X, seq_len))
        ref.append(X)
        aug = [make_sequences(extract_windowed_features(augment_run(run, rng), window_s, overlap)[0], seq_len)
               for _ in range(augment)]
        aug_groups.append(np.concatenate(aug) if aug else np.empty((0, seq_len, X.shape[1]), np.float32))
        if verbose:
            print(f"  condition {i}: {len(x)/sample_rate:.0f} s -> {len(X)} windows")
    ae = MODELS[model](seq_len, groups[0].shape[2], threshold_method="percentile")
    ae.fit_multicondition(groups, epochs=epochs, batch_size=8,
                          aug_groups=aug_groups if augment else None)
    if verbose:
        print(f"Cross-condition threshold: {ae.threshold_:.5f} "
              f"(from {len(ae.val_errors_)} leave-one-condition-out errors)")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ae.model.save(os.path.join(out_dir, "lstm_autoencoder.keras"))
        ae.save_calibration(os.path.join(out_dir, "threshold_and_norm.npz"))
        np.save(os.path.join(out_dir, "healthy_reference_features.npy"),
                np.concatenate(ref).astype(np.float32))
        try:
            from export_onnx import export
            export(os.path.join(out_dir, "lstm_autoencoder.keras"),
                   os.path.join(out_dir, "lstm_autoencoder.onnx"))
        except Exception as e:
            print(f"(ONNX export skipped: {e})")
    return ae


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", action="append", required=True)
    ap.add_argument("--sample-rate", type=int, default=SAMPLE_RATE_HZ)
    ap.add_argument("--temperature", type=float, default=25.0,
                    help="typical healthy temperature (C); only used if you have no temperature log")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--augment", type=int, default=3, help="augmented copies per recording (0 = off)")
    ap.add_argument("--model", choices=sorted(MODELS), default="lstm")
    a = ap.parse_args()
    if len(a.recording) < 3:
        sys.exit("need >= 3 healthy recordings from different operating conditions")
    train([load_recording(p) for p in a.recording], a.sample_rate, a.temperature,
          a.out_dir, a.epochs, augment=a.augment, model=a.model)
