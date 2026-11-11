"""
tune.py -- Optuna hyperparameter search (seq_len, latent_dim, units, architecture)
for the multi-condition training pipeline.

Objective (minimise) = false-alarm rate on an UNSEEN healthy operating condition
                       + miss rate on a faulty run.
Data: generated at the project's sample rate -- healthy recordings at 3 shaft
speeds (train + cross-condition calibration), a 4th unseen speed (false alarms),
and a degrading run (misses). Replace `build_data()` with your own recordings to
tune on real data; keep a held-out condition for the objective.

    python3 tune.py --trials 8          # resumable: results stored in tune.db (SQLite)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import extract_windowed_features
from lstm_autoencoder import make_sequences
from models import MODELS
from pdm_config import SAMPLE_RATE_HZ, WINDOW_S, WINDOW_OVERLAP
from synthetic_data import generate_run

DB = "sqlite:///" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tune.db"))


def build_data(duration=150):
    def feats(**kw):
        run = generate_run(duration_s=duration, sample_rate=SAMPLE_RATE_HZ, **kw)
        X, _, y = extract_windowed_features(run, WINDOW_S, WINDOW_OVERLAP)
        return X, y
    healthy = [feats(shaft_hz=hz, seed=i, degradation_start_frac=1.1, failure_frac=1.2)[0]
               for i, hz in enumerate((25.0, 29.5, 34.0))]
    unseen = feats(shaft_hz=31.5, seed=50, degradation_start_frac=1.1, failure_frac=1.2)[0]
    fault = feats(shaft_hz=31.5, seed=51, degradation_start_frac=0.05, failure_frac=0.4)[0]
    return healthy, unseen, fault


def make_objective(data, epochs):
    import tensorflow as tf
    healthy, unseen, fault = data

    def objective(trial):
        seq_len = trial.suggest_int("seq_len", 8, 24, step=4)
        latent = trial.suggest_categorical("latent_dim", [4, 8, 16])
        units = trial.suggest_categorical("units", [16, 32, 64])
        arch = trial.suggest_categorical("arch", ["cnn", "tcn"])
        tf.keras.utils.set_random_seed(trial.number)
        ae = MODELS[arch](seq_len, healthy[0].shape[1], latent_dim=latent, lstm_units=units,
                          threshold_method="percentile")
        ae.fit_multicondition([make_sequences(h, seq_len) for h in healthy], epochs=epochs, batch_size=8)
        fpr = float((ae.reconstruction_error(make_sequences(unseen, seq_len)) > ae.threshold_).mean())
        fs = ae.reconstruction_error(make_sequences(fault, seq_len)) > ae.threshold_
        miss = 1.0 - float(fs[-max(len(fs) // 3, 1):].mean())      # last third of the fault run should alarm
        trial.set_user_attr("fpr", fpr)
        trial.set_user_attr("miss", miss)
        return fpr + miss
    return objective


if __name__ == "__main__":
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="pdm", storage=DB, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(make_objective(build_data(), a.epochs), n_trials=a.trials)
    done = [t for t in study.trials if t.value is not None]
    print(f"{len(done)} trials stored. Best objective {study.best_value:.3f} with {study.best_params}")
    for t in sorted(done, key=lambda t: t.value)[:5]:
        print(f"  #{t.number:2d} obj={t.value:.3f} fpr={t.user_attrs['fpr']:.3f} miss={t.user_attrs['miss']:.3f} {t.params}")
