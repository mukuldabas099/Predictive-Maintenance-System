"""
lstm_autoencoder.py
---------------------
The technical core of the project.

Approach:
  1. Train ONLY on healthy-operation feature sequences (unsupervised - no
     failure labels needed, addressing the rare-failure-data problem).
  2. The LSTM encoder compresses a sequence of feature vectors into a small
     latent representation; the LSTM decoder reconstructs the original
     sequence from it.
  3. Because the model has only ever seen healthy patterns, it reconstructs
     degrading/failing sequences poorly -> reconstruction error spikes.
  4. Reconstruction error (MSE per sequence) = the anomaly score.
  5. Threshold is calibrated from the reconstruction-error distribution on a
     held-out *healthy* validation set. The validation block is separated from
     the training block by a GAP of seq_len sequences, because sequences are
     built with stride 1 (consecutive sequences share seq_len-1 windows):
     without the gap, validation sequences are near-copies of training ones,
     errors look unrealistically small and the threshold ends up too tight.
     Threshold method is selectable (see calibration.py): mean+k*std (original),
     high percentile, or extreme-value (peaks-over-threshold).
     Features are normalised with robust (median/IQR) scaling + clipping.

LSTMs specifically (vs. a plain dense autoencoder) are used because
degradation is a *temporal* process - the model needs to learn how normal
feature sequences evolve over consecutive windows, not just treat each
window independently.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from calibration import fit_scaler, apply_norm, compute_threshold, DEFAULT_CLIP


def make_sequences(features, seq_len=20, stride=1):
    """
    Slide a window of `seq_len` consecutive feature vectors to build
    sequences for the LSTM. Returns (n_sequences, seq_len, n_features).
    """
    n = len(features)
    seqs = [features[i:i + seq_len] for i in range(0, n - seq_len + 1, stride)]
    return np.array(seqs, dtype=np.float32)


class LSTMAutoencoder:
    def __init__(self, seq_len, n_features, latent_dim=8, lstm_units=32,
                 scaling="robust", clip=DEFAULT_CLIP,
                 threshold_method="percentile", threshold_q=99.5,
                 threshold_sigma=3.0, threshold_margin=1.0):
        """
        scaling:          "robust" (median/IQR + floor) or "zscore" (original)
        clip:             clip normalised features to +/-clip (None = no clip)
        threshold_method: "percentile" | "pot" | "sigma" (original mean+k*std)
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.scaling = scaling
        self.clip = clip
        self.threshold_method = threshold_method
        self.threshold_q = threshold_q
        self.threshold_sigma = threshold_sigma
        self.threshold_margin = threshold_margin
        self.center_ = None
        self.scale_ = None
        self.threshold_ = None
        self.model = self._build(latent_dim, lstm_units)

    def _build(self, latent_dim, lstm_units):
        inputs = keras.Input(shape=(self.seq_len, self.n_features))

        # Encoder: compress the sequence down to a latent vector
        x = layers.LSTM(lstm_units, activation="tanh", return_sequences=True)(inputs)
        x = layers.LSTM(latent_dim, activation="tanh", return_sequences=False)(x)
        latent = x

        # Decoder: repeat latent vector across time steps, then reconstruct
        x = layers.RepeatVector(self.seq_len)(latent)
        x = layers.LSTM(latent_dim, activation="tanh", return_sequences=True)(x)
        x = layers.LSTM(lstm_units, activation="tanh", return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(x)

        model = keras.Model(inputs, outputs, name="lstm_autoencoder")
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return model

    def _normalize(self, X):
        return apply_norm(X, self.center_, self.scale_, self.clip)

    # backwards-compatible aliases (older scripts read ae.mean_ / ae.std_)
    @property
    def mean_(self):
        return self.center_

    @property
    def std_(self):
        return self.scale_

    def _calibrate(self, val_errors):
        self.threshold_ = compute_threshold(
            val_errors, method=self.threshold_method, sigma=self.threshold_sigma,
            q=self.threshold_q, margin=self.threshold_margin)

    def fit(self, healthy_sequences, val_sequences=None, epochs=40, batch_size=32,
            val_split=0.15, val_gap=True, verbose=0):
        """
        healthy_sequences: (n, seq_len, n_features) built ONLY from healthy data,
                           in chronological order.
        val_sequences:     optional healthy sequences from a block that does NOT
                           overlap training (e.g. a later time block or a
                           different file). Used for early stopping and
                           threshold calibration. If omitted, the last
                           `val_split` of healthy_sequences is used, separated
                           from the training part by a gap of seq_len sequences
                           (val_gap=True). val_gap=False reproduces the original
                           (leaky) behaviour for ablation studies.
        """
        healthy_sequences = np.asarray(healthy_sequences, dtype=np.float32)
        if val_sequences is None:
            n = len(healthy_sequences)
            n_val = max(int(n * val_split), 1)
            gap = self.seq_len if val_gap else 0
            n_train = n - n_val - gap
            if n_train < 8:                      # tiny dataset: shrink the gap
                gap = max(0, n - n_val - 8)
                n_train = n - n_val - gap
            train_raw = healthy_sequences[:n_train]
            val_raw = healthy_sequences[n - n_val:]
        else:
            train_raw = healthy_sequences
            val_raw = np.asarray(val_sequences, dtype=np.float32)

        self.center_, self.scale_ = fit_scaler(
            train_raw.reshape(-1, self.n_features), method=self.scaling)
        X_train = self._normalize(train_raw)
        X_val = self._normalize(val_raw)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_val, X_val),
            shuffle=True, callbacks=[early_stop], verbose=verbose,
        )

        val_errors = self.reconstruction_error(X_val, already_normalized=True)
        self._calibrate(val_errors)
        self.val_errors_ = val_errors
        return history

    def reconstruction_error(self, sequences, already_normalized=False):
        """Per-sequence MSE reconstruction error = anomaly score."""
        X = sequences if already_normalized else self._normalize(sequences)
        recon = self.model.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=(1, 2))

    def predict_anomaly(self, sequences):
        """Returns (anomaly_scores, is_anomaly_bool) using the calibrated threshold."""
        if self.threshold_ is None:
            raise RuntimeError("Model must be fit() before calling predict_anomaly().")
        scores = self.reconstruction_error(sequences)
        return scores, scores > self.threshold_

    def _clone_untrained(self):
        return type(self)(
            self.seq_len, self.n_features, self.latent_dim, self.lstm_units,
            scaling=self.scaling, clip=self.clip,
            threshold_method=self.threshold_method, threshold_q=self.threshold_q,
            threshold_sigma=self.threshold_sigma, threshold_margin=self.threshold_margin)

    def fit_multicondition(self, groups, epochs=40, batch_size=8, verbose=0, aug_groups=None):
        """
        Fit on healthy data from SEVERAL operating conditions (e.g. motor loads,
        speeds, seasons) and calibrate the threshold on CROSS-CONDITION errors.

        groups: list of arrays (n_i, seq_len, n_features), one per condition.

        Why: a threshold calibrated on a validation block from the same
        recording as the training data only reflects short-term noise. A model
        deployed on a machine whose load or speed changes sees a systematic
        shift that threshold never accounted for (leave-one-load-out CV on
        CWRU: ~50% false alarms on an unseen load). Here, for each condition g
        we train on the others and score g, so the pooled errors measure how
        the model behaves on a condition it has NOT seen. The final model is
        then trained on all conditions with that pooled threshold.
        Needs >= 3 conditions to be meaningful (each inner model needs >= 2).

        aug_groups: optional list (same length as groups) of AUGMENTED healthy
        sequences per condition. They are added to TRAINING only; the pooled
        calibration errors are always measured on the original sequences.
        """
        groups = [np.asarray(g, dtype=np.float32) for g in groups]
        if len(groups) < 3:
            raise ValueError("fit_multicondition needs data from >= 3 conditions")
        pooled = []
        for g in range(len(groups)):
            inner = self._clone_untrained()
            tr = [x for i, x in enumerate(groups) if i != g]
            if aug_groups is not None:
                tr += [a for i, a in enumerate(aug_groups) if i != g and len(a)]
            inner.fit(np.concatenate(tr), epochs=epochs, batch_size=batch_size, verbose=verbose)
            pooled.append(inner.reconstruction_error(groups[g]))
        pooled = np.concatenate(pooled)
        full = list(groups) + ([a for a in aug_groups if len(a)] if aug_groups is not None else [])
        hist = self.fit(np.concatenate(full), epochs=epochs, batch_size=batch_size,
                        verbose=verbose)
        # NB the final model has seen every condition, so its errors on them are
        # optimistic; the inner (leave-one-condition-out) errors are the honest ones.
        self._calibrate(pooled)
        self.val_errors_ = pooled
        return hist

    def set_threshold(self, healthy_val_sequences, method=None, **kwargs):
        """Recalibrate the threshold on (raw) healthy validation sequences with
        a different method / parameters, e.g. method="sigma", sigma=4."""
        errors = self.reconstruction_error(healthy_val_sequences)
        self.threshold_ = compute_threshold(
            errors, method=method or self.threshold_method,
            **{"sigma": self.threshold_sigma, "q": self.threshold_q,
               "margin": self.threshold_margin, **kwargs})

    def save_calibration(self, path):
        """Write everything the live bridge/demo need to score new data."""
        np.savez(path, threshold=self.threshold_, center=self.center_,
                 scale=self.scale_, clip=(np.inf if self.clip is None else self.clip),
                 seq_len=self.seq_len, threshold_method=self.threshold_method)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from synthetic_data import generate_run
    from feature_engineering import extract_windowed_features

    print("Generating synthetic run and extracting features...")
    run = generate_run(duration_s=1800, sample_rate=200)
    X, t, y = extract_windowed_features(run)

    seq_len = 15
    sequences = make_sequences(X, seq_len=seq_len)
    seq_labels = np.array([
        int(np.round(np.mean(y[i:i + seq_len]))) for i in range(0, len(y) - seq_len + 1)
    ])

    healthy_mask = seq_labels == 0
    healthy_seqs = sequences[healthy_mask]
    print(f"Total sequences: {len(sequences)}, healthy for training: {len(healthy_seqs)}")

    # Train/val split of healthy sequences only
    split = int(len(healthy_seqs) * 0.8)
    train_seqs, val_seqs = healthy_seqs[:split], healthy_seqs[split:]

    ae = LSTMAutoencoder(seq_len=seq_len, n_features=X.shape[1])
    print("Training LSTM autoencoder on healthy data only...")
    ae.fit(train_seqs, epochs=30, verbose=2)
    print(f"Calibrated threshold (method={ae.threshold_method}): {ae.threshold_:.5f}")

    scores, is_anom = ae.predict_anomaly(sequences)
    for state, name in [(0, "healthy"), (1, "degrading"), (2, "failed")]:
        mask = seq_labels == state
        if mask.sum() > 0:
            print(f"  {name:10s}: mean score={scores[mask].mean():.5f}, "
                  f"flagged anomaly rate={is_anom[mask].mean()*100:.1f}%")
