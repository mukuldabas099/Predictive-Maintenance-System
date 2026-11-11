"""
baselines.py
-------------
Classical anomaly detectors used to benchmark the LSTM Autoencoder against,
as specified in the project's evaluation plan:

  1. Isolation Forest   - tree-based; isolates points via random splits,
                           anomalies need fewer splits to isolate.
  2. One-Class SVM      - learns a boundary around normal data in feature
                           space (RBF kernel); points outside = anomalies.
  3. Statistical thresholding - naive baseline: flag a window anomalous if
                           any single feature exceeds mean +/- k*std of the
                           healthy training distribution. No temporal or
                           multivariate structure at all.

All three train on healthy-only feature vectors (same unsupervised setup as
the LSTM autoencoder) and score PER-WINDOW (not per-sequence), so scores are
aligned 1:1 with feature_engineering.py's output for fair comparison.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    name = "Isolation Forest"

    def __init__(self, contamination=0.01, n_estimators=200, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X_healthy):
        Xs = self.scaler.fit_transform(X_healthy)
        self.model.fit(Xs)
        return self

    def score(self, X):
        """Higher = more anomalous (we negate sklearn's score_samples, which
        is higher-for-normal)."""
        Xs = self.scaler.transform(X)
        return -self.model.score_samples(Xs)

    def predict_anomaly(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs) == -1


class OneClassSVMDetector:
    name = "One-Class SVM"

    def __init__(self, nu=0.02, kernel="rbf", gamma="scale"):
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X_healthy):
        Xs = self.scaler.fit_transform(X_healthy)
        self.model.fit(Xs)
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return -self.model.decision_function(Xs)  # higher = more anomalous

    def predict_anomaly(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs) == -1


class StatisticalThresholdDetector:
    """Naive baseline: flag if ANY feature exceeds mean +/- k*std (healthy stats)."""
    name = "Statistical Threshold"

    def __init__(self, k=3.0):
        self.k = k
        self.mean_ = None
        self.std_ = None

    def fit(self, X_healthy):
        self.mean_ = X_healthy.mean(axis=0)
        self.std_ = X_healthy.std(axis=0) + 1e-8
        return self

    def score(self, X):
        """Max absolute z-score across features = anomaly score."""
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=1)

    def predict_anomaly(self, X):
        return self.score(X) > self.k


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from synthetic_data import generate_run
    from feature_engineering import extract_windowed_features

    run = generate_run(duration_s=1800, sample_rate=200)
    X, t, y = extract_windowed_features(run)
    X_healthy = X[y == 0]
    split = int(len(X_healthy) * 0.8)
    X_train, X_val = X_healthy[:split], X_healthy[split:]

    for cls in [IsolationForestDetector, OneClassSVMDetector, StatisticalThresholdDetector]:
        det = cls().fit(X_train)
        flags = det.predict_anomaly(X)
        print(f"\n{det.name}:")
        for state, name in [(0, "healthy"), (1, "degrading"), (2, "failed")]:
            mask = y == state
            if mask.sum() > 0:
                print(f"  {name:10s}: flagged anomaly rate={flags[mask].mean()*100:.1f}%")
