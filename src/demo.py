"""
demo.py -- one-command evaluation demo (no broker/hardware/DB needed).

Shows the core of the project end to end on a machine the model has NEVER
seen (different random seed from training):
  raw vibration -> time+FFT features -> LSTM Autoencoder -> anomaly score
  vs calibrated threshold -> alert.

Uses the already-trained model in ../outputs (run main.py first if missing).
Run:  python3 demo.py            (add --fast to skip the pacing delays)
Writes: ../outputs/demo_result.png and ../outputs/demo_replay.json
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_data import generate_run
from pdm_config import SAMPLE_RATE_HZ, WINDOW_S, WINDOW_OVERLAP
from feature_engineering import extract_windowed_features, FEATURE_NAMES

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def live(args):
    from tensorflow import keras                 # imported here so --offline never needs TF
    from lstm_autoencoder import make_sequences
    pause = 0 if args.fast else 0.03

    print("=" * 64)
    print(" AI/IoT PREDICTIVE MAINTENANCE - LIVE DEMO")
    print("=" * 64)
    model = keras.models.load_model(os.path.join(OUT, "lstm_autoencoder.keras"))
    from calibration import load_calibration, apply_norm
    c = load_calibration(os.path.join(OUT, "threshold_and_norm.npz"))
    thr, L = c["threshold"], c["seq_len"]
    print(f"[1] Loaded LSTM Autoencoder (trained on HEALTHY data only)")
    print(f"    calibrated threshold (healthy validation error) = {thr:.4f}")

    run = generate_run(duration_s=900, sample_rate=SAMPLE_RATE_HZ, seed=2024,
                       degradation_start_frac=0.5, failure_frac=0.95)
    print(f"[2] New unseen machine: 900 s of vibration + temperature @ {SAMPLE_RATE_HZ} Hz")
    print(f"    (bearing fault secretly begins at t = 450 s)")

    X, t, y = extract_windowed_features(run, window_s=WINDOW_S, overlap=WINDOW_OVERLAP)
    print(f"[3] Feature engineering: {X.shape[0]} windows x {X.shape[1]} features")
    print(f"    {', '.join(FEATURE_NAMES)}")

    seqs = make_sequences(X, seq_len=L)
    Z = apply_norm(seqs, c["center"], c["scale"], c["clip"])
    err = np.mean((Z - model.predict(Z, verbose=0)) ** 2, axis=(1, 2))
    st = t[L - 1:]                       # time of each scored sequence
    ys = y[L - 1:]
    flag = err > thr
    print(f"[4] Scoring {len(err)} sequences with the LSTM Autoencoder\n")

    print(" time(s)  score      status")
    print(" -------  ---------  --------------------------")
    alerted = None
    for i in range(0, len(err), 25):
        s = "ANOMALY" if flag[i] else "normal"
        print(f" {st[i]:7.0f}  {err[i]:9.3f}  {s}")
        time.sleep(pause * 5)
    onset = 450.0
    # Sustained alarm = 3 consecutive flagged windows (same rule as evaluate.py)
    sus = np.array([flag[i] and flag[i+1] and flag[i+2] for i in range(len(flag)-2)] + [False, False])
    post = np.where(sus & (st >= onset))[0]
    alerted = int(post[0]) if len(post) else None
    healthy = ys == 0
    n_fa = int(flag[healthy].sum())
    fa_rate = flag[healthy].mean() * 100
    print("\n" + "=" * 64)
    if alerted is not None:
        print(f" ALERT at t = {st[alerted]:.0f} s  (score {err[alerted]:.2f} > threshold {thr:.2f})")
        print(f" True fault onset: {onset:.0f} s  ->  detection lag: {st[alerted]-onset:.0f} s after onset")
        print(f" (machine still far from failure at 900 s -> time to schedule maintenance)")
    print(f" Healthy period: {n_fa} of {healthy.sum()} windows flagged "
          f"= {fa_rate:.2f}% false-alarm rate")
    pre = np.where(sus & healthy)[0]
    if len(pre):
        print(f" NOTE: {len(pre)} window(s) in the healthy period met the sustained-alarm rule "
              f"(false-alert episode near t = {st[pre[0]]:.0f} s,")
        print(f"       score only marginally above threshold). Honest limitation: mean+3*std "
              f"threshold trades a")
        print(f"       small false-alert risk for early detection; raise the sigma multiplier to reduce it.")
    print("=" * 64)

    # ---- figure ----
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(run["time"][::10], run["vibration"][::10], lw=.5, color="#3b6ea5")
    ax[0].set_ylabel("vibration (g)")
    ax[1].plot(t, X[:, 1], color="#8e44ad"); ax[1].set_ylabel("kurtosis")
    ax[2].plot(t, X[:, 9], color="#e67e22"); ax[2].set_ylabel("temp (C)")
    ax[3].semilogy(st, err, color="#c0392b", label="anomaly score")
    ax[3].axhline(thr, color="k", ls="--", label="calibrated threshold")
    ax[3].set_ylabel("anomaly score (log)"); ax[3].set_xlabel("time (s)")
    for a in ax:
        a.axvline(onset, color="gray", ls=":", lw=1)
    if alerted is not None:
        ax[3].axvline(st[alerted], color="red", lw=1.5, label="ALERT raised")
    ax[3].legend(fontsize=8)
    ax[0].set_title("Unseen machine: fault begins at 450 s (dotted line)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "demo_result.png"), dpi=130)

    # ---- replay data for the HTML page ----
    k = np.arange(0, len(err), 4)
    # align features to scored sequences (end-of-sequence window)
    Xs = X[L - 1:]
    replay = {
        "threshold": round(thr, 4), "onset": onset,
        "alert_t": float(st[alerted]) if alerted is not None else None,
        "t": [int(st[i]) for i in k],
        "score": [round(float(np.log10(max(err[i], 1e-6))), 2) for i in k],
        "rms": [round(float(Xs[i, 0]), 3) for i in k],
        "kurt": [round(float(Xs[i, 1]), 1) for i in k],
        "temp": [round(float(Xs[i, 9]), 1) for i in k],
    }
    json.dump(replay, open(os.path.join(OUT, "demo_replay.json"), "w"))
    print(" Saved outputs/demo_result.png and outputs/demo_replay.json")


def offline():
    """TensorFlow-free path: prints the saved results of the SAME demo run
    (outputs/demo_replay.json, written by a previous live run)."""
    r = json.load(open(os.path.join(OUT, "demo_replay.json")))
    thr = r["threshold"]
    print("=" * 64)
    print(" AI/IoT PREDICTIVE MAINTENANCE - DEMO (OFFLINE / PRECOMPUTED)")
    print("=" * 64)
    print(" NOTE: model is NOT being run now. These are the saved scores from a")
    print(" previous live run on an unseen simulated machine (fault starts at 450 s).")
    print(f" calibrated threshold = {thr:.4f}\n")
    print(" time(s)  score      status")
    print(" -------  ---------  --------------------------")
    for i in range(0, len(r["t"]), 6):
        sc = 10 ** r["score"][i]
        print(f" {r['t'][i]:7d}  {sc:9.3f}  {'ANOMALY' if sc > thr else 'normal'}")
    print("\n" + "=" * 64)
    if r.get("alert_t") is not None:
        print(f" ALERT at t = {r['alert_t']:.0f} s; true fault onset {r['onset']:.0f} s "
              f"-> detection lag {r['alert_t'] - r['onset']:.0f} s after onset")
    print(" Figure of this run: outputs/demo_result.png")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip pacing delays")
    ap.add_argument("--offline", action="store_true",
                    help="no TensorFlow: print saved results of the same run")
    args = ap.parse_args()
    if args.offline:
        return offline()
    try:
        import tensorflow  # noqa: F401
    except Exception as e:
        print(f"TensorFlow could not be loaded ({e}).\nFalling back to --offline mode.\n")
        return offline()
    live(args)


if __name__ == "__main__":
    main()
