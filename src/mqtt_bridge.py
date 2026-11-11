"""
mqtt_bridge.py
----------------
The backend server referenced in the project brief's Layer 1 description
("Sensors sample continuously and stream readings via WiFi/MQTT to a
backend server"). This is that backend:

  1. Subscribes to pdm/<machine_id>/vibration and pdm/<machine_id>/temperature
     (published by the real Arduino R4 firmware, or mqtt_simulator.py for testing).
  2. Extracts the same time+frequency-domain features per window as the
     training pipeline (feature_engineering.extract_single_window_features).
  3. Buffers a rolling sequence of feature windows per machine and feeds it
     to the trained LSTM Autoencoder (outputs/lstm_autoencoder.keras) for a
     live reconstruction-error anomaly score.
  4. Publishes the result to pdm/<machine_id>/anomaly_score (for a future
     dashboard to subscribe to) and writes it to outputs/live_status.json
     (a simple polling-friendly alternative for the Streamlit dashboard,
     no MQTT client required on that side).
  5. Triggers an alert (console + hook for email/notification) when the
     anomaly score crosses the calibrated threshold, per the project brief's
     "System Output" section.

Prerequisite: run src/main.py first to train and save the model + threshold.

Run: python3 mqtt_bridge.py --broker localhost
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import paho.mqtt.client as mqtt
try:
    import psycopg2  # noqa: F401  -- must load BEFORE TensorFlow (OpenSSL clash, see timescale_db.py)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import extract_single_window_features, FEATURE_NAMES
from calibration import load_calibration, apply_norm, PersistenceFilter
from payload_codec import decode_binary_window
from rul import RULEstimator
from drift import FeatureDriftMonitor
from mqtt_common import add_mqtt_args, make_client, effective_port
from storage import open_db
from alerting import EmailAlerter
import threading

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_PATH = os.path.join(OUT_DIR, "lstm_autoencoder.keras")
ONNX_PATH = os.path.join(OUT_DIR, "lstm_autoencoder.onnx")
CALIB_PATH = os.path.join(OUT_DIR, "threshold_and_norm.npz")
LIVE_STATUS_PATH = os.path.join(OUT_DIR, "live_status.json")
SCORE_HISTORY_PATH = os.path.join(OUT_DIR, "score_history.jsonl")
ALERT_LOG_PATH = os.path.join(OUT_DIR, "alert_log.jsonl")
_alerter = None  # set in main(); None = console-only alerts
MAX_HISTORY_LINES_PER_MACHINE = 500  # keeps the dashboard's trend chart bounded


class LiveScorer:
    """Wraps the trained model + calibration for scoring live windows."""

    def __init__(self, backend="auto"):
        """backend: "onnx" (onnxruntime only, no TensorFlow), "keras", or "auto"
        (ONNX if outputs/lstm_autoencoder.onnx exists and onnxruntime is installed)."""
        have_onnx = os.path.exists(ONNX_PATH)
        have_keras = os.path.exists(MODEL_PATH)
        if not os.path.exists(CALIB_PATH) or not (have_onnx or have_keras):
            raise FileNotFoundError(
                f"Trained model/calibration not found. Run main.py first to "
                f"produce {MODEL_PATH} and {CALIB_PATH} (and export_onnx.py for ONNX)."
            )
        self.backend = None
        if backend in ("auto", "onnx") and have_onnx:
            try:
                import onnxruntime as ort
                self._sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
                self._in = self._sess.get_inputs()[0].name
                self.backend = "onnx"
            except ImportError:
                if backend == "onnx":
                    raise
        if self.backend is None:
            from tensorflow import keras
            self.model = keras.models.load_model(MODEL_PATH)
            self.backend = "keras"
        calib = load_calibration(CALIB_PATH)
        self.threshold = calib["threshold"]
        self.mean = calib["center"]      # (kept under the old name for callers)
        self.std = calib["scale"]
        self.clip = calib["clip"]
        self.seq_len = calib["seq_len"]
        self.reference_features = None
        ref_path = os.path.join(OUT_DIR, "healthy_reference_features.npy")
        if os.path.exists(ref_path):
            self.reference_features = np.load(ref_path)
        self.failure_hi = 8.0
        rc_path = os.path.join(OUT_DIR, "rul_config.json")
        if os.path.exists(rc_path):
            self.failure_hi = float(json.load(open(rc_path))["failure_hi"])
        print(f"Loaded model ({self.backend}): seq_len={self.seq_len}, "
              f"threshold={self.threshold:.5f}, clip={self.clip}")

    def score_sequence(self, feature_sequence):
        """feature_sequence: (seq_len, n_features) raw (un-normalized) features."""
        x = apply_norm(feature_sequence, self.mean, self.std, self.clip)
        x = x[np.newaxis, ...]  # add batch dim
        if self.backend == "onnx":
            recon = self._sess.run(None, {self._in: x})[0]
        else:
            recon = self.model.predict(x, verbose=0)
        error = float(np.mean((x - recon) ** 2))
        return error, error > self.threshold


class MachineState:
    """Per-machine rolling buffers (one instance per machine_id seen on MQTT)."""

    def __init__(self, seq_len, fallback_temperature, alert_k=3, alert_n=5):
        self.feature_history = deque(maxlen=seq_len)
        # Alert only when >= k of the last n windows exceed the threshold, so a
        # single noisy window doesn't page anyone (see calibration.PersistenceFilter).
        self.persistence = PersistenceFilter(alert_k, alert_n)
        self.last_raw_exceeds = False
        self.score_times = deque(maxlen=600)     # (window time s) for RUL
        self.scores = deque(maxlen=600)
        self.rul = None
        self.last_stats = None
        self.drift = None
        self.drift_monitor = None
        # IMPORTANT: seed with the TRAINING mean, not an arbitrary constant.
        # The training data's healthy temperature has very low variance
        # (tight std), so an arbitrary fallback (e.g. "25.0") can land many
        # std-devs away from the training distribution and, once normalized,
        # saturate the LSTM's activations -- corrupting the reconstruction
        # for the *entire* sequence, not just the one affected window. This
        # was caught by testing the bridge end-to-end before shipping it;
        # see README's "known limitations" section.
        self.last_temperature = fallback_temperature
        self.last_score = None
        self.last_is_anomaly = False
        self.window_count = 0


def on_alert(machine_id, score, threshold):
    """Fired once when a machine's sustained anomaly begins (see the
    `_alerted` flag in on_message). Does three things:
      1. prints to the console,
      2. appends to outputs/alert_log.jsonl (an audit trail of every alert),
      3. emails the configured recipients, if PDM_SMTP_* env vars are set.
    The email goes out on a background thread: an SMTP send can block for
    seconds, and this runs inside the MQTT callback, so sending inline would
    stall live scoring of every machine."""
    print(f"\n\U0001F6A8 ALERT: {machine_id} anomaly score {score:.5f} exceeds "
          f"threshold {threshold:.5f} - flag for inspection.\n")
    try:
        with open(ALERT_LOG_PATH, "a") as f:
            f.write(json.dumps({"wall_time": time.time(), "machine_id": machine_id,
                                "score": score, "threshold": threshold}) + "\n")
    except OSError as e:
        print(f"WARNING: could not write alert log: {e}")
    if _alerter is not None and _alerter.enabled:
        threading.Thread(target=_alerter.send_alert,
                         args=(machine_id, score, threshold), daemon=True).start()


def write_live_status(all_states, threshold):
    """Simple JSON file the dashboard can poll instead of running its own MQTT client."""
    status = {
        "updated_at": time.time(),
        "threshold": threshold,
        "machines": {
            mid: {
                "last_score": s.last_score,
                "is_anomaly": s.last_is_anomaly,
                "last_temperature_c": s.last_temperature,
                "windows_seen": s.window_count,
                "rul": s.rul,
                "device_stats": s.last_stats,
                "drift": s.drift,
            }
            for mid, s in all_states.items()
        },
    }
    tmp_path = LIVE_STATUS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, LIVE_STATUS_PATH)  # atomic write, safe for a concurrent dashboard reader


def append_score_history(machine_id, score, threshold, is_anomaly, t_ms):
    """Append one scored window to a per-run history log, for the dashboard's
    trend chart (live_status.json only ever holds the latest snapshot)."""
    entry = {
        "machine_id": machine_id, "t_ms": t_ms, "wall_time": time.time(),
        "score": score, "threshold": threshold, "is_anomaly": is_anomaly,
    }
    with open(SCORE_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Periodically trim so the file doesn't grow unbounded over a long-running demo.
    if random_trim_check():
        _trim_history_file()


def random_trim_check():
    import random as _random
    return _random.random() < 0.01  # ~1% of writes, cheap amortized trimming


def _trim_history_file():
    if not os.path.exists(SCORE_HISTORY_PATH):
        return
    by_machine = defaultdict(deque)
    with open(SCORE_HISTORY_PATH) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            dq = by_machine[e["machine_id"]]
            dq.append(e)
            if len(dq) > MAX_HISTORY_LINES_PER_MACHINE:
                dq.popleft()
    tmp_path = SCORE_HISTORY_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        for dq in by_machine.values():
            for e in dq:
                f.write(json.dumps(e) + "\n")
    os.replace(tmp_path, SCORE_HISTORY_PATH)


def main():
    ap = argparse.ArgumentParser()
    add_mqtt_args(ap)
    ap.add_argument("--backend", choices=["auto", "onnx", "keras"], default="auto",
                    help="model runtime (default: ONNX if available)")
    ap.add_argument("--alert-k", type=int, default=3,
                    help="alert when >= K of the last N windows exceed the threshold")
    ap.add_argument("--alert-n", type=int, default=5)
    args = ap.parse_args()

    scorer = LiveScorer(args.backend)
    temperature_feature_index = FEATURE_NAMES.index("mean_temperature")
    fallback_temperature = float(scorer.mean[temperature_feature_index])
    machines = defaultdict(lambda: MachineState(scorer.seq_len, fallback_temperature,
                                                args.alert_k, args.alert_n))
    rul_estimator = RULEstimator(failure_hi=scorer.failure_hi)
    db = open_db()  # PostgreSQL/TimescaleDB or InfluxDB; fails soft if down -- see storage.py
    global _alerter
    _alerter = EmailAlerter()  # console-only unless PDM_SMTP_* env vars are set

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"Connected to MQTT broker {args.broker}:{args.port} (reason_code={reason_code})")
        client.subscribe("pdm/+/vibration")     # legacy JSON text
        client.subscribe("pdm/+/vib_bin")       # binary window from the ADXL345 firmware
        client.subscribe("pdm/+/stats")         # on-device summary statistics
        client.subscribe("pdm/+/temperature")
        client.subscribe("pdm/+/status")

    def on_message(client, userdata, msg):
        try:
            topic_parts = msg.topic.split("/")  # pdm / <machine_id> / <kind>
            machine_id, kind = topic_parts[1], topic_parts[2]
        except IndexError:
            print(f"WARNING: unexpected topic shape: {msg.topic}")
            return

        state = machines[machine_id]

        if kind == "status":
            # status is a plain retained string ("online"/"offline"), not JSON
            print(f"[{machine_id}] status: {msg.payload.decode()}")
            return

        if kind == "vib_bin":
            try:
                payload = decode_binary_window(msg.payload)
            except ValueError as e:
                print(f"WARNING: bad binary vibration packet on {msg.topic}: {e}")
                return
            kind = "vibration"            # same handling from here on
        else:
            try:
                payload = json.loads(msg.payload.decode())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"WARNING: malformed JSON on {msg.topic}: {e}")
                return

        if kind == "stats":
            # compact per-window statistics computed ON the device (pdm_features.h).
            # Sent every window even when raw data is only sent every Nth window;
            # used for monitoring, the model itself scores the raw windows.
            state.last_stats = {k: payload.get(k) for k in ("rms", "p2p", "crest", "kurtosis", "skewness")}
            return

        if kind == "temperature":
            state.last_temperature = payload["temperature_c"]
            db.write_temperature(machine_id, payload["temperature_c"], payload.get("t_ms"))
            return

        if kind == "vibration":
            samples = payload["samples"]
            sample_rate = payload["sample_rate"]
            feats = extract_single_window_features(samples, sample_rate, state.last_temperature)
            state.feature_history.append(feats)
            state.window_count += 1
            if state.drift_monitor is None and scorer.reference_features is not None:
                state.drift_monitor = FeatureDriftMonitor(scorer.reference_features, FEATURE_NAMES)

            db.write_features(machine_id, dict(zip(FEATURE_NAMES, feats.tolist())), payload.get("t0_ms"))

            if len(state.feature_history) < scorer.seq_len:
                print(f"[{machine_id}] buffering... "
                      f"({len(state.feature_history)}/{scorer.seq_len} windows)")
            else:
                score, raw_exceeds = scorer.score_sequence(list(state.feature_history))
                is_anomaly = state.persistence.push(raw_exceeds)   # k-of-n persistent alarm
                state.last_score = score
                state.last_raw_exceeds = raw_exceeds
                state.last_is_anomaly = is_anomaly
                state.scores.append(score)
                state.score_times.append(state.window_count * (payload["n"] / sample_rate))
                if state.drift_monitor is not None:
                    state.drift_monitor.update(feats, is_normal=not raw_exceeds)  # faults are not drift
                    if state.window_count % 30 == 0:
                        d = state.drift_monitor.check()
                        if d is not None:
                            state.drift = {"drifted": d["drifted"], "features": d["features"]}
                            if d["drifted"] and not getattr(state, "_drift_warned", False):
                                print(f"[{machine_id}] DRIFT: healthy behaviour has shifted "
                                      f"({', '.join(d['features'])}). Consider recalibrating the baseline.")
                            state._drift_warned = d["drifted"]
                if raw_exceeds or state.rul is not None:
                    if state.window_count % 5 == 0 or is_anomaly:
                        r = rul_estimator.estimate(list(state.score_times), list(state.scores),
                                                   scorer.threshold)
                        state.rul = {k: r[k] for k in ("status", "rul_s", "rul_low_s", "rul_high_s", "hi_now")}
                        if r["status"] == "degrading":
                            client.publish(f"pdm/{machine_id}/rul", json.dumps({"machine_id": machine_id, **state.rul}))
                flag = "ANOMALY" if is_anomaly else ("spike" if raw_exceeds else "normal")
                print(f"[{machine_id}] window {state.window_count:4d} | "
                      f"score={score:.5f} | threshold={scorer.threshold:.5f} | {flag}")

                client.publish(f"pdm/{machine_id}/anomaly_score", json.dumps({
                    "machine_id": machine_id,
                    "score": score,
                    "threshold": scorer.threshold,
                    "is_anomaly": is_anomaly,
                    "t_ms": payload.get("t0_ms"),
                }))
                append_score_history(machine_id, score, scorer.threshold, is_anomaly,
                                      payload.get("t0_ms"))
                db.write_score(machine_id, score, scorer.threshold, is_anomaly, payload.get("t0_ms"))

                if is_anomaly and not getattr(state, "_alerted", False):
                    on_alert(machine_id, score, scorer.threshold)
                    state._alerted = True
                elif not is_anomaly:
                    state._alerted = False

            write_live_status(machines, scorer.threshold)

    client = make_client(args)
    client.on_connect = on_connect
    client.on_message = on_message
    port = effective_port(args)
    args.port = port
    client.connect(args.broker, port, keepalive=30)

    print("Bridge running. Waiting for Arduino R4 (or mqtt_simulator.py) data... (Ctrl+C to stop)")
    client.loop_forever()


if __name__ == "__main__":
    main()
