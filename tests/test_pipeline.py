"""
Run from the project root:   python -m unittest discover -s tests -v
(or:  python tests/test_pipeline.py)
Covers: feature extraction, train/serve feature consistency, detection-lag
metric, email alerting (against a local in-process SMTP server), the
bridge's on_alert hook, and the trained model's end-to-end behaviour.
"""
import json, os, sys, tempfile, time, unittest
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from pdm_config import SAMPLE_RATE_HZ, WINDOW_S, WINDOW_OVERLAP

from synthetic_data import generate_run
from feature_engineering import (extract_windowed_features,
                                 extract_single_window_features, FEATURE_NAMES)
from evaluate import detection_lag, false_positive_rate


class TestFeatures(unittest.TestCase):
    def test_shape_and_names(self):
        run = generate_run(duration_s=30, sample_rate=200)
        X, t, y = extract_windowed_features(run)
        self.assertEqual(X.shape[1], len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(X).all())

    def test_fault_raises_kurtosis(self):
        run = generate_run(duration_s=600, sample_rate=200, seed=1)
        X, t, y = extract_windowed_features(run)
        k = FEATURE_NAMES.index("kurtosis")
        self.assertGreater(X[y == 2, k].mean(), X[y == 0, k].mean() + 5)

    def test_train_serve_consistency(self):
        """The live (MQTT) path must compute the SAME features as training."""
        run = generate_run(duration_s=20, sample_rate=200, seed=3)
        X, _, _ = extract_windowed_features(run, window_s=2.0, overlap=0.5)
        n = 400
        mean_temp = float(np.mean(run["temperature"][:n]))
        live = extract_single_window_features(run["vibration"][:n], 200, mean_temp)
        np.testing.assert_allclose(live, X[0], rtol=1e-4, atol=1e-4)


class TestMetrics(unittest.TestCase):
    def test_detection_lag_and_fpr(self):
        times = np.arange(100.0)
        labels = np.r_[np.zeros(50, int), np.ones(50, int)]
        flags = np.zeros(100, bool); flags[60:] = True
        self.assertEqual(detection_lag(times, flags, labels), 10.0)
        flags[5] = True
        self.assertAlmostEqual(false_positive_rate(flags, labels), 1 / 50)

    def test_isolated_spike_is_not_detection(self):
        times = np.arange(100.0)
        labels = np.r_[np.zeros(50, int), np.ones(50, int)]
        flags = np.zeros(100, bool); flags[55] = True      # single spike only
        self.assertIsNone(detection_lag(times, flags, labels))


class TestEmail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aiosmtpd.controller import Controller
        class H:
            def __init__(s): s.msgs = []
            async def handle_DATA(s, server, session, envelope):
                s.msgs.append(envelope.content.decode("utf8", "replace")); return "250 OK"
        cls.h = H(); cls.ctl = Controller(cls.h, hostname="127.0.0.1", port=8026)
        cls.ctl.start()
        os.environ.update({"PDM_SMTP_HOST": "127.0.0.1", "PDM_SMTP_PORT": "8026",
                           "PDM_SMTP_SECURITY": "none", "PDM_ALERT_FROM": "pdm@test.local",
                           "PDM_ALERT_TO": "ops@test.local"})

    @classmethod
    def tearDownClass(cls):
        cls.ctl.stop()

    def test_unconfigured_is_disabled(self):
        saved = os.environ.pop("PDM_SMTP_HOST")
        try:
            from alerting import EmailAlerter
            self.assertFalse(EmailAlerter().enabled)
        finally:
            os.environ["PDM_SMTP_HOST"] = saved

    def test_send_alert(self):
        from alerting import EmailAlerter
        a = EmailAlerter(); self.assertTrue(a.enabled)
        self.assertTrue(a.send_alert("PUMP_01", 5.5, 1.16))
        self.assertTrue(any("PUMP_01" in m and "5.50000" in m for m in self.h.msgs))

    def test_bridge_on_alert_logs_and_emails(self):
        import mqtt_bridge
        from alerting import EmailAlerter
        with tempfile.TemporaryDirectory() as d:
            mqtt_bridge.ALERT_LOG_PATH = os.path.join(d, "alert_log.jsonl")
            mqtt_bridge._alerter = EmailAlerter()
            before = len(self.h.msgs)
            mqtt_bridge.on_alert("FAN_07", 9.0, 1.16)
            for _ in range(50):                      # email is sent on a thread
                if len(self.h.msgs) > before: break
                time.sleep(0.1)
            self.assertGreater(len(self.h.msgs), before)
            self.assertIn("FAN_07", self.h.msgs[-1])
            rec = json.loads(open(mqtt_bridge.ALERT_LOG_PATH).read().splitlines()[-1])
            self.assertEqual(rec["machine_id"], "FAN_07")


class TestTrainedModel(unittest.TestCase):
    def test_unseen_machine_detected_with_low_false_alarms(self):
        mp = os.path.join(ROOT, "outputs", "lstm_autoencoder.keras")
        if not os.path.exists(mp):
            self.skipTest("run src/main.py first to create the model")
        from tensorflow import keras
        from lstm_autoencoder import make_sequences
        from calibration import load_calibration, apply_norm
        c = load_calibration(os.path.join(ROOT, "outputs", "threshold_and_norm.npz"))
        thr, L = c["threshold"], c["seq_len"]
        model = keras.models.load_model(mp)
        run = generate_run(duration_s=900, sample_rate=SAMPLE_RATE_HZ, seed=99,
                           degradation_start_frac=0.5, failure_frac=0.95)
        X, t, y = extract_windowed_features(run, window_s=WINDOW_S, overlap=WINDOW_OVERLAP)
        Z = apply_norm(make_sequences(X, seq_len=L), c["center"], c["scale"], c["clip"])
        err = np.mean((Z - model.predict(Z, verbose=0)) ** 2, axis=(1, 2))
        flags, ys = err > thr, y[L - 1:]
        self.assertLess(flags[ys == 0].mean(), 0.05)       # <5% false alarms (healthy)
        self.assertGreater(flags[ys == 2].mean(), 0.95)    # >95% detected (failed)


class TestCalibration(unittest.TestCase):
    def test_robust_scaler_survives_near_constant_feature(self):
        """Regression test for the temperature bug: a healthy feature with
        std ~0.007 must not turn a normal reading into a huge z-score."""
        from calibration import fit_scaler, apply_norm
        rng = np.random.default_rng(0)
        H = np.c_[rng.normal(0, 1, 500), 40.0 + rng.normal(0, 0.007, 500)]
        c, s = fit_scaler(H, "robust")
        z = apply_norm(np.array([[0.0, 25.0]]), c, s, clip=5.0)
        self.assertLessEqual(np.abs(z).max(), 5.0)          # clipped, not -1350
        z_old = apply_norm(np.array([[0.0, 25.0]]), *fit_scaler(H, "zscore"), clip=None)
        self.assertGreater(np.abs(z_old).max(), 100)       # old behaviour was far worse

    def test_threshold_methods_ordering(self):
        from calibration import compute_threshold
        rng = np.random.default_rng(1)
        e = rng.gamma(2.0, 0.5, 2000)                       # skewed, non-normal errors
        p995 = compute_threshold(e, "percentile", q=99.5)
        self.assertGreater(p995, np.percentile(e, 99))
        self.assertLess((e > p995).mean(), 0.01)
        self.assertGreater(compute_threshold(e, "pot"), np.percentile(e, 99))
        # tiny sample: POT falls back to percentile instead of crashing
        self.assertTrue(np.isfinite(compute_threshold(e[:20], "pot")))

    def test_persistence_filter(self):
        from calibration import persistent_flags, PersistenceFilter
        f = np.zeros(20, bool); f[5] = True                 # isolated spike
        self.assertFalse(persistent_flags(f, 3, 5).any())
        f[10:14] = True                                     # sustained fault
        out = persistent_flags(f, 3, 5)
        self.assertTrue(out[12])
        pf = PersistenceFilter(3, 5)
        stream = [pf.push(x) for x in f]
        self.assertEqual(stream, list(out))                 # streaming == batch


class TestEnvelopeFeatures(unittest.TestCase):
    def test_envelope_detects_repetition_rate(self):
        from feature_engineering import _envelope_features
        sr = 12000
        t = np.arange(sr // 2) / sr
        rng = np.random.default_rng(0)
        healthy = 0.05 * rng.normal(size=t.size)
        faulty = healthy + (np.mod(t, 1 / 107) < 0.0005) * np.sin(2 * np.pi * 3000 * t)
        kh, ph, _, _ = _envelope_features(healthy, sr)
        kf, pf, df, _ = _envelope_features(faulty, sr)
        self.assertGreater(kf, kh + 5)
        self.assertGreater(pf, ph * 5)
        self.assertLess(abs(df - 107) % 107 if df % 107 else 0, 5)  # peak at 107 Hz or a harmonic



class TestCwruLoader(unittest.TestCase):
    def test_picks_signal_matching_file_id(self):
        """Regression: 99_Normal_2.mat contains a stray X098_DE_time before its
        own X099_DE_time; the loader must return the file's OWN signal."""
        import scipy.io as sio
        from cwru_loader import load_mat_vibration
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "99_Normal_2.mat")
            sio.savemat(path, {"X098_DE_time": np.full((10, 1), 98.0),
                               "X099_DE_time": np.full((10, 1), 99.0)})
            self.assertTrue(np.all(load_mat_vibration(path) == 99.0))
            single = os.path.join(d, "174_0.mat")            # id/key mismatch, one key
            sio.savemat(single, {"X173_DE_time": np.full((5, 1), 173.0)})
            self.assertTrue(np.all(load_mat_vibration(single) == 173.0))



class TestFirmwarePayload(unittest.TestCase):
    """pdm_payload.h (Arduino side) must produce bytes the Python bridge decodes."""
    def test_cpp_packer_matches_python_decoder(self):
        import shutil, subprocess
        from payload_codec import decode_binary_window
        if shutil.which("g++") is None:
            self.skipTest("g++ not available")
        fw = os.path.join(ROOT, "arduino_r4_firmware")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "t.cpp")
            open(src, "w").write(
                '#include "pdm_payload.h"\n#include <cstdio>\n'
                'int main(){static int16_t c[1600];static uint8_t b[4096];'
                'for(int i=0;i<1600;i++)c[i]=(int16_t)((i*37)%2001-1000);'
                'size_t n=pdm_pack_window(b,sizeof b,c,1600,1600,123456u,0.0039f);'
                'fwrite(b,1,n,stdout);return 0;}')
            exe = os.path.join(d, "t")
            subprocess.run(["g++", "-std=c++11", "-I", fw, src, "-o", exe], check=True)
            raw = subprocess.run([exe], capture_output=True, check=True).stdout
        out = decode_binary_window(raw, remove_dc=False)
        exp = ((np.arange(1600) * 37) % 2001 - 1000).astype(np.float32) * np.float32(0.0039)
        self.assertEqual((out["sample_rate"], out["n"], out["t0_ms"]), (1600, 1600, 123456))
        self.assertTrue(np.allclose(out["samples"], exp))

    def test_codec_roundtrip_and_rejects_garbage(self):
        from payload_codec import encode_binary_window, decode_binary_window
        x = 0.3 * np.sin(np.linspace(0, 50, 1600)) + 1.0     # 1 g DC offset
        d = decode_binary_window(encode_binary_window(x, 1600, 99))
        self.assertAlmostEqual(float(d["samples"].mean()), 0.0, places=5)
        self.assertLess(np.abs(d["samples"] - (x - x.mean())).max(), 0.003)
        for bad in (b"", b"XXXX" + bytes(20), encode_binary_window(x, 1600, 1)[:-3]):
            with self.assertRaises(ValueError):
                decode_binary_window(bad)



class TestRulAndDrift(unittest.TestCase):
    def test_rul_unbiased_and_interval_coverage_on_synthetic_degradation(self):
        from rul import RULEstimator
        t = np.arange(0, 600.0); fail_t = 500.0
        hi_true = np.where(t < 100, 0.5, np.exp(np.log(8) / 400 * (t - 100)))
        errs, cov = [], 0
        for seed in range(5000, 5060):                      # seeds unseen during calibration
            sc = hi_true * np.exp(np.random.default_rng(seed).normal(0, 0.15, len(t)))
            r = RULEstimator(n_boot=100).estimate(t[:300], sc[:300], 1.0)
            true = fail_t - 299
            errs.append((r["rul_s"] - true) / true)
            cov += r["rul_low_s"] <= true <= r["rul_high_s"]
        self.assertLess(abs(np.median(errs)), 0.10)         # unbiased within 10%
        self.assertGreater(cov / 60, 0.68)                  # nominal 80%
        self.assertLess(cov / 60, 0.95)

    def test_rul_status_transitions(self):
        from rul import RULEstimator
        t = np.arange(100.0)
        self.assertEqual(RULEstimator().estimate(t, np.full(100, 0.4), 1.0)["status"], "healthy")
        self.assertEqual(RULEstimator().estimate(t, np.full(100, 50.0), 1.0)["status"], "imminent")
        self.assertEqual(RULEstimator().estimate(t[:5], np.full(5, 2.0), 1.0)["status"], "uncertain")

    def test_drift_psi_and_monitor(self):
        from drift import psi, FeatureDriftMonitor, PageHinkley
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, (2000, 3))
        self.assertLess(psi(ref[:, 0], rng.normal(0, 1, 500)), 0.1)
        self.assertGreater(psi(ref[:, 0], rng.normal(1.5, 1, 500)), 0.25)
        mon = FeatureDriftMonitor(ref, ["a", "b", "c"], window=100)
        for x in rng.normal(0, 1, (100, 3)): mon.update(x)
        self.assertFalse(mon.check()["drifted"])
        mon = FeatureDriftMonitor(ref, ["a", "b", "c"], window=100)
        for x in rng.normal([2, 2, 0], 1, (100, 3)): mon.update(x)
        self.assertTrue(mon.check()["drifted"])
        mon = FeatureDriftMonitor(ref, ["a", "b", "c"], window=100)   # faults must not count as drift
        for x in rng.normal(5, 1, (100, 3)): mon.update(x, is_normal=False)
        self.assertIsNone(mon.check())
        ph = PageHinkley(); xs = np.r_[rng.normal(0, .1, 300), rng.normal(1, .1, 100)]
        hits = [i for i, x in enumerate(xs) if ph.update(x)]
        self.assertTrue(hits and 300 <= hits[0] < 400)   # detects within ~50-100 samples of the shift
        ph = PageHinkley()
        self.assertFalse(any(ph.update(x) for x in rng.normal(0, .1, 2000)))



class TestServingAndStorage(unittest.TestCase):
    def test_onnx_matches_keras(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        import mqtt_bridge
        if not os.path.exists(mqtt_bridge.ONNX_PATH):
            self.skipTest("run main.py / export_onnx.py first")
        k, o = mqtt_bridge.LiveScorer("keras"), mqtt_bridge.LiveScorer("onnx")
        rng = np.random.default_rng(0)
        for _ in range(5):
            H = k.mean + rng.normal(size=(k.seq_len, len(k.mean))).astype(np.float32) * k.std * 2
            self.assertAlmostEqual(k.score_sequence(H)[0], o.score_sequence(H)[0], places=4)

    def test_postgres_backend_roundtrip_and_failsoft(self):
        from timescale_db import TimescaleDB
        dsn = os.environ.get("PDM_PG_DSN")
        bad = TimescaleDB("postgresql://x:y@localhost:1/none")      # must not raise
        bad.write_score("M", 1, 1, False)
        self.assertFalse(bad.enabled)
        if not dsn:
            self.skipTest("set PDM_PG_DSN to test against a real PostgreSQL")
        db = TimescaleDB(dsn)
        if not db.enabled:
            self.skipTest("PDM_PG_DSN is set but PostgreSQL is not reachable")
        mid = f"TEST_{int(time.time())}"
        for i in range(4):
            db.write_score(mid, 0.5 + i, 1.0, i >= 2, i)
        rows = db.query_recent_scores(mid, minutes=5)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(db.query_anomaly_history(mid, 1)), 2)



class TestAdvancedFeaturesModelsAugment(unittest.TestCase):
    def test_advanced_features_detect_fault_frequency_impacts(self):
        from feature_engineering import advanced_features, ADVANCED_FEATURE_NAMES
        sr, sh = 12000, 30.0
        t = np.arange(sr // 2) / sr
        healthy = 0.05 * np.random.default_rng(0).normal(size=t.size)
        faulty = healthy + (np.mod(t, 1 / (3.5848 * sh)) < 0.0005) * np.sin(2 * np.pi * 3000 * t)
        fh, ff = advanced_features(healthy, sr, sh), advanced_features(faulty, sr, sh)
        self.assertEqual(len(fh), len(ADVANCED_FEATURE_NAMES))
        i = ADVANCED_FEATURE_NAMES.index
        self.assertGreater(ff[i("bpfo_env_ratio")], 5 * fh[i("bpfo_env_ratio")])
        self.assertGreater(ff[i("spec_kurt_mean")], fh[i("spec_kurt_mean")] + 0.5)
        self.assertAlmostEqual(float(ff[2:6].sum()), 1.0, places=4)       # wavelet shares sum to 1
        self.assertTrue(np.all(advanced_features(np.zeros(2000), 1600, 30.0) == advanced_features(np.zeros(2000), 1600, 30.0)))
        self.assertTrue(np.isfinite(advanced_features(np.zeros(2000), 1600, None)).all())   # silent signal, no shaft speed

    def test_augmentation_is_plausible(self):
        from augment import augment_signal
        rng = np.random.default_rng(0)
        x = np.sin(2 * np.pi * 50 * np.arange(8000) / 1600).astype(np.float32)
        y = augment_signal(x, rng, scale=(1.0, 1.0), noise_rel=0.0, warp=(1.05, 1.05))
        f = lambda v: np.argmax(np.abs(np.fft.rfft(v))) * 1600 / len(v)
        self.assertAlmostEqual(f(y) / f(x), 1.05, delta=0.02)             # warp scales frequencies
        self.assertTrue(np.isfinite(augment_signal(x, rng)).all())

    def test_alternative_models_share_training_and_onnx(self):
        import tensorflow as tf
        from models import MODELS
        X = np.random.default_rng(0).normal(size=(120, 15, 14)).astype(np.float32)
        for name, cls in MODELS.items():
            tf.keras.utils.set_random_seed(0)
            ae = cls(15, 14)
            ae.fit(X, epochs=3, batch_size=16)
            self.assertTrue(np.isfinite(ae.threshold_))
            self.assertIs(type(ae._clone_untrained()), cls)
            self.assertEqual(ae.reconstruction_error(X[:5]).shape, (5,))

    def test_multicondition_with_augmented_groups(self):
        import tensorflow as tf
        from models import ConvAutoencoder
        rng = np.random.default_rng(1)
        groups = [rng.normal(c, 1, size=(60, 15, 14)).astype(np.float32) for c in (0.0, 0.3, 0.6)]
        aug = [g + 0.05 for g in groups]
        tf.keras.utils.set_random_seed(0)
        ae = ConvAutoencoder(15, 14)
        ae.fit_multicondition(groups, epochs=2, batch_size=16, aug_groups=aug)
        self.assertEqual(len(ae.val_errors_), 180)          # pooled over ORIGINAL sequences only
        with self.assertRaises(ValueError):
            ae.fit_multicondition(groups[:2], epochs=1)



class TestImsSnapshots(unittest.TestCase):
    def test_ims_format_parsing_and_bpfo_feature(self):
        """Files in the real IMS text format (timestamp names, 4 tab-separated
        channels): parsing, elapsed-time axis, and the fault-frequency feature."""
        from datetime import datetime, timedelta
        from ims_rul import snapshot_features, IMS_SHAFT_HZ
        from feature_engineering import FEATURE_NAMES, ADVANCED_FEATURE_NAMES
        rng = np.random.default_rng(0)
        sr, n = 20000, 20480
        t = np.arange(n) / sr
        t0 = datetime(2004, 2, 12, 10, 32, 39)
        with tempfile.TemporaryDirectory() as d:
            for i in range(6):
                x = 0.1 * rng.normal(size=(n, 4))
                if i >= 4:                                   # late snapshots: BPFO impacts on channel 0
                    x[:, 0] += (np.mod(t, 1 / (7.092 * IMS_SHAFT_HZ)) < 0.0004) * np.sin(2 * np.pi * 4000 * t) * 2
                name = (t0 + timedelta(minutes=10 * i)).strftime("%Y.%m.%d.%H.%M.%S")
                np.savetxt(os.path.join(d, name), x, fmt="%.4f", delimiter="\t")
            times, X = snapshot_features(d, 0)
            _, X1 = snapshot_features(d, 1)                  # a healthy channel for comparison
        self.assertEqual(X.shape, (6, len(FEATURE_NAMES) + len(ADVANCED_FEATURE_NAMES)))
        self.assertTrue(np.allclose(times, np.arange(6) * 600.0))
        j = (FEATURE_NAMES + ADVANCED_FEATURE_NAMES).index("bpfo_env_ratio")
        self.assertGreater(X[5, j], 4 * X[0, j])
        self.assertLess(abs(X1[5, j] - X1[0, j]), 0.1)



class TestOnDeviceStats(unittest.TestCase):
    def test_cpp_stats_match_numpy(self):
        import shutil, subprocess
        from scipy.stats import kurtosis, skew
        if shutil.which("g++") is None:
            self.skipTest("g++ not available")
        fw = os.path.join(ROOT, "arduino_r4_firmware")
        rng = np.random.default_rng(3)
        counts = np.clip(np.round(rng.normal(0, 80, 1600) + 256 + 600 * (np.arange(1600) % 97 == 0)), -32768, 32767).astype(np.int16)
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "t.cpp")
            open(src, "w").write(
                '#include "pdm_features.h"\n#include <cstdio>\n#include <cstdlib>\n'
                'int main(){static int16_t c[1600];for(int i=0;i<1600;i++){int v;if(scanf("%d",&v)!=1)return 1;c[i]=(int16_t)v;}'
                'PdmStats s=pdm_compute_stats(c,1600,0.0039f);'
                'printf("%.9g %.9g %.9g %.9g %.9g",s.rms,s.p2p,s.crest,s.kurtosis,s.skewness);return 0;}')
            exe = os.path.join(d, "t")
            subprocess.run(["g++", "-std=c++11", "-I", fw, src, "-o", exe], check=True)
            out = subprocess.run([exe], input=" ".join(map(str, counts)), capture_output=True, text=True, check=True).stdout
        got = np.array(out.split(), dtype=float)
        x = (counts.astype(np.float64) - counts.mean()) * 0.0039
        rms = np.sqrt(np.mean(x ** 2))
        exp = [rms, (counts.max() - counts.min()) * 0.0039, np.max(np.abs(x)) / rms, kurtosis(x), skew(x)]
        self.assertTrue(np.allclose(got, exp, rtol=2e-4, atol=1e-6), (got, exp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
