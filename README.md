# AI/IoT Predictive Maintenance — ML Core (Phase 1)

This is the **ML core** of the predictive maintenance project: feature engineering,
LSTM Autoencoder anomaly detection, baseline benchmarking, and threshold calibration.
It runs end-to-end on synthetic bearing-degradation data and is built so you can
drop in real CWRU/NASA data or live Arduino R4 data later with minimal changes.


## Project status

| Component | Status |
|---|---|
| Feature engineering, LSTM Autoencoder, threshold calibration, baselines | Built; verified on synthetic data and on 161 real CWRU files |
| Live scoring backend (MQTT bridge), persistent storage (InfluxDB), dashboard | Built; verified end-to-end against a local broker + simulator |
| Email alerts | Built and wired into the bridge; verified against a local SMTP server (not yet against Gmail/real SMTP) |
| Test suite (`tests/`) | 27 tests, all passing |
| Arduino R4 firmware | Written and reviewed; payload code verified on a PC; **never compiled or run on real hardware** |
| NASA IMS validation | Code written; **only tested on fake IMS-shaped data** (dataset not downloaded) |


## Phase 2 changes (calibration, features, hardware, serving)

Everything below was run in a sandbox. "Tested" says exactly what was exercised;
nothing was run on real Arduino hardware or on a real Grafana/Docker/TimescaleDB.

| Change | Where | Status |
|---|---|---|
| **Bug fix:** CWRU loader returned the wrong signal for `99_Normal_2.mat` | `cwru_loader.py` | fixed + regression test |
| Gap-separated validation, robust scaling + clipping, percentile / POT thresholds | `lstm_autoencoder.py`, `calibration.py` | tested |
| **Multi-condition calibration** (nested leave-one-condition-out threshold) | `fit_multicondition`, `train_multicondition.py` | tested on CWRU CV + synthetic |
| k-of-n alert persistence (3 of 5) | `calibration.py`, bridge | tested end to end |
| Envelope (Hilbert) features, 14 total | `feature_engineering.py` | tested at 12 kHz and 1.6 kHz |
| **1600 Hz / 1 s windows**, single source of truth | `pdm_config.py` | tested |
| ADXL345 SPI firmware + compact binary payload | `arduino_r4_firmware/` | **payload tested on a PC; firmware NOT compiled/run on hardware** |
| Secured Mosquitto (auth, ACLs, TLS) + client auth/TLS flags | `deploy/`, `mqtt_common.py` | tested on a real broker |
| ONNX export + onnxruntime serving (no TensorFlow needed) | `export_onnx.py`, bridge | tested: matches Keras to 1e-6 |
| PostgreSQL/TimescaleDB storage, Grafana dashboard, docker-compose | `timescale_db.py`, `deploy/` | Postgres + dashboard SQL tested; **compose/Grafana/hypertables not run** |
| Drift detection (PSI, KS, Page-Hinkley) | `drift.py` | unit-tested; live in bridge |
| Remaining useful life from the anomaly score | `rul.py` | tested on synthetic degradation only |
| MLflow tracking (optional, fail-soft), fixed training seed | `tracking.py`, `main.py` | tested |
| Leave-one-load-out CV + ablation, AUROC / balanced PR-AUC | `benchmark_cv.py` | done (1 seed) |

Test suite: **27 tests** (`python3 -m unittest discover -s tests`); two skip cleanly
without PostgreSQL / onnxruntime.

### Real-data results: CWRU leave-one-load-out (48 kHz DE, 4 folds, 1 seed)
Train and calibrate on 3 healthy loads, test false alarms on the unseen 4th.
Sensitivity is 100% and AUROC ~1.0 for every method (CWRU is too easy to rank them).

| method | FPR % on unseen load |
|---|---|
| LSTM-AE A: original (z-score, mean+3sd, leaky validation) | 53.1 |
| B: + gap-separated validation | 58.0 |
| C: + robust scaling, 99.5th-percentile threshold | 57.8 |
| D: + envelope features | 53.1 |
| E: threshold calibrated on one whole different load | 25.0 |
| **F: multi-condition (nested) calibration** | **1.4** (5.5 / 0 / 0 / 0 per fold) |
| Isolation Forest / One-Class SVM / Statistical (basic features) | 42.8 / 67.3 / 48.6 |

- Scores separate healthy from faulty almost perfectly everywhere; what fails is the
  **threshold** when the operating condition changes. Only F fixes it.
- B, C and D showed no consistent benefit on this protocol (single seed, differences
  within noise). Persistence barely helps here because the false alarms are a steady shift.
- Caveats: one seed; only 4 loads; faults are all at 0 HP; a single 5.5% fold remains.
- **The older "23.1% FPR" CWRU numbers further below used the buggy loader (3 distinct
  healthy recordings, not 4) and should not be cited.**
- Practical consequence: record healthy data across the machine's real operating
  conditions and train with `train_multicondition.py`.

### Synthetic data at 1600 Hz (seed 0, reproducible)
| method | false-positive % | detection lag (s) |
|---|---|---|
| LSTM Autoencoder | 0.08 | 143.5 |
| Isolation Forest | 1.04 | 0.0 |
| One-Class SVM | 2.84 | 0.0 |
| Statistical Threshold | 4.94 | 0.0 |

The LSTM's stricter threshold trades earlier detection for 13-60x fewer false alarms.
Lag at a single threshold is fragile on this synthetic ramp. Robust scaling lowers
synthetic AUROC (0.999 -> 0.973) because the old z-score exploited the simulator's
noise-free temperature; without temperature both score ~0.946.
Before the fixed seed was added, runs varied (lag 0 s vs 143 s).

### Serving
ONNX scoring takes 0.13 ms per window vs 66.9 ms with Keras, and the ONNX path never
imports TensorFlow. Known crash fixed: TensorFlow + psycopg2 SSL (see `deploy/README.md`).

### RUL caveat
`rul.py` extrapolates an exponential trend of the health indicator. Its interval was
calibrated on **synthetic** exponential degradation (80% +/- 3% coverage; the raw bootstrap
only covered ~58%). The failure level in `outputs/rul_config.json` comes from the
simulated failure state: replace it with values from your own run-to-failure data.

### Phase 3: features, models, augmentation, fusion (CWRU leave-one-load-out, 4 folds, seed 0, multi-condition protocol)

| config | FPR % | sensitivity % (mean / weakest fault) | separation* |
|---|---|---|---|
| F: LSTM baseline | 1.4 | 100 / 100 | 0.49 |
| **I: F + signal augmentation (time-warp, amplitude, noise)** | **0.0** | **100 / 100** | **0.58** |
| G: F + advanced features (spectral kurtosis, wavelet packet, fault-band energy) | 0.0 | **96.2 / 77.7** | 0.52 |
| H: F + fan-end channel fusion | 5.5 | 100 / 100 | 0.52 |
| J: 1D-CNN autoencoder | 4.1 | 100 / 100 | 0.54 |
| K: TCN autoencoder | 7.2 | 100 / 100 | 0.54 |

*separation = log10(weakest fault's median score / healthy 99.5th percentile); higher is better; AUROC is 1.0 for all, so it cannot rank these.

Read this carefully:
- **Augmentation is the only addition that did not hurt**, and it is now the default in `train_multicondition.py`. Its false-alarm advantage over F is **not robust**: on the hardest fold F failed (5.5%) with seed 0 but scored 0% with seed 1. Its separation margin was higher in both seeds on that fold (+0.43 vs +0.22, +0.27 vs +0.20). Treat it as a modest, consistent gain in margin.
- **Advanced features reached 0% false alarms only by loosening the threshold**: the weakest fault was detected in only 77.7% of windows. Not recommended as-is. They remain available (`advanced=True`).
- Fusion and the CNN/TCN were worse on the hardest fold. The LSTM remains the default.
- One seed per cell (two on one fold), four loads, faults all at 0 HP. Differences between the top configs are small relative to that uncertainty.

### Run-to-failure / RUL: experimental
`ims_rul.py` processes one feature vector per snapshot (IMS-style data). It is tested on generated IMS-format files (parsing, timestamps, fault-frequency feature) but **never on the real IMS download (unreachable from the dev sandbox)**.
On generated run-to-failure data, RUL accuracy was **poor** (about -50% to -70% error at 75-90% of the run, true value outside the interval). Two real bugs were found and fixed on the way (±5 clipping saturates the health indicator at ~30x threshold; failure level must not come from the run being predicted), but a failure level learned from one run does not transfer when another run ends at a different damage level. **Do not use RUL numbers for decisions without calibrating on your own run-to-failure data.** `rul.py` interval coverage (80% +/- 3%) only holds for ideal exponential trends.

### Tuning, pipeline, on-device statistics
- `tune.py`: Optuna search (seq_len, latent_dim, units, architecture), resumable in `tune.db`. On the generated data **every trial scored a perfect 0.000**, so it cannot discriminate; run it on your own recordings.
- `dvc.yaml`: tested with a real `dvc repro` (reproduces threshold 0.86994 and the 0.08% / 143.5 s results exactly). The CWRU path is outside the repo so DVC cannot track it.
- `arduino_r4_firmware/pdm_features.h`: RMS, peak-to-peak, crest, kurtosis, skewness computed on the device (matches NumPy/SciPy in a PC test); sent on `pdm/<id>/stats`; raw windows every `RAW_EVERY_N_WINDOWS` (default 1). The bridge shows them under `device_stats`. Hardware-untested like the rest of the firmware.

### Not done
PyTorch migration (ONNX already removes TensorFlow from serving; a training rewrite has no measured benefit); real run-to-failure datasets (IMS, FEMTO, XJTU-SY, Paderborn: hosts unreachable from the dev sandbox); USAD / Deep SVDD / MiniRocket; tsfresh/tsflex; InfluxDB 3 / QuestDB; second physical sensor (fusion tested only with CWRU's fan-end channel); Docker, Grafana and TimescaleDB hypertables (never run); all hardware.

Reproduce the CV: `python3 benchmark_cv.py --fold 0 --epochs 40` (folds 0-3, ~3 min each),
then `--aggregate`. Add `--only F --no-baselines` for the multi-condition config.

## Quick start (Windows)

1. Install Python 3.10-3.12 from python.org (tick "Add python.exe to PATH").
2. In a terminal: `pip install -r requirements.txt`
3. Double-click `run_demo.bat` (or `python src/demo.py`). If TensorFlow won't load, it
   automatically falls back to `--offline` mode (saved results of the same run).
4. Optional: `run_tests.bat` (runs all tests), `run_dashboard.bat` (needs the live system running).
Linux/Mac: `./run_demo.sh`. Use `python3` instead of `python` there.

## What's here

```
pdm_project/
├── src/
│   ├── synthetic_data.py      # Physics-informed healthy→degrading→failed bearing simulator
│   ├── feature_engineering.py # Time-domain (RMS, kurtosis, skew, crest factor...) +
│   │                          # frequency-domain (FFT: spectral energy, dominant freq,
│   │                          # centroid, entropy) features over sliding windows
│   ├── lstm_autoencoder.py    # The core model: trains on healthy-only sequences,
│   │                          # reconstruction error = anomaly score, threshold
│   │                          # calibrated as mean + 3*std of healthy validation error
│   ├── baselines.py           # Isolation Forest, One-Class SVM, statistical thresholding
│   ├── calibration.py         # Robust scaling, threshold methods, k-of-n persistence
│   ├── benchmark_cv.py        # Leave-one-load-out CV + ablation on real CWRU data
│   ├── evaluate.py            # False-positive rate, detection lag, comparison plot
│   └── main.py                # Runs the full pipeline end-to-end
├── outputs/                   # Generated: plot, benchmark table, saved model
└── README.md
```

## Run it

```bash
cd src
pip install numpy scipy scikit-learn tensorflow-cpu matplotlib
python3 main.py
```

Produces in `outputs/`:
- `anomaly_comparison.png` — anomaly score over time, all 4 methods, degradation onset marked
- `benchmark_results.json` — false positive rate + detection lag per method
- `lstm_autoencoder.keras` — trained model
- `threshold_and_norm.npz` — calibrated threshold + feature normalization stats (needed to score new data)

## ORIGINAL benchmark run (200 Hz, 2 s windows; superseded by the 1600 Hz table in "Phase 2" above)

| method | false positive rate | detection lag |
|---|---|---|
| **LSTM Autoencoder** | **0.8%** | **0.0s** (detects at earliest possible window) |
| Isolation Forest | 0.9% | 284.0s |
| One-Class SVM | 3.4% | 0.0s |
| Statistical Threshold | 2.6% | 27.0s |

Note: on this synthetic data, degradation is a smooth deterministic ramp, so several
methods detect it almost immediately — the more informative comparison is the **score
trajectory shape** (see plot). The LSTM Autoencoder and Statistical Threshold both
produce smooth, low-noise scores; Isolation Forest and One-Class SVM are much noisier
on raw feature space, meaning they'd generate more nuisance alerts on a real noisy
factory floor even though their aggregate numbers look competitive here. This gap will
be far more pronounced on real bearing data (CWRU/NASA), which has more sensor noise
and non-stationary healthy-state variation than this simulator — worth re-running this
same benchmark on real data before writing conclusions in your report.

## Why these specific design choices (for your report)

- **Unsupervised, healthy-only training**: directly solves the "failures are rare"
  problem stated in the project brief — no labeled failure examples needed to train.
- **LSTM over a plain dense/CNN autoencoder**: bearing degradation is a *temporal*
  process (fault signatures build up gradually across consecutive windows, not within
  a single window), so the model needs to learn normal *sequences* of feature vectors,
  not just normal single windows.
- **Time + frequency domain features together**: kurtosis/crest factor catch the
  "peakiness" from impact-like fault transients; spectral centroid/dominant frequency
  catch the fault-frequency shift that shows up *before* it's visible in raw amplitude
  — this is what justifies including FFT features per the project brief.
- **mean + 3σ threshold**: statistically principled (assumes ~healthy reconstruction
  error is roughly normal; 3σ ≈ 99.7% of healthy data below threshold) rather than an
  arbitrary cutoff — and it's a single number (`sigma_multiplier` in
  `LSTMAutoencoder.set_threshold()`) you can tune to trade off false alarms vs. missed
  detections for your specific report section on that tradeoff.

## Swapping in real data (CWRU / NASA IMS bearing datasets)

Everything downstream of raw data only depends on this dict shape (see the docstring
in `synthetic_data.py`):

```python
{
    "time": np.array,          # seconds
    "vibration": np.array,     # accelerometer signal, same length as time
    "temperature": np.array,   # same length as time
    "state": np.array,         # optional, for evaluation only: 0=healthy,1=degrading,2=failed
    "sample_rate": float,      # Hz
}
```

Write a loader (e.g. `load_cwru_run(path)` using `scipy.io.loadmat`) that returns this
shape, and everything in `feature_engineering.py`, `lstm_autoencoder.py`, `baselines.py`
works unchanged. CWRU dataset: engineering.case.edu/bearingdatacenter. NASA IMS: search
"NASA IMS Bearing Dataset" on data.nasa.gov — this one is especially good because it
has full run-to-failure sequences with no `state` labels needed (you provide your own
by inspecting when RMS starts trending up), closer to a real deployment scenario.

## Real-data validation (CWRU Bearing Dataset)

`src/cwru_loader.py` and `src/validate_real_data.py` re-run the exact same trained
pipeline on **real** Case Western Reserve University bearing recordings (cloned from
`github.com/s-whynot/CWRU-dataset`, 161 `.mat` files, 890MB — see setup below).

**Important dataset difference from the synthetic run:** CWRU is a set of short
(~1–5s), *fixed-condition* recordings — either fully healthy or one fixed fault
type/severity — not a continuous run-to-failure trace. So "detection lag" doesn't
apply; instead this measures **false positive rate on held-out real healthy data**
and **sensitivity per real fault type × severity** (Inner Race / Ball / Outer Race,
at 0.007"/0.014"/0.021" defect diameters). CWRU also has no temperature channel
(zero-filled) and only ~35s of healthy baseline audio total across 4 files — a real
limitation of the dataset, not of the pipeline; more baseline recordings would give
a more stable threshold.

### Results

| method | false positive rate (real healthy data) | sensitivity, all 9 real fault conditions |
|---|---|---|
| LSTM Autoencoder | 23.1% | 100% |
| Isolation Forest | 5.9% | 100% |
| One-Class SVM | 14.1% | 100% |
| Statistical Threshold | 3.5% | 100% |

Full breakdown in `outputs/outdated/real_data_benchmark.json` (outdated), plot in `outputs/real_data_sensitivity.png`.

**Honest reading of these numbers, for your report:**
- **Every method hits 100% sensitivity on every real fault condition, including the
  smallest 0.007" defects.** This is expected — CWRU is a widely-used *easy* benchmark
  in PdM literature (its seeded faults produce very pronounced vibration signatures
  even at the smallest severity), so it doesn't strongly differentiate the methods on
  detection difficulty. Don't over-claim "the LSTM Autoencoder proves superior fault
  detection" from this alone.
- **The real differentiator here is false positive rate, and on this run the LSTM
  Autoencoder is worse than all three baselines (23% vs 3.5–14%).** This is very
  likely a *training data volume* artifact, not an architectural weakness: the LSTM
  had only ~189 training sequences (vs. thousands in the synthetic run) — deep
  sequence models are more data-hungry than Isolation Forest / One-Class SVM /
  simple z-scores, and CWRU's healthy baseline is short. This is a legitimate,
  reportable limitation: it demonstrates the real-world engineering tradeoff between
  the synthetic run's result (LSTM had the *best* FPR when given enough data) and
  this run (LSTM has the *worst* FPR when data is scarce) — a good discussion point
  for your report's "limitations and future work" section, and a concrete argument
  for why the Arduino deployment should collect a longer healthy-baseline period before
  calibrating the threshold.
- A dataset with a genuine continuous degradation trace (NASA IMS Bearing Dataset) is
  needed to validate the *detection lag* claim from the project brief — CWRU's
  fixed-condition files can't test that at all. Worth pursuing as a follow-up if your
  report needs that number from real data.

### Setup (if you want to re-run or extend this)

```bash
cd pdm_project/..   # one level above pdm_project/, so paths in the scripts resolve
git clone https://github.com/s-whynot/CWRU-dataset.git cwru_data
cd pdm_project/src
python3 validate_real_data.py
```

## IoT layer: Arduino R4 firmware + live scoring backend

`arduino_r4_firmware/` and `src/mqtt_bridge.py` complete the loop from raw sensor data
to a live anomaly alert:

```
Arduino UNO R4 WiFi (ADXL345 + DS18B20) --MQTT--> mqtt_bridge.py --uses trained model--> alert
        (arduino_r4_firmware/)          (feature extraction + LSTM scoring, live)
```

- **`arduino_r4_firmware/arduino_r4_firmware.ino`** — Arduino sketch: samples vibration at 200Hz,
  batches 2s windows, reads temperature periodically, streams both over MQTT.
  Full wiring diagram, library list, and MQTT topic schema in
  `arduino_r4_firmware/README.md`.
- **`src/mqtt_bridge.py`** — the backend server referenced in the project brief:
  subscribes to the Arduino's MQTT topics, extracts the same features the model was
  trained on, feeds a rolling window sequence to the trained LSTM Autoencoder, and
  triggers an alert when the anomaly score crosses the calibrated threshold. Also
  writes `outputs/live_status.json` for a future dashboard to poll.
- **`src/mqtt_simulator.py`** — publishes synthetic data using the *exact same*
  MQTT schema as the real firmware, so the backend (and later, the dashboard) can
  be built and tested end-to-end without physical hardware.

### Verified end-to-end (no hardware required)

This was tested locally with a Mosquitto broker: simulator → MQTT → bridge →
live LSTM scoring → alert. A real bug was caught and fixed in the process, worth
keeping in mind for your report's limitations/testing section:

> **Bug found:** the bridge's fallback temperature value (used only before the
> first real MQTT temperature reading arrives) was an arbitrary constant
> (25°C). But the synthetic training data's healthy-state temperature has an
> extremely tight standard deviation (~0.007°C — see `main.py`'s output), so
> that fallback normalized to a z-score of roughly -1350, which saturated the
> LSTM's internal activations and corrupted reconstruction error for the
> entire first sequence, not just the affected window (score ≈ 12,000 against
> a threshold of ≈ 1.16). **Fix:** seed the fallback with the trained model's
> own feature mean instead of an arbitrary constant (see `mqtt_bridge.py`,
> `MachineState.__init__`). **Broader lesson for the report:** this is a
> symptom of the synthetic simulator's healthy-state temperature being
> unrealistically noise-free — a real DS18B20 will have more natural sensor
> noise, which would make the trained threshold's tightness a smaller risk in
> practice, but it's a good concrete example of a synthetic-to-real gap to
> discuss.

To reproduce:
```bash
# terminal 1
mosquitto -p 1883
# terminal 2
cd src && python3 mqtt_bridge.py --broker localhost
# terminal 3
cd src && python3 mqtt_simulator.py --broker localhost --scenario degrading --duration 90 --speed 25
```

## Monitoring dashboard (Streamlit)

`src/dashboard.py` is the "System Output" piece from the project brief: live
sensor readings, current anomaly score per machine, and a threshold-crossing
alert banner. It's decoupled from MQTT entirely — it just polls the files
`mqtt_bridge.py` writes (`outputs/live_status.json` for the latest snapshot,
`outputs/score_history.jsonl` for the trend chart), so it can run on a
completely different machine from the bridge if needed.

```bash
# terminal 1: broker
mosquitto -p 1883
# terminal 2: backend
cd src && python3 mqtt_bridge.py --broker localhost
# terminal 3: data source (simulator, or the real Arduino R4)
cd src && python3 mqtt_simulator.py --broker localhost --scenario degrading
# terminal 4: dashboard
cd src && streamlit run dashboard.py
```

Open the URL Streamlit prints (default `http://localhost:8501`). Each machine
gets a card: current anomaly score, temperature, a trend chart with the
threshold drawn as a dashed line and anomalous points highlighted in red, and
a status badge that flips to 🚨 **ANOMALY** the moment the score crosses
threshold — with a banner at the top listing every flagged machine, satisfying
the brief's "flagging the machine for inspection before it actually fails"
requirement. The email/notification hook mentioned in the brief is a single
function (`on_alert()` in `mqtt_bridge.py`) — currently prints to console;
wire in `smtplib` or a webhook there for a real deployment.

**Verified:** ran the full chain (broker → bridge → simulator → dashboard)
headlessly, confirmed the dashboard serves without errors across multiple
auto-refresh cycles against real streamed data, and fixed a Streamlit API
deprecation (`use_container_width` → `width="stretch"`) found in the process.

## Persistent storage (InfluxDB)

`src/timeseries_db.py` is the "Data storage" tech-stack item from the project
brief (it lists Firebase/InfluxDB/MongoDB — InfluxDB was chosen here because
it's purpose-built for exactly this data shape, free/self-hostable with no
cloud account needed for a student project demo, and has a simple Python
client). `mqtt_bridge.py` now writes every score, temperature reading, and
full feature vector to it, in addition to the flat files the dashboard reads
(those stay as-is — they're simpler for the dashboard's polling design and
act as a redundant local cache).

```bash
sudo apt-get install influxdb influxdb-client
pip install influxdb --break-system-packages
influxd          # start the server (foreground; use setsid/nohup to background)
```

No further setup needed — `mqtt_bridge.py` auto-creates the `pdm` database on
first connect. If InfluxDB isn't running, the bridge **fails soft**: it logs a
note and keeps doing live scoring/alerting with the flat files only, rather
than crashing (verified — killed InfluxDB mid-run, confirmed no crash, then
restarted it and confirmed writes resumed). Persisting the full feature
vectors (not just the anomaly score) also means the raw windowed data survives
for later offline analysis or retraining, not just the derived score.

Query examples (see `timeseries_db.py` for the full API):
```python
from timeseries_db import TimeSeriesDB
db = TimeSeriesDB()
db.query_recent_scores("PUMP_01", minutes=60)   # trend data
db.query_anomaly_history("PUMP_01", hours=24)   # maintenance-log style report
db.list_machines()
```

**Second real bug found and fixed while testing this** (first one was the
temperature-fallback bug in the MQTT bridge section above): the initial
version used the device's `t_ms`/`t0_ms` field (milliseconds elapsed since the
Arduino/simulator started — like Arduino's `millis()`, NOT wall-clock epoch
time) directly as the InfluxDB point timestamp. This silently wrote every
point to ~1970-01-01, making all of it invisible to any `time > now() - X`
query — confirmed by directly querying InfluxDB and seeing `1970-01-01`
timestamps despite the write call reporting success. **Fix:** use the
backend's own receipt time (`time.time()`) as the point timestamp instead,
keeping the device's relative timestamp as a stored field for reference/
debugging rather than for indexing. See the docstring on
`TimeSeriesDB._timestamp_ns()` for the full reasoning, including when you
*would* want to reintroduce device time (an NTP-synced Arduino clock on a WAN
link with real queueing delay).

## Email alerts

`mqtt_bridge.py` calls `on_alert()` once each time a machine enters a sustained anomaly.
It prints to the console, appends to `outputs/alert_log.jsonl`, and - if configured -
emails recipients (sent on a background thread so a slow mail server cannot stall
scoring). Configure with environment variables (never put credentials in code):

```powershell
# Windows PowerShell example (Gmail needs an App Password, not your normal password)
$env:PDM_SMTP_HOST="smtp.gmail.com"; $env:PDM_SMTP_PORT="587"
$env:PDM_SMTP_USER="you@gmail.com";  $env:PDM_SMTP_PASSWORD="your-app-password"
$env:PDM_ALERT_TO="you@gmail.com,teammate@example.com"
python mqtt_bridge.py --broker localhost
```
Without these variables the bridge simply keeps console-only alerts. `PDM_SMTP_SECURITY`
can be `starttls` (default), `ssl`, or `none` (local test servers only).

## Testing

`python -m unittest discover -s tests -v` runs 9 tests: feature extraction; a check that
the live (MQTT) path computes the same features as training; the detection-lag metric
(including that a single spike is not counted as detection); email delivery through an
in-process SMTP server; the bridge's alert hook; and the trained model on an unseen
simulated machine (<5% false alarms on healthy data, >95% detection once failed).

## Known limitations (state these openly in the report / viva)

- **Simulated data.** Headline synthetic results come from a physics-inspired simulator,
  not a factory. It now runs at 1600 Hz, so its ~124 Hz fault rate and ~236 Hz carrier are
  below Nyquist (at the old 200 Hz they were aliased). It is still easy: do not read its
  numbers as real-world performance. The ADXL345's 800 Hz bandwidth is still below the
  several-kHz resonance band typically used for bearing demodulation.
- **Window mismatch.** Training uses 50%-overlapping 1 s windows; the Arduino sends
  non-overlapping 1 s windows (with a ~50-150 ms gap), so a 15-window sequence spans ~7.5 s in
  training but ~15 s live.
  The live tests still worked, but this was not validated rigorously. Retraining with
  non-overlapping windows is the straightforward fix and is untested.
- **Brittle normalisation on synthetic data.** Simulated healthy temperature has a std of
  about 0.007 C, so tiny deviations become enormous z-scores and scores reach 1e5-1e6.
  Real sensors are noisier. Scores are relative: only above/below threshold is calibrated.
- **Threshold trade-off.** mean + 3 sigma gave one brief false-alert episode in the demo
  run (about 144 s). Raise the multiplier to trade detection speed for fewer false alerts.
- **Real-data weakness.** On CWRU the LSTM false-positive rate (23%) was worse than the
  baselines (3.5-14%), most likely because only ~35 s of real healthy data was available.
  Every method detected all 9 real fault conditions, so CWRU does not separate them.
- **Untested on hardware / real services:** Arduino R4 firmware, NASA IMS pipeline (fake data
  only), and email against a real provider.

## What still needs you

1. **Flash the Arduino UNO R4 WiFi** and check the Serial Monitor (wiring and libraries in
   `arduino_r4_firmware/README.md`; ArduinoJson is no longer required). Expect minor fixes.
2. **Download NASA IMS** (about 5 GB) and run `python src/validate_ims_data.py <folder> --channel <n>`
   for a real detection-lag number. Check the channel against IMS's readme.
3. **Test email with real credentials** (see above).
4. For a credible live demo, record a healthy baseline from the real machine and retrain
   (`src/main.py` shows how) instead of relying on the simulator's model.
