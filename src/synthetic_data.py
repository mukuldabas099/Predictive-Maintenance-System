"""
synthetic_data.py
------------------
Generates synthetic accelerometer (vibration) + temperature data that mimics
real rolling-element bearing degradation, for prototyping the PdM pipeline
without depending on external dataset downloads.

Physical basis (matches what CWRU / NASA IMS bearing datasets actually show):
  - Healthy bearing: low-amplitude, near-stationary vibration; broadband noise
    with energy concentrated at shaft rotation frequency and its harmonics.
  - Developing fault (e.g. early spall/pit on race or ball): amplitude RMS
    creeps up, kurtosis rises (signal becomes more "impulsive" / peaky) as
    periodic impact transients appear, and energy starts appearing at the
    fault's characteristic defect frequency (here modeled as ~4.2x shaft rate,
    typical of a ball-pass-frequency-outer-race fault).
  - Severe fault: amplitude and impulsiveness increase sharply, frequency
    content broadens/shifts further, temperature rises due to friction.

>>> TO SWAP IN REAL DATA (CWRU / NASA IMS bearing datasets):
Replace `generate_run()` with a loader that reads the .mat files and returns
a dict with the same keys: {"time", "vibration", "temperature", "state",
"sample_rate"}. Everything downstream (feature_engineering.py,
lstm_autoencoder.py, baselines.py) only depends on that dict shape, so no
other code needs to change. CWRU files: engineering.case.edu/bearingdatacenter
NASA IMS: data.nasa.gov (search "IMS Bearing Dataset").
"""

import numpy as np


def _bearing_signal(t, shaft_hz, fault_hz, fault_severity, rng):
    """One instant of vibration signal: shaft harmonics + fault impacts + noise."""
    # Baseline rotational vibration (always present, low amplitude)
    base = (
        0.15 * np.sin(2 * np.pi * shaft_hz * t)
        + 0.07 * np.sin(2 * np.pi * 2 * shaft_hz * t + 0.3)
        + 0.03 * np.sin(2 * np.pi * 3 * shaft_hz * t + 0.6)
    )

    # Fault-induced impulsive component: periodic impacts at the defect
    # frequency, amplitude modulated by severity. Impacts are short pulses
    # (modeled as a narrow raised-cosine train) which is what produces the
    # high kurtosis / peaky signature characteristic of bearing faults.
    impact_period = 1.0 / fault_hz
    phase = np.mod(t, impact_period) / impact_period  # 0..1 within each cycle
    pulse_width = 0.06
    pulse = np.where(
        phase < pulse_width,
        np.cos(np.pi * (phase / pulse_width - 0.5)) ** 2,
        0.0,
    )
    fault_component = fault_severity * 1.8 * pulse * np.sin(2 * np.pi * (8 * shaft_hz) * t)

    # Broadband measurement noise
    noise = rng.normal(0, 0.05 + 0.02 * fault_severity, size=t.shape)

    return base + fault_component + noise


def generate_run(
    duration_s=3600,
    sample_rate=200,          # Hz, downsampled realistic accelerometer rate for this demo
    shaft_hz=29.5,             # ~1770 RPM, typical induction-motor shaft speed
    fault_ball_pass_ratio=4.2, # BPFO-like ratio (defect freq / shaft freq)
    degradation_start_frac=0.55,  # fraction of run where degradation begins
    failure_frac=0.95,         # fraction of run where "failure" threshold is reached
    seed=42,
):
    """
    Generate one full run: healthy operation -> gradual degradation -> failure.

    Returns dict: time (s), vibration (g), temperature (C), state (0=healthy,
    1=degrading, 2=failed), severity (0..1 continuous ground truth), sample_rate.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    frac = t / duration_s

    # Severity ramps 0 -> 1 smoothly after degradation_start_frac
    severity = np.clip(
        (frac - degradation_start_frac) / (failure_frac - degradation_start_frac), 0, 1
    )
    severity = severity ** 1.5  # nonlinear (fast-then-faster) growth, realistic for spalling faults

    fault_hz = shaft_hz * fault_ball_pass_ratio
    vibration = _bearing_signal(t, shaft_hz, fault_hz, severity, rng)

    # Temperature: slow thermal lag behind mechanical severity + ambient noise
    # (friction heat doesn't show up instantly; simple first-order lag)
    temp_target = 35 + 25 * severity  # deg C, ambient ~35 -> ~60 at failure
    temperature = np.zeros(n)
    temperature[0] = 35.0
    alpha = 1.0 / (sample_rate * 30)  # ~30s thermal time constant
    for i in range(1, n):
        temperature[i] = temperature[i - 1] + alpha * (temp_target[i] - temperature[i - 1])
    temperature += rng.normal(0, 0.15, size=n)

    state = np.zeros(n, dtype=int)
    state[severity > 0] = 1
    state[severity > 0.8] = 2

    return {
        "time": t,
        "vibration": vibration.astype(np.float32),
        "temperature": temperature.astype(np.float32),
        "state": state,
        "severity": severity.astype(np.float32),
        "sample_rate": sample_rate,
    }


def generate_fleet(n_machines=3, seed=0, **kwargs):
    """Generate several machine runs with slightly different shaft speeds / noise,
    useful for training on 'healthy fleet' data and testing on a held-out unit."""
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_machines):
        shaft_hz = kwargs.get("shaft_hz", 29.5) + rng.normal(0, 0.8)
        runs.append(generate_run(seed=seed + i, shaft_hz=shaft_hz, **{k: v for k, v in kwargs.items() if k != "shaft_hz"}))
    return runs


if __name__ == "__main__":
    run = generate_run(duration_s=600, sample_rate=200)
    print("Generated run:")
    print(f"  samples: {len(run['time'])}, duration: {run['time'][-1]:.1f}s")
    print(f"  vibration range: [{run['vibration'].min():.3f}, {run['vibration'].max():.3f}] g")
    print(f"  temperature range: [{run['temperature'].min():.1f}, {run['temperature'].max():.1f}] C")
    print(f"  state distribution: healthy={np.sum(run['state']==0)}, "
          f"degrading={np.sum(run['state']==1)}, failed={np.sum(run['state']==2)}")
