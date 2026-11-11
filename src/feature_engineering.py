"""
feature_engineering.py
-----------------------
Converts raw high-frequency vibration + temperature streams into a sequence
of per-window feature vectors, combining:

  Time-domain features (signal shape / intensity):
    - RMS (root mean square)              -> overall vibration energy
    - Kurtosis                            -> "peakiness"; rises sharply when
                                              impulsive fault impacts appear
    - Skewness                            -> asymmetry of the signal
    - Peak-to-peak amplitude              -> max swing
    - Crest factor (peak / RMS)           -> impulsiveness independent of energy

  Frequency-domain features (via FFT):
    - Spectral energy (total power)       -> overall vibration power
    - Dominant frequency                  -> shaft/fault frequency shifts
    - Spectral centroid                   -> "center of mass" of the spectrum;
                                              shifts as fault harmonics appear
    - Spectral entropy                    -> how spread-out vs. concentrated
                                              the spectral energy is; faults
                                              often narrow/sharpen the spectrum
                                              at specific defect frequencies

  Plus mean temperature per window.

  Envelope / demodulation features (added in phase 2; the standard bearing
  diagnostics method). Faulty bearings produce short impacts that excite a
  high-frequency structural resonance; the impacts repeat at the fault
  frequency, so that repetition rate shows up in the ENVELOPE of the
  band-passed signal rather than in the raw spectrum:
    - Envelope kurtosis                   -> impulsiveness of the demodulated signal
    - Envelope spectrum peak ratio        -> strength of the strongest repetition
                                              rate vs. the spectrum's background
    - Envelope dominant frequency         -> that repetition rate (fault frequency)
    - High-frequency energy ratio         -> share of power in the upper half
                                              of the spectrum (resonance excitation)
  NOTE: these need a sample rate high enough to contain the resonance band
  (several kHz). At 200 Hz they still compute but carry little information.

These are computed over sliding windows (default: 2s windows, 50% overlap)
so that each feature vector represents a short snapshot of machine condition,
and consecutive vectors form a time series suitable for the LSTM autoencoder.
"""

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt, hilbert, stft


BASIC_FEATURE_NAMES = [
    "rms", "kurtosis", "skewness", "peak_to_peak", "crest_factor",
    "spectral_energy", "dominant_freq", "spectral_centroid", "spectral_entropy",
    "mean_temperature",
]
ENVELOPE_FEATURE_NAMES = [
    "env_kurtosis", "env_peak_ratio", "env_dominant_freq", "hf_energy_ratio",
]
# Envelope features are APPENDED so the first 10 columns keep their positions
# (mqtt_bridge looks up "mean_temperature" by name; baselines/ablations can
# slice [:, :len(BASIC_FEATURE_NAMES)]).
FEATURE_NAMES = BASIC_FEATURE_NAMES + ENVELOPE_FEATURE_NAMES


def _time_domain_features(x):
    rms = np.sqrt(np.mean(x ** 2))
    kurt = kurtosis(x, fisher=True)          # excess kurtosis (0 = Gaussian)
    sk = skew(x)
    p2p = x.max() - x.min()
    crest = (np.max(np.abs(x)) / rms) if rms > 1e-8 else 0.0
    return rms, kurt, sk, p2p, crest


def _frequency_domain_features(x, sample_rate):
    n = len(x)
    windowed = x * np.hanning(n)  # reduce spectral leakage
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(n, d=1.0 / sample_rate)

    power = spectrum ** 2
    total_energy = np.sum(power)
    if total_energy < 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    dominant_freq = freqs[np.argmax(power)]
    centroid = np.sum(freqs * power) / total_energy

    # Spectral entropy: normalize power to a probability distribution, then
    # compute Shannon entropy. Low entropy = energy concentrated at a few
    # frequencies (common once a defect frequency dominates).
    p = power / total_energy
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log2(p))
    entropy /= np.log2(len(p))  # normalize to [0, 1]

    return total_energy, dominant_freq, centroid, entropy



# ---------------------------------------------------------------- advanced
# Opt-in feature group (advanced=True). NOT part of FEATURE_NAMES, so the
# deployed 14-feature model is unaffected unless you retrain with it.
ADVANCED_FEATURE_NAMES = [
    "spec_kurt_max", "spec_kurt_mean",               # spectral kurtosis (impulsive bands)
    "wp_e0", "wp_e1", "wp_e2", "wp_e3",              # wavelet-packet band energy shares (low -> high)
    "bpfo_env_ratio", "bpfi_env_ratio", "bsf_env_ratio",  # envelope energy at fault freqs
]
# Fault-frequency multiples of shaft rate for the CWRU 6205-2RS drive-end bearing.
CWRU_BEARING_RATIOS = {"bpfo": 3.5848, "bpfi": 5.4152, "bsf": 4.7135}


def _spectral_kurtosis(x, sample_rate):
    n = len(x)
    # Frames must be SHORTER than the spacing between impacts or the impulsiveness
    # averages away: ~4 ms, but never fewer than 32 samples (freq. resolution).
    nperseg = int(min(max(32, int(0.004 * sample_rate)), max(n // 4, 8)))
    f, _, Z = stft(x, fs=sample_rate, nperseg=nperseg, noverlap=int(nperseg * 0.75))
    P2 = np.mean(np.abs(Z) ** 2, axis=1)
    P4 = np.mean(np.abs(Z) ** 4, axis=1)
    sk = P4 / (P2 ** 2 + 1e-18) - 2.0
    sk = sk[2:]                                       # drop DC bins
    if sk.size == 0 or not np.isfinite(sk).all():
        return 0.0, 0.0
    return float(sk.max()), float(sk.mean())


def _wavelet_packet_energies(x, wavelet="db4", level=2):
    import pywt
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=level)
    e = np.array([np.sum(node.data ** 2) for node in wp.get_level(level, order="freq")])
    tot = e.sum()
    return (e / tot) if tot > 1e-18 else np.zeros_like(e)


def _fault_band_ratios(x, sample_rate, shaft_hz, ratios, band_hz=None, tol=0.04):
    """Share of envelope-spectrum energy within +/-tol of each fault frequency
    and its 2nd harmonic. Needs the shaft speed; returns zeros without it."""
    out = [0.0] * len(ratios)
    if not shaft_hz or len(x) < 64 or np.max(np.abs(x)) < 1e-9:
        return out
    n, nyq = len(x), sample_rate / 2.0
    lo, hi = band_hz if band_hz else (0.25 * nyq, 0.90 * nyq)
    try:
        sos = butter(4, [lo, hi], btype="band", fs=sample_rate, output="sos")
        env = np.abs(hilbert(sosfiltfilt(sos, x)))
    except ValueError:
        return out
    env = env - env.mean()
    spec = np.abs(rfft(env * np.hanning(n))) ** 2
    freqs = rfftfreq(n, d=1.0 / sample_rate)
    f_min, f_max = 3.0 * sample_rate / n, min(0.25 * nyq, 1000.0)
    m = (freqs >= f_min) & (freqs <= f_max)
    total = spec[m].sum()
    if total <= 1e-18:
        return out
    res = []
    for r in ratios:
        f0, e = r * shaft_hz, 0.0
        for h in (1, 2):
            fc = h * f0
            if fc <= f_max:
                e += spec[(freqs >= fc * (1 - tol)) & (freqs <= fc * (1 + tol)) & m].sum()
        res.append(float(e / total))
    return res


def advanced_features(x, sample_rate, shaft_hz=None, bearing_ratios=None):
    """(9,) vector matching ADVANCED_FEATURE_NAMES."""
    x = np.asarray(x, dtype=np.float64)
    ratios = list((bearing_ratios or CWRU_BEARING_RATIOS).values())
    skm, skmean = _spectral_kurtosis(x, sample_rate)
    wp = _wavelet_packet_energies(x)
    fb = _fault_band_ratios(x, sample_rate, shaft_hz, ratios)
    return np.array([skm, skmean, *wp, *fb], dtype=np.float32)


def _envelope_features(x, sample_rate, band_hz=None):
    """Envelope-analysis features. band_hz: (low, high) pass band used before
    demodulation; default is 25%-90% of Nyquist, which for kHz-range sample
    rates covers the typical structural-resonance region."""
    n = len(x)
    nyq = sample_rate / 2.0
    lo, hi = band_hz if band_hz else (0.25 * nyq, 0.90 * nyq)
    if n < 64 or np.max(np.abs(x)) < 1e-9:
        return 0.0, 0.0, 0.0, 0.0
    try:
        sos = butter(4, [lo, hi], btype="band", fs=sample_rate, output="sos")
        xb = sosfiltfilt(sos, x)
    except ValueError:                       # filter unusable for this n / band
        return 0.0, 0.0, 0.0, 0.0
    env = np.abs(hilbert(xb))
    env = env - env.mean()
    env_kurt = float(kurtosis(env, fisher=True))

    spec = np.abs(rfft(env * np.hanning(n)))
    freqs = rfftfreq(n, d=1.0 / sample_rate)
    f_min = 3.0 * sample_rate / n            # skip DC / lowest bins
    f_max = min(0.25 * nyq, 1000.0)          # fault repetition rates live here
    m = (freqs >= f_min) & (freqs <= f_max)
    if m.sum() < 4:
        return env_kurt, 0.0, 0.0, 0.0
    band_spec = spec[m]
    peak_ratio = float(band_spec.max() / (np.median(band_spec) + 1e-12))
    env_dom = float(freqs[m][np.argmax(band_spec)])

    power = np.abs(rfft(x * np.hanning(n))) ** 2
    total = power.sum()
    hf_ratio = float(power[freqs >= 0.5 * nyq].sum() / total) if total > 1e-12 else 0.0
    return env_kurt, peak_ratio, env_dom, hf_ratio


def extract_windowed_features(run, window_s=2.0, overlap=0.5, extended=True,
                              advanced=False, shaft_hz=None, bearing_ratios=None):
    """
    run: dict from synthetic_data.generate_run() (or a real-data loader with
         the same keys: time, vibration, temperature, sample_rate, and
         optionally state/severity for evaluation labels).

    Returns:
      features: (n_windows, n_features) array, columns per FEATURE_NAMES
                (or BASIC_FEATURE_NAMES only if extended=False)
      window_times: (n_windows,) center time of each window, seconds
      window_labels: (n_windows,) majority `state` in each window (0/1/2),
                     or None if run has no "state" key
    """
    sr = run["sample_rate"]
    vib = run["vibration"]
    temp = run["temperature"]
    win_len = int(window_s * sr)
    step = int(win_len * (1 - overlap))
    has_labels = "state" in run

    feats, times, labels = [], [], []
    for start in range(0, len(vib) - win_len + 1, step):
        end = start + win_len
        x = vib[start:end]
        rms, kurt, sk, p2p, crest = _time_domain_features(x)
        energy, dom_freq, centroid, entropy = _frequency_domain_features(x, sr)
        mean_temp = float(np.mean(temp[start:end]))

        row = [rms, kurt, sk, p2p, crest, energy, dom_freq, centroid, entropy, mean_temp]
        if extended:
            row += list(_envelope_features(x, sr))
        if advanced:
            row += list(advanced_features(x, sr, shaft_hz, bearing_ratios))
        feats.append(row)
        times.append(run["time"][start + win_len // 2])
        if has_labels:
            window_states = run["state"][start:end]
            labels.append(int(np.round(np.mean(window_states))))

    features = np.array(feats, dtype=np.float32)
    window_times = np.array(times, dtype=np.float32)
    window_labels = np.array(labels, dtype=int) if has_labels else None
    return features, window_times, window_labels


def extract_single_window_features(vibration_samples, sample_rate, mean_temperature,
                                   extended=True, advanced=False, shaft_hz=None,
                                   bearing_ratios=None):
    """
    Same feature set as extract_windowed_features(), but for a single
    already-segmented window of samples (e.g. one MQTT batch arriving live
    from the Arduino R4), rather than slicing a long pre-recorded run.

    vibration_samples: 1D array-like of raw vibration samples for this window
    sample_rate: Hz
    mean_temperature: single temperature reading (or window average) in C

    Returns a (14,) feature vector matching FEATURE_NAMES (or (10,) matching
    BASIC_FEATURE_NAMES with extended=False).
    """
    x = np.asarray(vibration_samples, dtype=np.float32)
    rms, kurt, sk, p2p, crest = _time_domain_features(x)
    energy, dom_freq, centroid, entropy = _frequency_domain_features(x, sample_rate)
    row = [rms, kurt, sk, p2p, crest, energy, dom_freq, centroid, entropy, mean_temperature]
    if extended:
        row += list(_envelope_features(x, sample_rate))
    if advanced:
        row += list(advanced_features(x, sample_rate, shaft_hz, bearing_ratios))
    return np.array(row, dtype=np.float32)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from synthetic_data import generate_run

    run = generate_run(duration_s=600, sample_rate=200)
    X, t, y = extract_windowed_features(run)
    print(f"Feature matrix: {X.shape}  (windows x features)")
    print(f"Feature names: {FEATURE_NAMES}")
    print(f"Label distribution: healthy={np.sum(y==0)}, degrading={np.sum(y==1)}, failed={np.sum(y==2)}")
    print("\nSample healthy window features:\n", X[0])
    print("\nSample failed window features:\n", X[y == 2][0] if np.any(y == 2) else "none")
