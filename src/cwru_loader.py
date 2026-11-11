"""
cwru_loader.py
----------------
Loads real Case Western Reserve University (CWRU) Bearing Data Center .mat
files into the same run dict shape used everywhere else in this pipeline
(see synthetic_data.py docstring), so feature_engineering.py,
lstm_autoencoder.py and baselines.py work UNCHANGED on real data.

CWRU specifics handled here:
  - Each .mat file is a short (~5s at 48kHz) FIXED-CONDITION recording (either
    healthy baseline, or one fault type at one severity/load) -- NOT a
    continuous run-to-failure trace like the synthetic data or NASA IMS.
    So there is no meaningful "state" ramp within a single file; each file
    is entirely healthy or entirely one fault condition.
  - No temperature channel exists in CWRU -> temperature is filled with
    zeros. (Real deployments would still have the DS18B20 channel; this is
    purely a limitation of this particular benchmark dataset.)
  - Variable names inside the .mat follow the pattern 'X<id>_DE_time' for
    drive-end accelerometer data.
"""

import os
import re
import numpy as np
import scipy.io as sio


def load_mat_vibration(path, channel="DE"):
    """Load the drive-end (or fan-end) accelerometer channel from a CWRU .mat file.

    Some CWRU files contain MORE THAN ONE signal of the same channel (e.g.
    99_Normal_2.mat holds a stray copy of file 98's 'X098_DE_time' before its
    own 'X099_DE_time'). Taking the first match silently loads the wrong
    recording, so we prefer the key whose number matches the file's own id,
    and only fall back to the first match when the id isn't in the filename
    (or the file uses a different numbering, as the 3001-3008 files do)."""
    mat = sio.loadmat(path)
    keys = [k for k in mat if re.search(rf"_{channel}_time$", k)]
    if not keys:
        raise KeyError(f"no *_{channel}_time variable in {path}")
    m = re.match(r"(\d+)", os.path.basename(path))
    preferred = f"X{int(m.group(1)):03d}_{channel}_time" if m else None
    key = preferred if preferred in keys else keys[0]
    return mat[key].flatten().astype(np.float32)


def load_cwru_run(path, sample_rate=48000, state=0, channel="DE"):
    """
    Load a single CWRU .mat file as a run dict.
    state: 0 for healthy/normal files, 2 for fault files (no "degrading"
           category exists in this fixed-condition dataset -- see module docstring).
    """
    vib = load_mat_vibration(path, channel=channel)
    n = len(vib)
    return {
        "time": np.arange(n) / sample_rate,
        "vibration": vib,
        "temperature": np.zeros(n, dtype=np.float32),  # not available in CWRU
        "state": np.full(n, state, dtype=int),
        "sample_rate": sample_rate,
    }


def concat_runs(paths, sample_rate=48000, state=0, channel="DE"):
    """Concatenate several healthy files end-to-end into one longer 'run',
    for building enough healthy training data (CWRU only has 4 baseline files)."""
    vibs = [load_mat_vibration(p, channel=channel) for p in paths]
    vib = np.concatenate(vibs)
    n = len(vib)
    return {
        "time": np.arange(n) / sample_rate,
        "vibration": vib,
        "temperature": np.zeros(n, dtype=np.float32),
        "state": np.full(n, state, dtype=int),
        "sample_rate": sample_rate,
    }


if __name__ == "__main__":
    import glob
    base = os.path.join(os.path.dirname(__file__), "..", "..", "cwru_data")
    normal_files = sorted(glob.glob(os.path.join(base, "Normal", "*.mat")))
    print(f"Found {len(normal_files)} normal files: {[os.path.basename(f) for f in normal_files]}")
    run = concat_runs(normal_files)
    print(f"Concatenated healthy run: {len(run['vibration'])} samples, "
          f"{run['time'][-1]:.2f}s at {run['sample_rate']} Hz")
    print(f"Vibration range: [{run['vibration'].min():.3f}, {run['vibration'].max():.3f}]")
