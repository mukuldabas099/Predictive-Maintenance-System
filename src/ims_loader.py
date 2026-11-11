"""
ims_loader.py
----------------
Loader for the NASA IMS Bearing Dataset (Prognostics Center of Excellence),
the one public dataset that can actually validate this project's "detection
lag" claim -- unlike CWRU (see validate_real_data.py's docstring), IMS
contains genuine continuous run-to-failure recordings: a bearing sampled
every 10 minutes, for days, until it physically fails.

*** THIS FILE IS WRITTEN AND READY BUT UNTESTED. ***
I could not download the dataset myself: it's ~5-6GB, hosted at
data.nasa.gov / phm-datasets.s3.amazonaws.com, neither of which is reachable
from this sandboxed environment's network allowlist. See the main README's
"NASA IMS validation" section for exactly what to do to finish this.

Expected raw format (this is what NASA's download unpacks to):
  IMS/
    1st_test/1st_test/
      2003.10.22.12.06.24     <- one file per 10-min snapshot, no extension
      2003.10.22.12.09.13
      ...
    2nd_test/2nd_test/
      2004.02.12.10.32.39
      ...
    3rd_test/4th_test/txt/
      ...
Each snapshot file is plain text, whitespace-separated columns, one column
per accelerometer channel, 20480 rows (1 second at 20kHz). 1st_test has 8
channels (2 per bearing x 4 bearings), 2nd/3rd_test have 4 channels (1 per
bearing). Filenames ARE timestamps ("YYYY.MM.DD.HH.MM.SS") -- that's the only
place the real time-since-start comes from, since each snapshot is only 1
second of a much longer gap between snapshots (10 min).
"""

import glob
import os
from datetime import datetime

import numpy as np


def _parse_filename_timestamp(path):
    name = os.path.basename(path)
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")


def load_ims_run(test_dir, bearing_channel=0, sample_rate=20000):
    """
    Build a continuous "run" dict (same shape as synthetic_data.py /
    cwru_loader.py) out of an IMS test folder's many 1-second snapshots.

    test_dir: path to e.g. ".../1st_test/1st_test/" containing the snapshot files
    bearing_channel: which column (0-indexed) to use as the vibration signal
                      -- pick the channel for the bearing that's documented to
                      have failed in that test (see IMS's Readme Document.txt,
                      it names which bearing/channel failed in each test)
    sample_rate: 20000 Hz for all three IMS tests

    IMPORTANT CAVEAT (read before trusting detection-lag numbers from this):
    consecutive snapshots are 10 MINUTES apart but each is only 1 SECOND long
    -- i.e. ~99.8% of the actual timeline has no data at all. This function
    stitches snapshots back-to-back with a `time` array that reflects the
    REAL elapsed time between snapshots (so detection lag comes out in real
    units), but the LSTM Autoencoder sequence model will effectively be
    learning across a huge number of missing gaps, not a truly continuous
    signal. This is a property of the dataset, not a fix-able bug -- just
    something to caveat in the report alongside the detection-lag number,
    the same way the "small healthy training set" caveat was needed for CWRU.
    """
    files = sorted(glob.glob(os.path.join(test_dir, "*")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {test_dir}")

    timestamps = [_parse_filename_timestamp(f) for f in files]
    t0 = timestamps[0]

    vib_chunks = []
    time_chunks = []
    for f, ts in zip(files, timestamps):
        data = np.loadtxt(f)  # (20480, n_channels)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        channel_data = data[:, bearing_channel].astype(np.float32)

        elapsed_s = (ts - t0).total_seconds()
        t = elapsed_s + np.arange(len(channel_data)) / sample_rate

        vib_chunks.append(channel_data)
        time_chunks.append(t)

    vibration = np.concatenate(vib_chunks)
    time_arr = np.concatenate(time_chunks)

    # No temperature channel in IMS either (same limitation as CWRU).
    temperature = np.zeros_like(vibration)

    # No ground-truth state labels are provided by IMS itself -- failure is
    # only documented as "which bearing, confirmed at the END of the test"
    # (see the dataset's Readme Document.txt). A defensible proxy for
    # ground-truth degradation onset: the first snapshot where this
    # channel's RMS exceeds, say, mean + 3*std of the first 20% of snapshots
    # (i.e. the same statistical idea as the LSTM threshold, but applied to
    # raw RMS as an independent ground-truth signal). See
    # `estimate_degradation_onset()` below -- compute this from the DATA,
    # don't hand-pick a state array like the synthetic generator does.
    state = np.zeros(len(vibration), dtype=int)  # placeholder; see note above

    return {
        "time": time_arr,
        "vibration": vibration,
        "temperature": temperature,
        "state": state,  # NOT populated -- see estimate_degradation_onset()
        "sample_rate": sample_rate,
        "snapshot_times": timestamps,  # kept for estimate_degradation_onset()
    }


def estimate_degradation_onset(run, snapshot_len=20480, baseline_frac=0.2, sigma=3.0):
    """
    IMS provides no ground-truth "degradation started here" label (only
    "failure confirmed at the end"), so detection lag can't be computed
    against a labeled onset the way the synthetic/CWRU runs can. This
    provides a defensible DATA-DRIVEN proxy: per-snapshot RMS, threshold at
    mean + sigma*std of the first `baseline_frac` of snapshots (assumed
    healthy), first snapshot exceeding it = estimated onset.

    Returns (onset_time_s, per_snapshot_rms, per_snapshot_time).
    IMPORTANT: report this as "estimated onset (RMS-based proxy)", not as
    ground truth -- it's a reasonable stand-in, not a documented label.
    """
    n_snapshots = len(run["vibration"]) // snapshot_len
    rms = np.array([
        np.sqrt(np.mean(run["vibration"][i * snapshot_len:(i + 1) * snapshot_len] ** 2))
        for i in range(n_snapshots)
    ])
    snapshot_times = np.array([
        run["time"][i * snapshot_len] for i in range(n_snapshots)
    ])

    n_baseline = max(1, int(n_snapshots * baseline_frac))
    baseline_mean = rms[:n_baseline].mean()
    baseline_std = rms[:n_baseline].std()
    threshold = baseline_mean + sigma * baseline_std

    onset_idx = np.argmax(rms > threshold)
    if rms[onset_idx] <= threshold:
        onset_time = None  # never exceeded -- no clear onset detected by this proxy
    else:
        onset_time = float(snapshot_times[onset_idx])

    return onset_time, rms, snapshot_times


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 ims_loader.py /path/to/IMS/1st_test/1st_test [channel]")
        print("(dataset must be downloaded manually first -- see README)")
        sys.exit(1)

    test_dir = sys.argv[1]
    channel = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Loading IMS run from {test_dir}, channel {channel}...")
    run = load_ims_run(test_dir, bearing_channel=channel)
    print(f"  {len(run['vibration'])} samples, {run['time'][-1]/3600:.1f} hours total span")

    onset_time, rms, snapshot_times = estimate_degradation_onset(run)
    if onset_time is not None:
        print(f"  Estimated degradation onset: {onset_time/3600:.1f} hours "
              f"(RMS-based proxy, {onset_time/run['time'][-1]*100:.1f}% through the run)")
    else:
        print("  No clear degradation onset detected by the RMS proxy -- "
              "try a different channel/bearing, or check this test actually failed.")
