"""
payload_codec.py
-----------------
Binary vibration payload shared by the Arduino firmware (arduino_r4_firmware/
pdm_payload.h), the simulator and the bridge.

Why binary: a 1600-sample window as JSON text is ~11 KB; as raw int16 counts
it is 3.2 KB, which fits the R4's RAM and MQTT buffer and halves WiFi airtime.

Layout (little-endian), topic  pdm/<machine_id>/vib_bin :
    offset size  field
    0      4     magic  'P','D','M','1'
    4      2     uint16 sample_rate_hz
    6      2     uint16 n_samples
    8      4     uint32 t0_ms
    12     4     float32 scale_g   (g per count, e.g. 0.0039 for ADXL345 full-res)
    16     2*n   int16 samples (raw sensor counts, ONE axis)

The sensor's constant gravity offset is removed here (remove_dc=True) because
training data (CWRU, simulator) is zero-mean and a 1 g offset would otherwise
distort skewness / crest factor / kurtosis.
"""
import struct

import numpy as np

MAGIC = b"PDM1"
HEADER_FMT = "<4sHHIf"
HEADER_SIZE = struct.calcsize(HEADER_FMT)       # 16
ADXL345_SCALE_G = 0.0039                        # full-resolution mode, any range


def encode_binary_window(samples_g, sample_rate, t0_ms, scale_g=ADXL345_SCALE_G):
    counts = np.clip(np.round(np.asarray(samples_g, dtype=np.float64) / scale_g),
                     -32768, 32767).astype("<i2")
    return struct.pack(HEADER_FMT, MAGIC, int(sample_rate), len(counts),
                       int(t0_ms) & 0xFFFFFFFF, float(scale_g)) + counts.tobytes()


def decode_binary_window(payload, remove_dc=True):
    """Returns dict(sample_rate, n, t0_ms, samples[float32 g]). Raises ValueError."""
    if len(payload) < HEADER_SIZE:
        raise ValueError("payload shorter than header")
    magic, sr, n, t0, scale = struct.unpack_from(HEADER_FMT, payload, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if len(payload) != HEADER_SIZE + 2 * n:
        raise ValueError(f"length {len(payload)} != header+2*n ({HEADER_SIZE + 2 * n})")
    if sr == 0 or n == 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid header fields")
    counts = np.frombuffer(payload, dtype="<i2", count=n, offset=HEADER_SIZE)
    samples = counts.astype(np.float32) * np.float32(scale)
    if remove_dc:
        samples = samples - samples.mean()
    return {"sample_rate": int(sr), "n": int(n), "t0_ms": int(t0), "samples": samples}
