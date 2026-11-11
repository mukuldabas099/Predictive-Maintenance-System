"""
pdm_config.py -- single source of truth for the signal-acquisition settings.
MUST match arduino_r4_firmware/config.h (VIBRATION_SAMPLE_RATE_HZ, WINDOW_DURATION_S).

1600 Hz is the ADXL345's output rate with 800 Hz bandwidth; 1 s windows give a
1 Hz FFT resolution and 1600 samples (3.2 KB as int16) per MQTT message.
"""
SAMPLE_RATE_HZ = 1600
WINDOW_S = 1.0
WINDOW_OVERLAP = 0.5     # training windows only; the live bridge uses non-overlapping messages
