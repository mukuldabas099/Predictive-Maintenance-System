# Arduino UNO R4 WiFi firmware (ADXL345 version)

> **Status: written but NOT compiled or run on hardware.** The Arduino toolchain
> was not available when this was written. `pdm_payload.h` (the packet format)
> *is* unit-tested on a PC against the Python decoder (`tests/test_pipeline.py`),
> and the whole Python side was tested end to end with a simulator that sends the
> same packets. Treat the first flash as a bring-up and watch the Serial Monitor.

## What it does
Samples one axis of an **ADXL345** at **1600 Hz** (800 Hz bandwidth, hardware-timed
by the sensor and buffered in its FIFO), packs 1 s windows (1600 samples, 3216 bytes)
and publishes them to `pdm/<MACHINE_ID>/vib_bin`; publishes DS18B20 temperature to
`pdm/<MACHINE_ID>/temperature`. No analysis happens on the device.

## Wiring (SPI)
| ADXL345 | UNO R4 WiFi |
|---|---|
| CS | D10 |
| SDA / SDI (COPI) | D11 |
| SDO (CIPO) | D12 |
| SCL / SCLK | D13 |
| VCC, GND | 3.3 V / GND (or 5 V on boards with a regulator) |

**Level shifting:** the R4's I/O is 5 V; a bare ADXL345 tolerates 3.3 V. Use a
breakout with level shifting (e.g. Adafruit ADXL345) or add a shifter.
D13 is both SPI clock and the built-in LED, so the status LED is on **D7** (external).
DS18B20: data on D2 with a 4.7 kΩ pull-up to 3.3 V.

## Mounting matters more than the sensor
Fix the sensor rigidly (stud/epoxy) on the bearing housing, with the chosen axis
(`VIB_AXIS` in `config.h`) pointing at the bearing. A loose mount filters out the
high frequencies you bought the 1600 Hz rate for.

## Libraries (Library Manager)
OneWire (>= 2.3.8), DallasTemperature, PubSubClient. SPI and WiFiS3 come with the
"Arduino UNO R4 Boards" core.

## Configure and flash
Edit `config.h` (WiFi, broker IP, MQTT user/password, `MACHINE_ID`). Keep
`VIBRATION_SAMPLE_RATE_HZ` / `WINDOW_DURATION_S` equal to `src/pdm_config.py`.
Select board "Arduino UNO R4 WiFi", open the Serial Monitor at 115200 baud.

## Expected Serial output
`ADXL345 ready: 1600 Hz, +/-16 g full-resolution.` then `connected.` for MQTT. If you see
`ADXL345 not found (DEVID != 0xE5)`, check wiring/power and that CS is on D10.
`window dropped (FIFO overrun/timeout)` means samples were lost in a window; that
window is not sent. Occasional drops are fine; frequent ones mean something is
blocking the loop.

## Design notes and known limits
- A window is acquired with WiFi idle, then published, so consecutive windows have a
  gap of roughly 50-150 ms. The model works on per-window features, so this is fine,
  but it is not a gapless stream.
- The 32-sample FIFO holds only ~20 ms at 1600 Hz, so nothing slow may run during
  acquisition (OneWire/WiFi work happens between windows).
- The bridge removes the 1 g gravity offset; the device sends raw counts.
- Bandwidth is 800 Hz. Bearing-resonance demodulation normally wants several kHz;
  for serious bearing diagnostics use an analog ADXL1002-class sensor on the R4's ADC.
- `MQTT_USE_TLS 1` uses `WiFiSSLClient` and needs the broker CA installed on the
  WiFi module (untested here). Without TLS, username/password still travel in clear
  text on the LAN: keep the broker on a trusted network.
