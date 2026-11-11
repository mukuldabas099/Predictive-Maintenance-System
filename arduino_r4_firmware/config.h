/*
 * config.h — edit these before flashing (Arduino UNO R4 WiFi + ADXL345)
 */
#ifndef CONFIG_H
#define CONFIG_H

// ---- WiFi ----
// The UNO R4 WiFi's radio (ESP32-S3 co-processor) is 2.4 GHz ONLY.
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"

// ---- MQTT broker ----
// Use the IP address of the PC running Mosquitto (same WiFi network), e.g.
// "192.168.1.50". Do NOT use a public broker for real data: anyone can read it.
// The provided deploy/mosquitto config requires a username/password and
// supports TLS on port 8883 (see deploy/README.md).
#define MQTT_BROKER     "192.168.1.50"
#define MQTT_USE_TLS    0        // 1 = TLS via WiFiSSLClient (port 8883). The broker's
                                 // CA must be installed on the R4's WiFi module first
                                 // (Arduino IDE: Tools > WiFi firmware updater > upload
                                 // SSL root certificates). Untested on hardware.
#if MQTT_USE_TLS
  #define MQTT_PORT     8883
#else
  #define MQTT_PORT     1883
#endif
#define MQTT_USER       "pdm_device"
#define MQTT_PASSWORD   "CHANGE_ME"

// ---- Machine identity ----
// Change per physical machine so units don't collide on MQTT topics.
#define MACHINE_ID      "PUMP_01"

// ---- Pins (Arduino UNO R4 WiFi) ----
// ADXL345 on hardware SPI (use a breakout with level shifting, e.g. Adafruit
// ADXL345 #1231 -- the R4's I/O is 5 V, the bare ADXL345 is 3.3 V max):
//   CS->D10, SDA/SDI(COPI)->D11, SDO(CIPO)->D12, SCL/SCLK->D13, VCC->3.3V/5V per board, GND->GND
#define ADXL345_CS_PIN  10
#define VIB_AXIS        2        // which axis to stream: 0=X, 1=Y, 2=Z. Pick the axis
                                 // pointing radially at the bearing (strongest signal).
#define ONE_WIRE_PIN    2        // DS18B20 data pin = D2 (4.7k pull-up to 3.3V)
#define STATUS_LED_PIN  7        // external LED + resistor on D7. NOT D13 (LED_BUILTIN):
                                 // D13 is the SPI clock and would flicker/conflict.

// ---- Sampling ----
// ADXL345 hardware-timed output rate: 1600 Hz -> 800 Hz bandwidth (Nyquist).
// MUST match pdm_config.py (SAMPLE_RATE_HZ, WINDOW_S) in the Python pipeline.
#define VIBRATION_SAMPLE_RATE_HZ  1600
#define WINDOW_DURATION_S         1.0
#define ADXL345_SCALE_G           0.0039f   // g per count, full-resolution mode (all ranges)
#define SPI_CLOCK_HZ              4000000   // datasheet: >= 2 MHz needed at 1600 Hz ODR
#define TEMPERATURE_READ_INTERVAL_MS  1000

// ---- Bandwidth saving ----
// Stats (rms, peak-to-peak, crest, kurtosis, skewness) are computed on the device
// every window and sent to pdm/<id>/stats (~100 bytes). The raw 3.2 KB window is
// sent only every RAW_EVERY_N_WINDOWS-th cycle (1 = every window, the default).
// The model scores the RAW windows, so N > 1 means N times fewer scores: choose
// it knowing detection latency grows N-fold.
#define PUBLISH_STATS            1
#define RAW_EVERY_N_WINDOWS      1

#endif
