/*
 * arduino_r4_firmware.ino
 * -----------------------
 * IoT sensing layer (Layer 1) for the predictive maintenance project,
 * ARDUINO UNO R4 WiFi + ADXL345 accelerometer (SPI) + DS18B20 temperature.
 * Does NO analysis on-device: it streams windows of raw vibration over MQTT.
 *
 * What changed vs. the MPU6050 version (see README):
 *   - 1600 Hz hardware-timed sampling (800 Hz bandwidth) instead of 200 Hz,
 *     using the ADXL345's own sample clock + 32-deep FIFO (no software timing
 *     jitter).
 *   - ONE axis of raw counts (no gravity-magnitude): the bridge removes DC.
 *   - Compact binary payload (pdm_payload.h): 3216 bytes per 1 s window.
 *   - Windows are acquired back-to-back with the radio idle, then published,
 *     so the FIFO never overflows during a publish. Consequence: consecutive
 *     windows are separated by a short gap (~50-150 ms). Overrun windows are
 *     detected and dropped, not sent.
 *
 * NOT compiled or run on hardware by the author of this change (no Arduino
 * toolchain available). pdm_payload.h IS unit-tested on a PC. Treat the first
 * flash as a bring-up: open Serial Monitor @115200 and check the messages.
 *
 * Required libraries (Library Manager): OneWire (>= 2.3.8), DallasTemperature,
 * PubSubClient. SPI and WiFiS3 ship with the "Arduino UNO R4 Boards" core.
 * (No ADXL345 library needed: the register-level driver is below.)
 *
 * MQTT topics:
 *   pdm/<MACHINE_ID>/vib_bin      - binary window (see pdm_payload.h)
 *   pdm/<MACHINE_ID>/temperature  - JSON, ~1 Hz
 *   pdm/<MACHINE_ID>/status       - "online"/"offline" (retained, with LWT)
 */

#include <WiFiS3.h>
#include <PubSubClient.h>
#include <SPI.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#include "config.h"
#include "pdm_payload.h"
#include "pdm_features.h"

// ---------------------------------------------------------------------
// ADXL345 register-level SPI driver
// ---------------------------------------------------------------------
#define ADXL_REG_DEVID       0x00
#define ADXL_REG_BW_RATE     0x2C
#define ADXL_REG_POWER_CTL   0x2D
#define ADXL_REG_INT_SOURCE  0x30
#define ADXL_REG_DATA_FORMAT 0x31
#define ADXL_REG_DATAX0      0x32
#define ADXL_REG_FIFO_CTL    0x38
#define ADXL_REG_FIFO_STATUS 0x39
#define ADXL_BW_RATE_1600HZ  0x0E   // 1600 Hz output rate, 800 Hz bandwidth
#define ADXL_INT_OVERRUN     0x01

static SPISettings adxlSpi(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE3);  // ADXL345 needs mode 3

static void adxlWrite(uint8_t reg, uint8_t val) {
  SPI.beginTransaction(adxlSpi);
  digitalWrite(ADXL345_CS_PIN, LOW);
  SPI.transfer(reg & 0x3F);
  SPI.transfer(val);
  digitalWrite(ADXL345_CS_PIN, HIGH);
  SPI.endTransaction();
}

static uint8_t adxlRead(uint8_t reg) {
  SPI.beginTransaction(adxlSpi);
  digitalWrite(ADXL345_CS_PIN, LOW);
  SPI.transfer(0x80 | reg);
  uint8_t v = SPI.transfer(0x00);
  digitalWrite(ADXL345_CS_PIN, HIGH);
  SPI.endTransaction();
  return v;
}

// Reads one X/Y/Z sample from the FIFO output registers (multi-byte read).
static void adxlReadXYZ(int16_t* xyz) {
  uint8_t b[6];
  SPI.beginTransaction(adxlSpi);
  digitalWrite(ADXL345_CS_PIN, LOW);
  SPI.transfer(0x80 | 0x40 | ADXL_REG_DATAX0);   // read + multi-byte
  for (int i = 0; i < 6; i++) b[i] = SPI.transfer(0x00);
  digitalWrite(ADXL345_CS_PIN, HIGH);
  SPI.endTransaction();
  for (int i = 0; i < 3; i++) xyz[i] = (int16_t)((b[2 * i + 1] << 8) | b[2 * i]);
}

static bool adxlBegin() {
  pinMode(ADXL345_CS_PIN, OUTPUT);
  digitalWrite(ADXL345_CS_PIN, HIGH);
  SPI.begin();
  if (adxlRead(ADXL_REG_DEVID) != 0xE5) return false;
  adxlWrite(ADXL_REG_POWER_CTL, 0x00);                 // standby while configuring
  adxlWrite(ADXL_REG_BW_RATE, ADXL_BW_RATE_1600HZ);
  adxlWrite(ADXL_REG_DATA_FORMAT, 0x0B);               // FULL_RES, +/-16 g, 4-wire SPI
  adxlWrite(ADXL_REG_FIFO_CTL, 0x80);                  // stream mode
  adxlWrite(ADXL_REG_POWER_CTL, 0x08);                 // measurement on
  return true;
}

static void adxlResetFifo() {
  adxlWrite(ADXL_REG_FIFO_CTL, 0x00);                  // bypass: clears FIFO
  adxlWrite(ADXL_REG_FIFO_CTL, 0x80);                  // back to stream
  adxlRead(ADXL_REG_INT_SOURCE);                       // reading clears overrun flag
}

// ---------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------
#if MQTT_USE_TLS
WiFiSSLClient wifiClient;
#else
WiFiClient wifiClient;
#endif
PubSubClient mqttClient(wifiClient);

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature tempSensor(&oneWire);

const int SAMPLES_PER_WINDOW = (int)(VIBRATION_SAMPLE_RATE_HZ * WINDOW_DURATION_S);
const unsigned long WINDOW_TIMEOUT_US =
    (unsigned long)(WINDOW_DURATION_S * 1000000.0 * 2.0);  // give up after 2x window time

int16_t vibCounts[SAMPLES_PER_WINDOW];
uint8_t packet[PDM_HEADER_BYTES + 2 * SAMPLES_PER_WINDOW];

unsigned long lastTempReadMillis = 0;
unsigned long droppedWindows = 0;

char mqttTopicVibration[64];
char mqttTopicTemperature[64];
char mqttTopicStatus[64];
char mqttTopicStats[64];
unsigned long windowCounter = 0;

// ---------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && millis() - t < 3000) { }

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  snprintf(mqttTopicVibration, sizeof(mqttTopicVibration), "pdm/%s/vib_bin", MACHINE_ID);
  snprintf(mqttTopicTemperature, sizeof(mqttTopicTemperature), "pdm/%s/temperature", MACHINE_ID);
  snprintf(mqttTopicStatus, sizeof(mqttTopicStatus), "pdm/%s/status", MACHINE_ID);
  snprintf(mqttTopicStats, sizeof(mqttTopicStats), "pdm/%s/stats", MACHINE_ID);

  setupSensors();   // sensors first: fail fast on wiring problems
  setupWiFi();
  setupMQTT();
  Serial.println("Setup complete. Streaming started.");
}

void setupWiFi() {
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("ERROR: WiFi module not found (is this an UNO R4 WiFi?).");
    while (1) { delay(1000); }
  }
  Serial.print("Connecting to WiFi '");
  Serial.print(WIFI_SSID);
  Serial.print("'");
  while (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    for (int i = 0; i < 10 && WiFi.status() != WL_CONNECTED; i++) {
      delay(500);
      Serial.print(".");
    }
  }
  Serial.print("\nWiFi connected. IP: ");
  Serial.println(WiFi.localIP());
}

void setupMQTT() {
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setBufferSize(sizeof(packet) + 128);   // header + samples + topic + MQTT overhead
  connectMQTT();
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi lost, reconnecting...");
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      delay(3000);
      continue;
    }
    Serial.print("Connecting to MQTT broker...");
    String clientId = String("r4wifi-") + MACHINE_ID + "-" + String(random(0xffff), HEX);
    bool ok;
    if (strlen(MQTT_USER) > 0) {
      ok = mqttClient.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD,
                              mqttTopicStatus, 1, true, "offline");
    } else {
      ok = mqttClient.connect(clientId.c_str(), mqttTopicStatus, 1, true, "offline");
    }
    if (ok) {
      Serial.println(" connected.");
      mqttClient.publish(mqttTopicStatus, "online", true);
      digitalWrite(STATUS_LED_PIN, HIGH);
    } else {
      Serial.print(" failed (rc=");
      Serial.print(mqttClient.state());
      Serial.println("), retrying in 2s");
      delay(2000);
    }
  }
}

void setupSensors() {
  Serial.println("Initializing ADXL345 (SPI)...");
  if (!adxlBegin()) {
    Serial.println("ERROR: ADXL345 not found (DEVID != 0xE5). Check CS/COPI/CIPO/SCLK wiring,");
    Serial.println("       power, and that the breakout is in SPI mode (CS tied to the CS pin).");
    while (1) { delay(1000); }
  }
  Serial.println("ADXL345 ready: 1600 Hz, +/-16 g full-resolution.");

  Serial.println("Initializing DS18B20...");
  tempSensor.begin();
  if (tempSensor.getDeviceCount() == 0) {
    Serial.println("WARNING: no DS18B20 found. Check wiring & 4.7k pull-up resistor.");
  } else {
    tempSensor.setWaitForConversion(false);
    tempSensor.requestTemperatures();
    Serial.println("DS18B20 ready.");
  }
}

// ---------------------------------------------------------------------
// Acquisition + publish
// ---------------------------------------------------------------------
// Fills vibCounts with SAMPLES_PER_WINDOW consecutive samples from the FIFO.
// Returns false if the FIFO overran (samples lost) or acquisition timed out.
// Nothing else (WiFi, OneWire) may run in here: the 32-sample FIFO holds only
// ~20 ms of data at 1600 Hz.
bool acquireWindow() {
  adxlResetFifo();
  int n = 0;
  unsigned long start = micros();
  while (n < SAMPLES_PER_WINDOW) {
    uint8_t entries = adxlRead(ADXL_REG_FIFO_STATUS) & 0x3F;
    while (entries-- > 0 && n < SAMPLES_PER_WINDOW) {
      int16_t xyz[3];
      adxlReadXYZ(xyz);
      vibCounts[n++] = xyz[VIB_AXIS];
      delayMicroseconds(5);                  // datasheet: >= 5 us between FIFO reads
    }
    if (micros() - start > WINDOW_TIMEOUT_US) return false;
  }
  return (adxlRead(ADXL_REG_INT_SOURCE) & ADXL_INT_OVERRUN) == 0;
}

void publishWindow(unsigned long t0Ms) {
  size_t len = pdm_pack_window(packet, sizeof(packet), vibCounts,
                               (uint16_t)SAMPLES_PER_WINDOW,
                               (uint16_t)VIBRATION_SAMPLE_RATE_HZ,
                               (uint32_t)t0Ms, ADXL345_SCALE_G);
  if (len == 0 || !mqttClient.publish(mqttTopicVibration, packet, len, false)) {
    Serial.println("WARNING: vibration publish failed (disconnected or buffer too small)");
  }
}

void publishStats(unsigned long t0Ms) {
  PdmStats st = pdm_compute_stats(vibCounts, (uint16_t)SAMPLES_PER_WINDOW, ADXL345_SCALE_G);
  char buf[160];
  int len = snprintf(buf, sizeof(buf),
      "{\"t0_ms\":%lu,\"rms\":%.5f,\"p2p\":%.4f,\"crest\":%.3f,\"kurtosis\":%.3f,\"skewness\":%.3f}",
      t0Ms, st.rms, st.p2p, st.crest, st.kurtosis, st.skewness);
  mqttClient.publish(mqttTopicStats, (const uint8_t*)buf, len, false);
}

void handleTemperature() {
  unsigned long now = millis();
  if (now - lastTempReadMillis < TEMPERATURE_READ_INTERVAL_MS) return;
  lastTempReadMillis = now;

  float c = tempSensor.getTempCByIndex(0);
  tempSensor.requestTemperatures();
  if (c == DEVICE_DISCONNECTED_C) {
    Serial.println("WARNING: DS18B20 read failed.");
    return;
  }
  char payload[128];
  int len = snprintf(payload, sizeof(payload),
      "{\"machine_id\":\"%s\",\"t_ms\":%lu,\"temperature_c\":%.2f}", MACHINE_ID, now, c);
  mqttClient.publish(mqttTopicTemperature, (const uint8_t*)payload, len, false);
}

void loop() {
  if (!mqttClient.connected()) connectMQTT();

  unsigned long t0 = millis();
  if (acquireWindow()) {
#if PUBLISH_STATS
    publishStats(t0);
#endif
    if ((windowCounter++ % RAW_EVERY_N_WINDOWS) == 0) publishWindow(t0);
  } else {
    droppedWindows++;
    Serial.print("WARNING: window dropped (FIFO overrun/timeout). Total dropped: ");
    Serial.println(droppedWindows);
  }
  handleTemperature();      // slow OneWire/network work happens between windows
  mqttClient.loop();
}
