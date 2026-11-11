# AI/IoT Predictive Maintenance System

An IoT- and machine-learning-based predictive maintenance prototype that
monitors machine vibration and temperature, learns normal operating
behaviour, and flags unusual patterns that may indicate developing
equipment faults.

The project combines an Arduino-based sensing design, MQTT
communication, Python signal processing, an LSTM Autoencoder, live
anomaly scoring, a Streamlit dashboard, and optional time-series storage
and alerting.

> **Project status:** The Python ML pipeline and simulator-based MQTT
> flow have been tested. The Arduino UNO R4 WiFi firmware is written but
> has **not** been compiled or tested on physical hardware. Review the
> limitations section before interpreting results as real-world machine
> performance.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [System Architecture](#system-architecture)
-   [Key Features](#key-features)
-   [Technology Stack](#technology-stack)
-   [Machine Learning Approach](#machine-learning-approach)
-   [Repository Structure](#repository-structure)
-   [Getting Started](#getting-started)
-   [Run the Demo](#run-the-demo)
-   [Run the MQTT Live Pipeline](#run-the-mqtt-live-pipeline)
-   [Run the Dashboard](#run-the-dashboard)
-   [Run Tests](#run-tests)
-   [Hardware Setup](#hardware-setup)
-   [Configuration](#configuration)
-   [Validation and Results](#validation-and-results)
-   [Current Limitations](#current-limitations)
-   [Future Improvements](#future-improvements)

------------------------------------------------------------------------

## Project Overview

Unexpected machine failures can cause downtime, maintenance costs, and
production losses. This project explores condition monitoring through
vibration and temperature data.

The system extracts useful characteristics from vibration signals and
passes time-ordered feature sequences to an **LSTM Autoencoder** trained
primarily on healthy operating data. The model reconstructs the input
sequence. A high reconstruction error indicates that the observed
behaviour differs from the learned healthy pattern and may require
inspection.

### Project objectives

-   Collect machine-condition data using vibration and temperature
    sensors.
-   Transmit sensor readings through an IoT messaging protocol.
-   Extract time-domain and frequency-domain features from vibration
    signals.
-   Detect abnormal behaviour using an unsupervised deep-learning
    approach.
-   Display machine status and anomaly-score trends.
-   Generate alerts when an anomaly persists beyond the configured rule.

------------------------------------------------------------------------

## System Architecture

``` text
 ADXL345 Accelerometer       DS18B20 Temperature Sensor
          |                            |
          +--------- Arduino UNO R4 WiFi+
                         |
                       Wi-Fi
                         |
                  MQTT / Mosquitto
                         |
              Python MQTT Bridge
                         |
             Signal Processing & Features
                         |
                LSTM Autoencoder
                         |
             Reconstruction Error Score
                         |
              Threshold / Alert Logic
                         |
          +--------------+---------------+
          |              |               |
      Streamlit       Time-Series      Email Alert
      Dashboard        Storage          (optional)
```

The repository also includes an MQTT simulator. It publishes data using
the project's MQTT message format, allowing the Python pipeline to be
exercised without physical sensors.

------------------------------------------------------------------------

## Key Features

-   **Vibration monitoring:** Processes accelerometer readings in
    fixed-size windows.
-   **Temperature monitoring:** Supports temperature readings from a
    DS18B20 sensor.
-   **Signal feature extraction:** Calculates statistical and spectral
    characteristics.
-   **Unsupervised anomaly detection:** Learns patterns from healthy
    data using an LSTM Autoencoder.
-   **Calibrated anomaly threshold:** Compares reconstruction error
    against a threshold derived from healthy validation data.
-   **Persistent alert logic:** Supports a k-of-n rule to reduce alerts
    caused by isolated spikes.
-   **MQTT integration:** Connects the simulator or firmware data source
    to the Python scoring backend.
-   **Dashboard:** Displays machine status, anomaly scores, temperature,
    and score history.
-   **Optional storage and notifications:** Includes time-series
    database and SMTP alert integrations.

------------------------------------------------------------------------

## Technology Stack

  ------------------------------------------------------------------------
  Area                    Technologies             Purpose
  ----------------------- ------------------------ -----------------------
  Microcontroller / IoT   Arduino UNO R4 WiFi      Sensor acquisition and
                                                   wireless connectivity

  Sensors                 ADXL345, DS18B20         Vibration and
                                                   temperature
                                                   measurements

  Firmware                Embedded C/C++, Arduino  Reads sensors and
                          libraries                publishes data

  Communication           Wi-Fi, MQTT, Eclipse     Transfers sensor
                          Mosquitto                messages

  Backend                 Python                   Data handling and live
                                                   scoring

  Data processing         NumPy, Pandas, SciPy     Numerical processing
                                                   and signal analysis

  Signal processing       FFT, RMS, kurtosis,      Converts raw vibration
                          skewness, crest factor   into useful features

  Machine learning        TensorFlow, Keras, LSTM  Learns normal
                          Autoencoder              time-series behaviour

  Model serving           ONNX, ONNX Runtime       Runs exported model
                          (optional)               inference

  Dashboard               Streamlit, Altair,       Monitoring and
                          Matplotlib               visualization

  Storage                 InfluxDB or              Optional time-series
                          PostgreSQL/TimescaleDB   persistence

  Experiment / pipeline   MLflow, DVC, Optuna      Experiment tracking,
  tools                   (optional)               reproducibility, tuning

  Deployment              Docker, Docker Compose   Containerized
                          (optional)               infrastructure

  Notifications           Python SMTP integration  Email alerts
                          (optional)               
  ------------------------------------------------------------------------

**Note:** Optional and alternative components are included in the
repository, but they are not all required to run the basic demo.
InfluxDB and PostgreSQL/TimescaleDB are alternative storage paths.

------------------------------------------------------------------------

## Machine Learning Approach

### LSTM Autoencoder

The main model is an **LSTM Autoencoder**, designed for sequential
sensor features.

1.  **Training data:** The model learns patterns from healthy operating
    sequences.
2.  **Encoder:** LSTM layers compress the input sequence into a learned
    representation.
3.  **Decoder:** The decoder attempts to reconstruct the original
    sequence.
4.  **Reconstruction error:** Mean Squared Error (MSE) measures the
    difference between the input and reconstruction.
5.  **Thresholding:** The error is compared with a calibrated threshold.
6.  **Anomaly decision:** A score above the threshold is flagged as
    unusual. The live pipeline can apply a persistence rule before
    triggering an alert.

### Signal features

The feature-engineering pipeline combines time-domain and
frequency-domain information. Examples include:

  -----------------------------------------------------------------------
  Feature                             What it indicates
  ----------------------------------- -----------------------------------
  RMS                                 Overall vibration energy /
                                      intensity

  Peak-to-peak                        Difference between maximum and
                                      minimum signal values

  Crest factor                        Peak amplitude relative to RMS

  Kurtosis                            How strongly the signal contains
                                      impulsive peaks

  Skewness                            Asymmetry of the signal
                                      distribution

  Dominant frequency                  Frequency component with the
                                      greatest magnitude

  Spectral energy / centroid /        Distribution of signal energy
  entropy                             across frequencies
  -----------------------------------------------------------------------

The current Python configuration uses a **1600 Hz sampling rate** and
**1-second windows** (1600 samples per window). Training feature windows
use 50% overlap.

------------------------------------------------------------------------

## Repository Structure

``` text
pdm_project/
├── arduino_r4_firmware/
│   ├── arduino_r4_firmware.ino
│   ├── config.h
│   ├── pdm_features.h
│   ├── pdm_payload.h
│   └── README.md
│
├── deploy/
│   ├── mosquitto/
│   ├── grafana/
│   ├── docker-compose.yml
│   └── README.md
│
├── outputs/
│   ├── lstm_autoencoder.keras
│   ├── lstm_autoencoder.onnx
│   ├── threshold_and_norm.npz
│   ├── anomaly_comparison.png
│   └── benchmark_results.json
│
├── src/
│   ├── main.py
│   ├── demo.py
│   ├── synthetic_data.py
│   ├── feature_engineering.py
│   ├── lstm_autoencoder.py
│   ├── calibration.py
│   ├── mqtt_bridge.py
│   ├── mqtt_simulator.py
│   ├── dashboard.py
│   ├── alerting.py
│   ├── timeseries_db.py
│   ├── timescale_db.py
│   └── ...
│
├── tests/
├── requirements.txt
├── params.yaml
├── dvc.yaml
├── run_demo.bat
├── run_tests.bat
└── README.md
```

------------------------------------------------------------------------

## Getting Started

### Prerequisites

-   Python **3.10--3.12** recommended.
-   Git.
-   Windows, Linux, or macOS.
-   Internet access to install Python dependencies.

For the MQTT pipeline, install and run an Eclipse Mosquitto broker.
Physical Arduino hardware is not needed for the simulator-based demo.

### 1. Clone the repository

``` bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

Replace the placeholders with your GitHub username and repository name.

### 2. Create a virtual environment

**Windows PowerShell:**

``` powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some dependencies are optional and are only needed for specific
integrations, such as ONNX export, database storage, experiment
tracking, or DVC.

------------------------------------------------------------------------

## Run the Demo

The simplest way to see the core ML workflow is to run the demo. It does
not require an MQTT broker, physical hardware, or a database.

From the project root:

**Windows:**

``` powershell
.\run_demo.bat
```

Or:

``` bash
python src/demo.py
```

If the trained model or calibration files are missing, train the model
first:

``` bash
python src/main.py
```

The demo illustrates the flow:

``` text
Synthetic vibration data
        ↓
Feature extraction
        ↓
LSTM Autoencoder
        ↓
Reconstruction error
        ↓
Threshold comparison
        ↓
Normal / Anomaly result
```

Generated plots and result files are saved in the `outputs/` directory.

------------------------------------------------------------------------

## Run the MQTT Live Pipeline

This mode demonstrates the sensor-to-model communication flow using
synthetic data and a local MQTT broker.

### 1. Start Mosquitto

In a separate terminal:

``` bash
mosquitto -p 1883
```

Make sure Mosquitto is installed and available in your system PATH.

### 2. Start the Python MQTT bridge

From the project root:

``` bash
python src/mqtt_bridge.py --broker localhost
```

### 3. Start the MQTT simulator

Open another terminal:

``` bash
python src/mqtt_simulator.py --broker localhost --machine-id PUMP_01 --scenario degrading --duration 90 --speed 25
```

The simulator supports:

-   `healthy` --- produces a healthy operating scenario.
-   `degrading` --- simulates a machine moving toward a fault condition.

The simulator publishes messages using the project's MQTT schema. The
bridge receives the data, calculates features, scores the sequence, and
updates the live status/history outputs.

> This validates the software pipeline with simulated sensor data. It
> does not prove that the firmware has been tested on a physical
> machine.

------------------------------------------------------------------------

## Run the Dashboard

Start the broker, bridge, and simulator as described above. Then open
another terminal and run:

``` bash
streamlit run src/dashboard.py
```

Open the local URL printed by Streamlit, normally:

``` text
http://localhost:8501
```

The dashboard reads the status and score-history files generated by the
MQTT bridge. It can display machine status, temperature, anomaly score,
and score trends.

------------------------------------------------------------------------

## Run Tests

From the project root:

``` bash
python -m unittest discover -s tests -v
```

On Windows, the included helper script can also be used:

``` powershell
.\run_tests.bat
```

------------------------------------------------------------------------

## Hardware Setup

The firmware is designed for:

-   Arduino UNO R4 WiFi
-   ADXL345 accelerometer using SPI
-   DS18B20 temperature sensor
-   Wi-Fi connection to an MQTT broker

The firmware configuration and wiring instructions are documented in
[`arduino_r4_firmware/README.md`](arduino_r4_firmware/README.md).

**Important hardware notes:**

-   Configure Wi-Fi credentials, broker address, MQTT credentials, and
    machine ID in the firmware configuration.
-   Keep the firmware sampling rate and window duration aligned with
    `src/pdm_config.py`.
-   Use appropriate voltage-level protection for a bare 3.3 V ADXL345
    module with the UNO R4's 5 V I/O.
-   Mount the accelerometer firmly to the machine housing.
-   The firmware has not yet been compiled or run on a physical Arduino
    board; hardware bring-up and validation remain necessary.

------------------------------------------------------------------------

## Configuration

Common settings are defined in:

-   `src/pdm_config.py` --- Python sampling rate, window duration, and
    overlap.
-   `arduino_r4_firmware/config.h` --- firmware sensor, Wi-Fi, and MQTT
    settings.
-   `params.yaml` --- model and pipeline parameters.
-   `deploy/.env.example` --- example deployment environment variables.

Do not commit real passwords, API keys, Wi-Fi credentials, SMTP
credentials, private certificates, or other secrets. Use environment
variables or a local `.env` file excluded from Git.

------------------------------------------------------------------------

## Validation and Results

The project includes:

-   Synthetic bearing-condition data generation.
-   Feature extraction and model evaluation.
-   Baseline comparisons, including Isolation Forest and One-Class SVM.
-   Cross-validation and ablation experiments on the Case Western
    Reserve University (CWRU) bearing dataset.
-   A simulator-based MQTT-to-bridge live scoring test.
-   Unit tests for key pipeline components.

The README and experiment output files contain the detailed benchmark
tables and evaluation caveats. Results depend on the dataset, operating
conditions, threshold calibration, and evaluation protocol. They should
not be interpreted as guaranteed industrial performance.

------------------------------------------------------------------------

## Current Limitations

1.  **Physical hardware validation:** The Arduino UNO R4 WiFi firmware
    is written but has not been compiled or tested on the actual board.
2.  **Simulated operating conditions:** Some headline demo results use
    synthetic data. Synthetic performance does not establish real
    factory performance.
3.  **Sensor and model gap:** The ADXL345 has an 800 Hz bandwidth; some
    bearing-resonance diagnostic methods require higher-frequency
    measurements.
4.  **Windowing difference:** Training uses overlapping windows, while
    the live firmware sends non-overlapping windows with a short gap.
    This difference needs further validation.
5.  **Real-data generalization:** CWRU evaluation shows that
    false-positive performance can vary by operating load and
    calibration condition. More healthy recordings across real operating
    conditions are needed.
6.  **NASA IMS validation:** The IMS validation code exists, but the
    dataset-based validation has not been completed.
7.  **External services:** Email has been tested with a local SMTP
    server, not verified against a real email provider. The Docker
    Compose / Grafana deployment also requires full environment testing.

------------------------------------------------------------------------

## Future Improvements

-   Compile, flash, and validate the Arduino firmware with real sensor
    readings.
-   Collect vibration and temperature data from operating machines.
-   Validate the model across different loads, speeds, and machine
    types.
-   Improve threshold calibration and reduce false alarms.
-   Complete NASA IMS dataset validation and run-to-failure evaluation.
-   Test the complete Docker, database, and Grafana deployment.
-   Validate email delivery with a real SMTP provider.
-   Explore edge inference and higher-bandwidth vibration sensors.

------------------------------------------------------------------------

## Contributing

Contributions, bug reports, and suggestions are welcome.

1.  Fork the repository.
2.  Create a feature branch.
3.  Make and test your changes.
4.  Open a pull request with a clear description.

------------------------------------------------------------------------


------------------------------------------------------------------------

**Built as an AI + IoT predictive maintenance project.**
