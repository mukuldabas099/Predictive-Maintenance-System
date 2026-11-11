"""
mqtt_simulator.py
--------------------
Publishes synthetic vibration/temperature data over MQTT using the EXACT
same JSON schema as the real Arduino R4 firmware (see arduino_r4_firmware/arduino_r4_firmware.ino
and arduino_r4_firmware/README.md), so mqtt_bridge.py and the future dashboard can
be built and tested end-to-end without physical hardware.

Run alongside mqtt_bridge.py:
    python3 mqtt_simulator.py --broker localhost --machine-id PUMP_01 --scenario degrading
"""

import argparse
import json
import time
import sys
import numpy as np
import paho.mqtt.client as mqtt

from synthetic_data import generate_run
from pdm_config import SAMPLE_RATE_HZ as SAMPLE_RATE, WINDOW_S
from payload_codec import encode_binary_window
from mqtt_common import add_mqtt_args, make_client, effective_port

SAMPLES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_S)


def main():
    ap = argparse.ArgumentParser()
    add_mqtt_args(ap)
    ap.add_argument("--format", choices=["binary", "json"], default="binary",
                    help="'binary' = what the ADXL345 firmware sends (topic vib_bin); "
                         "'json' = legacy text format (topic vibration)")
    ap.add_argument("--machine-id", default="PUMP_01")
    ap.add_argument("--scenario", choices=["healthy", "degrading"], default="degrading",
                     help="'healthy': stays healthy the whole run. "
                          "'degrading': ramps into a fault, like the Arduino would "
                          "stream during a real degradation event.")
    ap.add_argument("--duration", type=float, default=120, help="seconds of data to stream")
    ap.add_argument("--speed", type=float, default=1.0,
                     help="playback speed multiplier (10.0 = 10x faster than real time, for quick demos)")
    args = ap.parse_args()

    print(f"Generating {args.duration:.0f}s of synthetic '{args.scenario}' data "
          f"({SAMPLE_RATE} Hz)...")
    if args.scenario == "healthy":
        run = generate_run(duration_s=args.duration, sample_rate=SAMPLE_RATE,
                            degradation_start_frac=1.1, failure_frac=1.2)  # never degrades
    else:
        run = generate_run(duration_s=args.duration, sample_rate=SAMPLE_RATE,
                            degradation_start_frac=0.4, failure_frac=0.95)

    client = make_client(args, client_id=f"simulator-{args.machine_id}")
    port = effective_port(args)
    client.connect(args.broker, port, keepalive=30)
    client.loop_start()

    topic_vib = f"pdm/{args.machine_id}/" + ("vib_bin" if args.format == "binary" else "vibration")
    topic_temp = f"pdm/{args.machine_id}/temperature"
    topic_status = f"pdm/{args.machine_id}/status"
    client.publish(topic_status, "online", retain=True)
    print(f"Publishing to {topic_vib} and {topic_temp} on {args.broker}:{port}")
    print("(Ctrl+C to stop)")

    n = len(run["vibration"])
    t0 = time.time()
    window_start = 0
    last_temp_publish = 0
    sample_interval = 1.0 / SAMPLE_RATE / args.speed

    try:
        i = 0
        while i < n:
            window_end = min(i + SAMPLES_PER_WINDOW, n)
            samples = run["vibration"][i:window_end].tolist()

            t0_ms = int((time.time() - t0) * 1000)
            if args.format == "binary":
                from scipy.stats import kurtosis as _k, skew as _s
                _x = np.asarray(samples, dtype=np.float64) - np.mean(samples)
                _rms = float(np.sqrt(np.mean(_x ** 2)))
                client.publish(f"pdm/{args.machine_id}/stats", json.dumps({
                    "t0_ms": t0_ms, "rms": _rms, "p2p": float(np.ptp(samples)),
                    "crest": float(np.max(np.abs(_x)) / (_rms + 1e-12)),
                    "kurtosis": float(_k(_x)), "skewness": float(_s(_x))}))
                # add a 1 g gravity offset like a real sensor; the bridge removes it
                client.publish(topic_vib, encode_binary_window(
                    np.asarray(samples) + 1.0, SAMPLE_RATE, t0_ms))
            else:
                client.publish(topic_vib, json.dumps({
                    "machine_id": args.machine_id, "t0_ms": t0_ms,
                    "sample_rate": SAMPLE_RATE, "n": len(samples),
                    "samples": [round(float(s), 4) for s in samples]}))

            temp_payload = {
                "machine_id": args.machine_id,
                "t_ms": int((time.time() - t0) * 1000),
                "temperature_c": round(float(np.mean(run["temperature"][i:window_end])), 2),
            }
            client.publish(topic_temp, json.dumps(temp_payload))

            elapsed_sim_s = i / SAMPLE_RATE
            print(f"\r  streamed {elapsed_sim_s:6.1f}s / {args.duration:.0f}s "
                  f"(state={run['state'][i]})", end="", flush=True)

            i = window_end
            time.sleep(WINDOW_S / args.speed)
    except KeyboardInterrupt:
        pass
    finally:
        client.publish(topic_status, "offline", retain=True)
        client.loop_stop()
        client.disconnect()
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
