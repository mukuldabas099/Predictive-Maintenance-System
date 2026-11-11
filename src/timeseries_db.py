"""
timeseries_db.py
-------------------
Persistent time-series storage (the "Data storage" tech-stack item from the
project brief: "Firebase / InfluxDB / MongoDB (time-series friendly)").

Uses InfluxDB (1.x line protocol) because it's purpose-built for exactly this
data shape (machine_id-tagged numeric readings over time), free/self-hostable
(no cloud account needed for a student project demo), and has a simple
Python client.

Replaces/augments the flat-file approach in mqtt_bridge.py
(live_status.json / score_history.jsonl) with a real queryable database:
history survives a backend restart, supports time-range + aggregate queries
("average anomaly score per hour", "all anomalies in the last 24h"), and
scales to many machines/months of data far better than an append-only file.

If InfluxDB isn't running, every method here fails SOFT (logs a warning,
returns empty results) rather than crashing the bridge — sensor streaming
and live scoring should keep working even if the DB is temporarily down;
losing some historical rows is recoverable, losing live monitoring isn't.
"""

import os
import time

try:
    from influxdb import InfluxDBClient
    from influxdb.exceptions import InfluxDBClientError
    _INFLUXDB_AVAILABLE = True
except ImportError:
    _INFLUXDB_AVAILABLE = False


DB_NAME = os.environ.get("PDM_INFLUXDB_NAME", "pdm")
DB_HOST = os.environ.get("PDM_INFLUXDB_HOST", "localhost")
DB_PORT = int(os.environ.get("PDM_INFLUXDB_PORT", "8086"))


class TimeSeriesDB:
    def __init__(self, host=DB_HOST, port=DB_PORT, database=DB_NAME):
        self.enabled = False
        self.client = None
        if not _INFLUXDB_AVAILABLE:
            print("NOTE: 'influxdb' package not installed (pip install influxdb) "
                  "-> persistent storage disabled, falling back to files only.")
            return
        try:
            self.client = InfluxDBClient(host=host, port=port, database=database, timeout=3)
            self.client.create_database(database)  # no-op if it already exists
            self.enabled = True
            print(f"Connected to InfluxDB at {host}:{port}, database '{database}'.")
        except Exception as e:
            print(f"NOTE: could not connect to InfluxDB at {host}:{port} ({e}) "
                  f"-> persistent storage disabled, falling back to files only. "
                  f"Start it with: influxd")
            self.enabled = False

    def write_score(self, machine_id, score, threshold, is_anomaly, t_ms=None):
        """Store one scored window's anomaly score + threshold."""
        if not self.enabled:
            return
        fields = {
            "score": float(score),
            "threshold": float(threshold),
            "is_anomaly": bool(is_anomaly),
        }
        if t_ms is not None:
            fields["device_t_ms"] = int(t_ms)
        point = {
            "measurement": "anomaly_score",
            "tags": {"machine_id": machine_id},
            "time": self._timestamp_ns(),
            "fields": fields,
        }
        self._write_safe([point])

    def write_temperature(self, machine_id, temperature_c, t_ms=None):
        if not self.enabled:
            return
        fields = {"temperature_c": float(temperature_c)}
        if t_ms is not None:
            fields["device_t_ms"] = int(t_ms)
        point = {
            "measurement": "temperature",
            "tags": {"machine_id": machine_id},
            "time": self._timestamp_ns(),
            "fields": fields,
        }
        self._write_safe([point])

    def write_features(self, machine_id, feature_dict, t_ms=None):
        """Store the raw extracted features too (rms, kurtosis, etc.) -- useful
        for later offline analysis/retraining, not just the anomaly score."""
        if not self.enabled:
            return
        fields = {k: float(v) for k, v in feature_dict.items()}
        if t_ms is not None:
            fields["device_t_ms"] = int(t_ms)
        point = {
            "measurement": "features",
            "tags": {"machine_id": machine_id},
            "time": self._timestamp_ns(),
            "fields": fields,
        }
        self._write_safe([point])

    def query_recent_scores(self, machine_id, minutes=60, limit=500):
        """Returns a list of {time, score, threshold, is_anomaly} dicts, most
        recent last -- used by the dashboard for the trend chart."""
        if not self.enabled:
            return []
        query = (
            f'SELECT score, threshold, is_anomaly FROM anomaly_score '
            f"WHERE machine_id = '{machine_id}' AND time > now() - {int(minutes)}m "
            f'ORDER BY time ASC LIMIT {int(limit)}'
        )
        try:
            result = self.client.query(query)
            return list(result.get_points())
        except InfluxDBClientError as e:
            print(f"WARNING: InfluxDB query failed: {e}")
            return []

    def query_anomaly_history(self, machine_id=None, hours=24):
        """All windows flagged anomalous in the last N hours, optionally
        filtered to one machine -- e.g. for a maintenance-log report."""
        if not self.enabled:
            return []
        where = f"is_anomaly = true AND time > now() - {int(hours)}h"
        if machine_id:
            where += f" AND machine_id = '{machine_id}'"
        query = f"SELECT * FROM anomaly_score WHERE {where} ORDER BY time ASC"
        try:
            result = self.client.query(query)
            return list(result.get_points())
        except InfluxDBClientError as e:
            print(f"WARNING: InfluxDB query failed: {e}")
            return []

    def list_machines(self):
        """Distinct machine_ids seen in the DB (for a dashboard that doesn't
        already know the machine list from live MQTT traffic)."""
        if not self.enabled:
            return []
        try:
            result = self.client.query('SHOW TAG VALUES FROM anomaly_score WITH KEY = "machine_id"')
            return [p["value"] for p in result.get_points()]
        except InfluxDBClientError as e:
            print(f"WARNING: InfluxDB query failed: {e}")
            return []

    def _write_safe(self, points):
        try:
            self.client.write_points(points)
        except Exception as e:
            print(f"WARNING: InfluxDB write failed ({e}) -- continuing without persisting this point.")

    @staticmethod
    def _timestamp_ns():
        """InfluxDB wants nanosecond epoch timestamps.

        IMPORTANT: this deliberately does NOT use the device-supplied t_ms
        (elapsed milliseconds since the Arduino/simulator started, i.e.
        millis()-style relative time, not wall-clock epoch time) as the point
        timestamp. Using an unsynchronized relative device clock as an
        absolute epoch timestamp silently writes every point to ~1970-01-01,
        making it invisible to any "time > now() - X" query -- this exact bug
        was caught while testing this module (see README's "known
        limitations" section). Using the bridge's own receipt time is simpler
        and correct as long as ingestion latency is small (true for MQTT over
        a local/LAN broker). The original device timestamp is preserved as a
        'device_t_ms' field for reference/debugging, just not used for
        indexing. For a deployment where clock sync matters more (e.g. WAN
        links with real queueing delay), sync the Arduino's clock via NTP and
        reintroduce it as the point timestamp instead.
        """
        return int(time.time() * 1e9)


if __name__ == "__main__":
    # Smoke test: write a few points and read them back.
    db = TimeSeriesDB()
    if not db.enabled:
        print("InfluxDB not available -- start it with `influxd` and re-run this test.")
    else:
        print("Writing test points...")
        for i in range(5):
            db.write_score("TEST_MACHINE", score=0.5 + i * 0.1, threshold=1.0,
                            is_anomaly=(0.5 + i * 0.1) > 1.0)
            db.write_temperature("TEST_MACHINE", temperature_c=35.0 + i)
            time.sleep(0.05)
        print("Reading back...")
        points = db.query_recent_scores("TEST_MACHINE", minutes=5)
        for p in points:
            print(" ", p)
        print(f"Machines in DB: {db.list_machines()}")
