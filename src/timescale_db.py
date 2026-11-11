"""
timescale_db.py -- PostgreSQL / TimescaleDB storage with the same interface as
timeseries_db.TimeSeriesDB (write_score / write_temperature / write_features /
query_recent_scores / query_anomaly_history).

Why: Grafana reads PostgreSQL natively (alerting, multi-machine dashboards,
retention), and TimescaleDB adds hypertables + compression on top. If the
timescaledb extension is not installed the tables are plain PostgreSQL tables
-- everything still works, just without hypertable partitioning.

Configure with PDM_PG_DSN, e.g.
    postgresql://pdm:secret@localhost:5432/pdm
Fails SOFT like the Influx version: a down database must never stop live
scoring; rows are dropped (with a warning) until it comes back.
"""
import json
import os
import time

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS pdm_scores (
    time TIMESTAMPTZ NOT NULL DEFAULT now(), machine_id TEXT NOT NULL,
    score DOUBLE PRECISION, threshold DOUBLE PRECISION, is_anomaly BOOLEAN,
    device_t_ms BIGINT);
CREATE TABLE IF NOT EXISTS pdm_temperature (
    time TIMESTAMPTZ NOT NULL DEFAULT now(), machine_id TEXT NOT NULL,
    temperature_c DOUBLE PRECISION, device_t_ms BIGINT);
CREATE TABLE IF NOT EXISTS pdm_features (
    time TIMESTAMPTZ NOT NULL DEFAULT now(), machine_id TEXT NOT NULL,
    features JSONB NOT NULL, device_t_ms BIGINT);
CREATE INDEX IF NOT EXISTS pdm_scores_m_t ON pdm_scores (machine_id, time DESC);
CREATE INDEX IF NOT EXISTS pdm_temperature_m_t ON pdm_temperature (machine_id, time DESC);
CREATE INDEX IF NOT EXISTS pdm_features_m_t ON pdm_features (machine_id, time DESC);
"""
TABLES = ("pdm_scores", "pdm_temperature", "pdm_features")


class TimescaleDB:
    def __init__(self, dsn=None):
        self.enabled = False
        self.conn = None
        self.hypertables = False
        self.dsn = dsn or os.environ.get("PDM_PG_DSN")
        if not _PG_AVAILABLE:
            print("NOTE: 'psycopg2' not installed (pip install psycopg2-binary) -> DB disabled.")
            return
        if not self.dsn:
            print("NOTE: PDM_PG_DSN not set -> PostgreSQL storage disabled.")
            return
        try:
            self._connect()
            with self.conn.cursor() as cur:
                cur.execute(SCHEMA)
            self._try_hypertables()
            self.enabled = True
            print(f"Connected to PostgreSQL ({'TimescaleDB hypertables' if self.hypertables else 'plain tables'}).")
        except Exception as e:
            print(f"NOTE: could not initialise PostgreSQL ({e}) -> DB disabled, files only.")

    def _connect(self):
        # KNOWN CRASH: psycopg2-binary's SSL negotiation segfaults when TensorFlow
        # was imported earlier in the same process (clashing OpenSSL libraries).
        # Local connections don't need SSL, so default it off there. For a REMOTE
        # database either keep TensorFlow out of the process (use the ONNX
        # backend) or import psycopg2 before TensorFlow (mqtt_bridge.py does).
        kwargs = {}
        if "sslmode" not in self.dsn and any(h in self.dsn for h in ("@localhost", "@127.0.0.1", "host=localhost", "host=127.0.0.1")):
            kwargs["sslmode"] = "disable"
        self.conn = psycopg2.connect(self.dsn, connect_timeout=3, **kwargs)
        self.conn.autocommit = True

    def _try_hypertables(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                for t in TABLES:
                    cur.execute("SELECT create_hypertable(%s, 'time', if_not_exists => TRUE)", (t,))
            self.hypertables = True
        except Exception:
            self.hypertables = False        # extension not installed: plain tables are fine

    def _exec(self, sql, params=()):
        """Run a write; reconnect once if the connection dropped. Never raises."""
        for attempt in (0, 1):
            try:
                with self.conn.cursor() as cur:
                    cur.execute(sql, params)
                return True
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                if attempt:
                    break
                try:
                    self._connect()
                except Exception:
                    break
            except Exception as e:
                print(f"WARNING: PostgreSQL write failed: {e}")
                return False
        print("WARNING: PostgreSQL unreachable; row dropped.")
        return False

    # ------------------------------------------------------------- writes
    def write_score(self, machine_id, score, threshold, is_anomaly, t_ms=None):
        if self.enabled:
            self._exec("INSERT INTO pdm_scores (machine_id, score, threshold, is_anomaly, device_t_ms) "
                       "VALUES (%s,%s,%s,%s,%s)",
                       (machine_id, float(score), float(threshold), bool(is_anomaly), t_ms))

    def write_temperature(self, machine_id, temperature_c, t_ms=None):
        if self.enabled:
            self._exec("INSERT INTO pdm_temperature (machine_id, temperature_c, device_t_ms) VALUES (%s,%s,%s)",
                       (machine_id, float(temperature_c), t_ms))

    def write_features(self, machine_id, feature_dict, t_ms=None):
        if self.enabled:
            self._exec("INSERT INTO pdm_features (machine_id, features, device_t_ms) VALUES (%s,%s,%s)",
                       (machine_id, Json({k: float(v) for k, v in feature_dict.items()}), t_ms))

    # ------------------------------------------------------------ queries
    def _query(self, sql, params):
        if not self.enabled:
            return []
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"WARNING: PostgreSQL query failed: {e}")
            return []

    def query_recent_scores(self, machine_id, minutes=60, limit=500):
        rows = self._query(
            "SELECT time, score, threshold, is_anomaly FROM ("
            " SELECT * FROM pdm_scores WHERE machine_id=%s AND time > now() - %s * interval '1 minute'"
            " ORDER BY time DESC LIMIT %s) s ORDER BY time ASC", (machine_id, int(minutes), int(limit)))
        return rows

    def query_anomaly_history(self, machine_id=None, hours=24):
        sql = "SELECT * FROM pdm_scores WHERE is_anomaly AND time > now() - %s * interval '1 hour'"
        params = [int(hours)]
        if machine_id:
            sql += " AND machine_id=%s"
            params.append(machine_id)
        return self._query(sql + " ORDER BY time ASC", params)
