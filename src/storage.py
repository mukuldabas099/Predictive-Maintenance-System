"""storage.py -- pick the time-series backend.
PDM_DB=timescale|influx|none  (default: timescale if PDM_PG_DSN is set, else influx)."""
import os


def open_db():
    choice = os.environ.get("PDM_DB", "").lower() or ("timescale" if os.environ.get("PDM_PG_DSN") else "influx")
    if choice == "timescale":
        from timescale_db import TimescaleDB
        return TimescaleDB()
    if choice == "none":
        class _Null:
            enabled = False
            def __getattr__(self, name):
                return lambda *a, **k: []
        return _Null()
    from timeseries_db import TimeSeriesDB
    return TimeSeriesDB()
