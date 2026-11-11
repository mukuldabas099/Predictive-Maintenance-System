"""
dashboard.py
--------------
The monitoring dashboard from the project brief's "System Output" section:
  "A monitoring dashboard displays live sensor readings and the current
   anomaly score for each monitored machine. When the anomaly score crosses
   the calibrated threshold, the system triggers an alert ... flagging the
   machine for inspection before it actually fails."

Reads from the files mqtt_bridge.py writes (no MQTT client needed on this
side, keeping the dashboard simple and decoupled):
  - outputs/live_status.json   -> latest snapshot per machine
  - outputs/score_history.jsonl -> trend line of anomaly score over time

Run: streamlit run dashboard.py
(mqtt_bridge.py must be running and receiving data from the Arduino R4 or
mqtt_simulator.py for there to be anything to show)
"""

import json
import os
import time

import pandas as pd
import altair as alt
import streamlit as st

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
LIVE_STATUS_PATH = os.path.join(OUT_DIR, "live_status.json")
SCORE_HISTORY_PATH = os.path.join(OUT_DIR, "score_history.jsonl")

REFRESH_SECONDS = 3
HISTORY_POINTS_SHOWN = 120  # per machine, most recent N scored windows


st.set_page_config(page_title="Predictive Maintenance Monitor", page_icon="⚙️", layout="wide")


def load_live_status():
    if not os.path.exists(LIVE_STATUS_PATH):
        return None
    try:
        with open(LIVE_STATUS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # tolerate a read racing an in-progress atomic write


def load_history():
    if not os.path.exists(SCORE_HISTORY_PATH):
        return pd.DataFrame(columns=["machine_id", "t_ms", "wall_time", "score", "threshold", "is_anomaly"])
    rows = []
    with open(SCORE_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def render_machine_card(machine_id, info, history_df):
    is_anomaly = info.get("is_anomaly", False)
    score = info.get("last_score")
    temp = info.get("last_temperature_c")
    windows_seen = info.get("windows_seen", 0)

    with st.container(border=True):
        header_col, status_col = st.columns([3, 1])
        with header_col:
            st.subheader(f"⚙️ {machine_id}")
        with status_col:
            if is_anomaly:
                st.error("🚨 ANOMALY", icon="🚨")
            elif score is None:
                st.info("Buffering…")
            else:
                st.success("✅ Normal")

        m1, m2, m3 = st.columns(3)
        m1.metric("Anomaly score", f"{score:.4f}" if score is not None else "—")
        m2.metric("Temperature", f"{temp:.1f} °C" if temp is not None else "—")
        m3.metric("Windows processed", windows_seen)

        machine_hist = history_df[history_df["machine_id"] == machine_id].tail(HISTORY_POINTS_SHOWN)
        if len(machine_hist) >= 2:
            machine_hist = machine_hist.copy()
            machine_hist["time"] = pd.to_datetime(machine_hist["wall_time"], unit="s")
            threshold_val = machine_hist["threshold"].iloc[-1]

            score_line = alt.Chart(machine_hist).mark_line(color="#3b6ea5").encode(
                x=alt.X("time:T", title=None),
                y=alt.Y("score:Q", title="anomaly score"),
            )
            anomaly_points = alt.Chart(machine_hist[machine_hist["is_anomaly"]]).mark_circle(
                color="#d9534f", size=60
            ).encode(x="time:T", y="score:Q")
            threshold_rule = alt.Chart(pd.DataFrame({"y": [threshold_val]})).mark_rule(
                color="#888888", strokeDash=[4, 4]
            ).encode(y="y:Q")

            chart = (score_line + anomaly_points + threshold_rule).properties(height=180)
            st.altair_chart(chart, width="stretch")
        else:
            st.caption("Collecting data for the trend chart…")

        if is_anomaly:
            threshold_display = machine_hist["threshold"].iloc[-1] if len(machine_hist) else None
            threshold_text = f"{threshold_display:.4f}" if threshold_display is not None else "the calibrated threshold"
            st.warning(
                f"**Action recommended:** {machine_id}'s reconstruction error "
                f"({score:.4f}) has crossed {threshold_text}. "
                f"Flag for inspection before failure. "
                f"(Email/notification hook: see `on_alert()` in `mqtt_bridge.py`.)"
            )


def main():
    st.title("⚙️ Predictive Maintenance Monitor")
    st.caption(
        "AI/IoT anomaly detection — live sensor readings and LSTM Autoencoder "
        "reconstruction-error anomaly scores per machine."
    )

    status = load_live_status()
    history_df = load_history()

    if status is None or not status.get("machines"):
        st.info(
            "No live data yet. Start the backend and a data source:\n\n"
            "```bash\n"
            "# terminal 1: local MQTT broker\n"
            "mosquitto -p 1883\n\n"
            "# terminal 2: the scoring backend\n"
            "python3 mqtt_bridge.py --broker localhost\n\n"
            "# terminal 3: simulate a machine (or flash the real Arduino R4 firmware)\n"
            "python3 mqtt_simulator.py --broker localhost --scenario degrading\n"
            "```"
        )
    else:
        updated_age_s = time.time() - status["updated_at"]
        stale = updated_age_s > 15
        cols = st.columns([1, 1, 2])
        cols[0].metric("Machines monitored", len(status["machines"]))
        cols[1].metric("Global threshold", f"{status['threshold']:.4f}")
        with cols[2]:
            if stale:
                st.warning(f"Last update {updated_age_s:.0f}s ago — backend may be idle/stopped.")
            else:
                st.caption(f"Last update: {updated_age_s:.1f}s ago")

        any_anomaly = any(m.get("is_anomaly") for m in status["machines"].values())
        if any_anomaly:
            flagged = [mid for mid, m in status["machines"].items() if m.get("is_anomaly")]
            st.error(f"🚨 **{len(flagged)} machine(s) flagged for inspection:** {', '.join(flagged)}")

        st.divider()
        machine_ids = sorted(status["machines"].keys())
        n_cols = min(2, len(machine_ids)) or 1
        cols = st.columns(n_cols)
        for i, mid in enumerate(machine_ids):
            with cols[i % n_cols]:
                render_machine_card(mid, status["machines"][mid], history_df)

    st.divider()
    st.caption(f"Auto-refreshing every {REFRESH_SECONDS}s.")
    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
