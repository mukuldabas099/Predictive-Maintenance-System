"""
alerting.py
-------------
Real email alerting for the project brief's "System Output" requirement:
"the system triggers an alert (e.g., notification/email)". This replaces the
console-only `on_alert()` stub in mqtt_bridge.py with an actual SMTP send.

Configuration is via environment variables (never hardcode credentials in
source control):
    PDM_SMTP_HOST       e.g. smtp.gmail.com
    PDM_SMTP_PORT       e.g. 587 (STARTTLS) or 465 (SSL)
    PDM_SMTP_USER       your SMTP username (often your full email address)
    PDM_SMTP_PASSWORD   your SMTP password or app-specific password
    PDM_ALERT_FROM      "from" address (often same as PDM_SMTP_USER)
    PDM_ALERT_TO        comma-separated recipient list, e.g. "a@x.com,b@y.com"
    PDM_SMTP_SECURITY   "starttls" (default, port 587) | "ssl" (port 465) | "none" (local test server only)

*** YOU MUST DO THIS PART *** -- see README's "Real email alerts" section for
the exact steps to get working SMTP credentials (e.g. a Gmail App Password)
and set these environment variables. Without them, this module fails soft:
it logs a clear one-line note and the bridge keeps running with console-only
alerts, exactly like before -- nothing breaks, you just don't get email until
you configure it.

Verified in this sandbox using a local aiosmtpd debug server (no real
internet email was sent/received, and none of my test credentials are real
-- see the "Verified" note in the README for exactly what that confirmed and
didn't confirm).
"""

import os
import smtplib
import ssl
from email.message import EmailMessage


def _config_from_env():
    host = os.environ.get("PDM_SMTP_HOST")
    port = os.environ.get("PDM_SMTP_PORT")
    user = os.environ.get("PDM_SMTP_USER")
    password = os.environ.get("PDM_SMTP_PASSWORD")
    from_addr = os.environ.get("PDM_ALERT_FROM", user)
    to_addrs = os.environ.get("PDM_ALERT_TO", "")
    # PDM_SMTP_SECURITY: "starttls" (default, port 587), "ssl" (port 465),
    # or "none" (plain SMTP, no login -- ONLY for a local dev/test mail server).
    security = os.environ.get("PDM_SMTP_SECURITY", "").lower()
    if not security:  # backwards-compatible with the older PDM_SMTP_USE_SSL flag
        security = "ssl" if os.environ.get("PDM_SMTP_USE_SSL", "0") == "1" else "starttls"

    needs_login = security != "none"
    required = [host, port, from_addr, to_addrs] + ([user, password] if needs_login else [])
    if not all(required):
        return None
    return {
        "host": host, "port": int(port), "user": user, "password": password,
        "from_addr": from_addr, "to_addrs": [a.strip() for a in to_addrs.split(",") if a.strip()],
        "security": security,
    }


class EmailAlerter:
    def __init__(self, config=None):
        """config: dict as returned by _config_from_env(), or None to read
        from environment variables (the normal usage)."""
        self.config = config if config is not None else _config_from_env()
        self.enabled = self.config is not None
        if not self.enabled:
            print("NOTE: email alerting not configured (missing PDM_SMTP_* "
                  "environment variables) -> alerts will only print to console. "
                  "See README's 'Real email alerts' section to enable this.")

    def send_alert(self, machine_id, score, threshold, extra_context=""):
        if not self.enabled:
            return False

        msg = EmailMessage()
        msg["Subject"] = f"\U0001F6A8 Predictive Maintenance Alert: {machine_id}"
        msg["From"] = self.config["from_addr"]
        msg["To"] = ", ".join(self.config["to_addrs"])
        msg.set_content(
            f"Machine: {machine_id}\n"
            f"Anomaly score: {score:.5f}\n"
            f"Calibrated threshold: {threshold:.5f}\n"
            f"{extra_context}\n\n"
            f"This machine's vibration/temperature pattern has diverged from "
            f"its learned healthy baseline. Recommend flagging for inspection "
            f"before failure.\n\n"
            f"-- AI/IoT Predictive Maintenance System"
        )

        try:
            sec = self.config["security"]
            if sec == "ssl":
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.config["host"], self.config["port"], context=context, timeout=10) as server:
                    server.login(self.config["user"], self.config["password"])
                    server.send_message(msg)
            else:
                with smtplib.SMTP(self.config["host"], self.config["port"], timeout=10) as server:
                    if sec == "starttls":
                        server.starttls(context=ssl.create_default_context())
                    if sec != "none":
                        server.login(self.config["user"], self.config["password"])
                    server.send_message(msg)
            print(f"  Email alert sent to {', '.join(self.config['to_addrs'])}")
            return True
        except Exception as e:
            print(f"WARNING: failed to send email alert ({e}). "
                  f"Continuing with console-only alerting.")
            return False


if __name__ == "__main__":
    alerter = EmailAlerter()
    if alerter.enabled:
        ok = alerter.send_alert("TEST_MACHINE", score=5.4321, threshold=1.1609,
                                 extra_context="(this is a test alert from alerting.py)")
        print("Send result:", ok)
    else:
        print("Set PDM_SMTP_* environment variables and re-run to test a real send.")
