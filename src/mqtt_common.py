"""
mqtt_common.py -- shared MQTT client setup (authentication + TLS) so the bridge
and the simulator connect to a secured broker the same way.

Credentials are read from --user/--password or the environment variables
PDM_MQTT_USER / PDM_MQTT_PASSWORD (preferred: they stay out of shell history).
"""
import os
import ssl

import paho.mqtt.client as mqtt


def add_mqtt_args(ap, default_port=1883):
    g = ap.add_argument_group("MQTT connection")
    g.add_argument("--broker", default="localhost")
    g.add_argument("--port", type=int, default=default_port,
                   help="1883 plain, 8883 TLS (default 1883, or 8883 with --tls)")
    g.add_argument("--user", default=os.environ.get("PDM_MQTT_USER"))
    g.add_argument("--password", default=os.environ.get("PDM_MQTT_PASSWORD"))
    g.add_argument("--tls", action="store_true", help="connect with TLS")
    g.add_argument("--cafile", default=os.environ.get("PDM_MQTT_CAFILE"),
                   help="CA certificate that signed the broker certificate")
    g.add_argument("--tls-insecure", action="store_true",
                   help="skip hostname check (lab use with self-signed certs only)")
    return ap


def make_client(args, client_id=""):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    if args.user:
        client.username_pw_set(args.user, args.password)
    if args.tls:
        client.tls_set(ca_certs=args.cafile, cert_reqs=ssl.CERT_REQUIRED)
        if args.tls_insecure:
            client.tls_insecure_set(True)
    return client


def effective_port(args):
    return 8883 if (args.tls and args.port == 1883) else args.port
