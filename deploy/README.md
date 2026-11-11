# Secured backend (Mosquitto + TimescaleDB + Grafana)

Replaces the public `broker.hivemq.com` (anyone could read your sensor data).

| File | Purpose | Tested? |
|---|---|---|
| `mosquitto/mosquitto.conf` | auth required; 1883 (LAN, password) and 8883 (TLS) | yes, real Mosquitto: anonymous/wrong password rejected, TLS works |
| `mosquitto/acl` | devices may only publish sensor topics; backend reads everything | yes, by message flow |
| `gen_certs.sh`, `setup_users.sh` | private CA + broker cert; password file | yes |
| `docker-compose.yml` | broker + TimescaleDB + Grafana | **no** (no Docker available) |
| `grafana/` | datasource + 5-panel dashboard | dashboard SQL tested on PostgreSQL; Grafana itself not run |

## Steps
```
cp .env.example .env                       # edit passwords
./gen_certs.sh <this-PC-hostname-or-IP>    # broker certificate (SAN must match what clients use)
set -a; . ./.env; set +a
DEVICE_PW=$MQTT_DEVICE_PW BACKEND_PW=$MQTT_BACKEND_PW ./setup_users.sh
docker compose up -d
```
Run the backend on the host:
```
export PDM_PG_DSN=postgresql://pdm:$PG_PASSWORD@localhost:5432/pdm
python3 src/mqtt_bridge.py --broker <host> --tls --cafile deploy/mosquitto/certs/ca.crt \
        --user pdm_backend --password "$MQTT_BACKEND_PW"
python3 src/mqtt_simulator.py --broker <host> --user pdm_device --password "$MQTT_DEVICE_PW"
```
(Prefer `PDM_MQTT_USER` / `PDM_MQTT_PASSWORD` / `PDM_MQTT_CAFILE` environment variables
over command-line passwords.) Grafana: http://localhost:3000, dashboard "PdM overview".

## Known issue: TensorFlow + psycopg2 crash
psycopg2-binary's SSL negotiation segfaults if TensorFlow was imported first in the
same process. The code avoids it (local connections skip SSL; the bridge imports
psycopg2 first; the ONNX backend never loads TensorFlow). For a remote database
with SSL, use `--backend onnx`.

TimescaleDB hypertables are created if the extension exists; this was **not**
tested (only plain PostgreSQL tables were).
