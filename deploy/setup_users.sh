#!/usr/bin/env bash
# Creates the Mosquitto password file.
# Usage: DEVICE_PW=... BACKEND_PW=... ./setup_users.sh [outdir]   (default: ./mosquitto)
set -euo pipefail
: "${DEVICE_PW:?set DEVICE_PW}" "${BACKEND_PW:?set BACKEND_PW}"
OUT="${1:-$(dirname "$0")/mosquitto}"
mkdir -p "$OUT"; rm -f "$OUT/passwd"; touch "$OUT/passwd"
mosquitto_passwd -b "$OUT/passwd" pdm_device "$DEVICE_PW"
mosquitto_passwd -b "$OUT/passwd" pdm_backend "$BACKEND_PW"
chmod 600 "$OUT/passwd"; echo "Wrote $OUT/passwd"
