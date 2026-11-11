#!/usr/bin/env bash
# Creates a private CA and a broker certificate. Usage: ./gen_certs.sh <broker-hostname-or-IP> [outdir]
set -euo pipefail
HOST="${1:?usage: gen_certs.sh <broker-hostname-or-IP> [outdir]}"; OUT="${2:-mosquitto/certs}"
mkdir -p "$OUT"; cd "$OUT"
openssl genrsa -out ca.key 4096 2>/dev/null
openssl req -x509 -new -key ca.key -sha256 -days 1825 -subj "/CN=PdM Local CA" -out ca.crt
openssl genrsa -out server.key 2048 2>/dev/null
openssl req -new -key server.key -subj "/CN=$HOST" -out server.csr
if [[ "$HOST" =~ ^[0-9.]+$ ]]; then SAN="IP:$HOST,DNS:localhost,IP:127.0.0.1"; else SAN="DNS:$HOST,DNS:localhost,IP:127.0.0.1"; fi
printf "subjectAltName=%s\n" "$SAN" > san.ext
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 825 -sha256 -extfile san.ext -out server.crt 2>/dev/null
rm -f server.csr san.ext ca.srl; chmod 600 ca.key server.key
echo "Created $OUT/{ca.crt,server.crt,server.key}. Keep ca.key private; give ca.crt to clients (--cafile)."
