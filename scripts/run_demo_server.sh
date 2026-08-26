#!/usr/bin/env bash
# Chạy server demo Trợ lý thị giác.
#
#   ./scripts/run_demo_server.sh            # HTTPS 0.0.0.0:8443 (cho điện thoại)
#   TLS=0 ./scripts/run_demo_server.sh      # HTTP dev, chỉ 127.0.0.1:8000
#
# Camera của trình duyệt điện thoại CHỈ hoạt động trên secure context (HTTPS/WSS)
# hoặc localhost — vì vậy demo thật phải dùng chế độ TLS mặc định.
set -euo pipefail

ROOT="/home/ubuntu/ai-assistant"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
# Trong container không có .venv — rơi về python hệ thống của image
[[ -x "$PY" ]] || PY="$(command -v python3)"

# Proxy của máy này chậm/hỏng và làm hỏng cả hub-check cho model ĐÃ CACHE
# (README: 1.8 MB/s qua proxy vs 83 MB/s trực tiếp). Kết nối trực tiếp đã xác
# nhận hoạt động cho cả HF lẫn Google AI -> bỏ proxy cho toàn server.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd "$ROOT"

if [[ "${TLS:-1}" == "1" ]]; then
  CERT="$ROOT/certs/cert.pem"
  KEY="$ROOT/certs/key.pem"
  if [[ ! -f "$CERT" ]]; then
    echo "Chưa có cert — tạo cert tự ký..." >&2
    mkdir -p "$ROOT/certs"
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
      -keyout "$KEY" -out "$CERT" \
      -subj "/CN=ai-assistant-demo" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
    chmod 600 "$KEY"
  fi
  exec "$PY" -m uvicorn api_main:app \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8443}" \
    --ssl-certfile "$CERT" --ssl-keyfile "$KEY"
else
  exec "$PY" -m uvicorn api_main:app \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
fi
