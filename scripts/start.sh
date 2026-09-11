#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 8 - Yönetim Scripti
#  Dosya:    scripts/start.sh
#  Amaç:     VProvider sunucusunu başlatır
#  Mekanik:  systemd servisi kurulmuşsa "systemctl start" kullanır;
#            aksi halde venv uvicorn'unu arka planda başlatır
#            (log: runtime/vprovider.log). PORT/HOST .env'den okunur.
#  Kullanım: scripts/start.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="vprovider.service"
LOG_DIR="${PROJECT_ROOT}/runtime"
LOG_FILE="${LOG_DIR}/vprovider.log"

# .env'den host/port değerlerini güvenle okur
host="$(grep -E '^HOST=' "${PROJECT_ROOT}/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
host="${host:-0.0.0.0}"
port="$(grep -E '^PORT=' "${PROJECT_ROOT}/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
port="${port:-9055}"

systemd_servisi_kurulu() {
  command -v systemctl >/dev/null 2>&1 \
    && systemctl list-unit-files "${SERVICE}" >/dev/null 2>&1
}

if systemd_servisi_kurulu; then
  systemctl start "${SERVICE}"
  systemctl enable "${SERVICE}" >/dev/null 2>&1 || true
  echo " ✓ VProvider servisi başlatıldı (systemd: ${SERVICE})"
else
  if pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
    echo " ✓ VProvider zaten çalışıyor (pid $(pgrep -f 'uvicorn app.main:app' | tr '\n' ' '))"
    exit 0
  fi
  mkdir -p "${LOG_DIR}"
  (cd "${PROJECT_ROOT}" && setsid nohup ./.venv/bin/uvicorn app.main:app --host "${host}" --port "${port}" < /dev/null >> "${LOG_FILE}" 2>&1 &)
  sleep 1
  if pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
    echo " ✓ VProvider başlatıldı: http://${host}:${port}  (log: ${LOG_FILE})"
  else
    echo " ✗ Sunucu başlatılamadı; log dosyasına bakın: ${LOG_FILE}" >&2
    exit 1
  fi
fi