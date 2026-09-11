#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti
#  Dosya:    scripts/stop.sh
#  Amaç:     VProvider sunucusunu durdurur
#  Mekanik:  systemd servisi kurulmuşsa "systemctl stop" kullanır;
#            aksi halde çalışan uvicorn sürecini sonlandırır.
#  Kullanım: scripts/stop.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="vprovider.service"

if command -v systemctl >/dev/null 2>&1 \
   && systemctl list-unit-files "${SERVICE}" >/dev/null 2>&1; then
  systemctl stop "${SERVICE}"
  echo " ✓ VProvider servisi durduruldu (systemd: ${SERVICE})"
elif pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
  pkill -f "uvicorn app.main:app"
  for _ in 1 2 3 4 5; do
    pgrep -f "uvicorn app.main:app" >/dev/null 2>&1 || break
    sleep 1
  done
  echo " ✓ VProvider durduruldu"
else
  echo " ✓ VProvider zaten çalışmıyor"
fi