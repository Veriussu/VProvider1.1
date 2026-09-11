#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 8 - Yönetim Scripti
#  Dosya:    scripts/restart.sh
#  Amaç:     VProvider sunucusunu durdurup yeniden başlatır
#  Mekanik:  systemd kuruluysa "systemctl restart"; değilse
#            stop.sh + start.sh sırasıyla çağrılır.
#  Kullanım: scripts/restart.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="vprovider.service"

if command -v systemctl >/dev/null 2>&1 \
   && systemctl list-unit-files "${SERVICE}" >/dev/null 2>&1; then
  systemctl restart "${SERVICE}"
  echo " ✓ VProvider servisi yeniden başlatıldı (systemd: ${SERVICE})"
else
  "${PROJECT_ROOT}/scripts/stop.sh"
  "${PROJECT_ROOT}/scripts/start.sh"
fi