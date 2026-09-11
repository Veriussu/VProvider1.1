#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    ComfyUI Yönetim Scripti
#  Dosya:    scripts/comfyui.sh
#  Amaç:     VProvider'ın görsel üretim köprüsü için gerekli olan
#            ComfyUI motorunu başlatır/durdurur/durum gösterir.
#  Mekanik:  systemd servisi (deploy/comfyui.service) kurulmuşsa
#            systemctl; aksi halde venv + nohup ile çalıştırır.
#  Kullanım: scripts/comfyui.sh {start|stop|status|restart}
# ─────────────────────────────────────────────────────────────

set -euo pipefail

ACTION="${1:-status}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Varsayılan ComfyUI dizini: ~/ComfyUI (deploy sürecinde kurulur).
COMFY_DIR="${COMFY_DIR:-${HOME}/ComfyUI}"
if grep -qE '^COMFYUI_DIR=' "${PROJECT_ROOT}/.env" 2>/dev/null; then
  COMFY_DIR="$(grep -E '^COMFYUI_DIR=' "${PROJECT_ROOT}/.env" | tail -1 | cut -d= -f2-)"
  COMFY_DIR="${COMFY_DIR:-${HOME}/ComfyUI}"
fi
PORT="$(grep -E '^COMFYUI_PORT=' "${PROJECT_ROOT}/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
PORT="${PORT:-8188}"
SERVICE="comfyui.service"
PIDFILE="${PROJECT_ROOT}/runtime/comfyui.pid"
LOGFILE="${PROJECT_ROOT}/runtime/comfyui.log"

systemd_servisi_kurulu() {
  command -v systemctl >/dev/null 2>&1 \
    && systemctl list-unit-files "${SERVICE}" >/dev/null 2>&1
}

case "${ACTION}" in
  start)
    if systemd_servisi_kurulu; then
      systemctl start "${SERVICE}"
      systemctl enable "${SERVICE}" >/dev/null 2>&1 || true
      echo " ✓ ComfyUI servisi başlatıldı (systemd)"
      exit 0
    fi
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo " ✓ ComfyUI zaten çalışıyor (pid $(cat "${PIDFILE}"))"
      exit 0
    fi
    if [ ! -x "${COMFY_DIR}/.venv/bin/python" ] || [ ! -f "${COMFY_DIR}/main.py" ]; then
      echo " ✗ ComfyUI bulunamadı: ${COMFY_DIR}" >&2
      echo "   Kurulum: deploy/comfyui-rehber.md adım 1-2'yi izleyin." >&2
      exit 1
    fi
    mkdir -p "$(dirname "${LOGFILE}")"
    (cd "${COMFY_DIR}" && setsid nohup ./.venv/bin/python main.py --listen 127.0.0.1 --port "${PORT}" < /dev/null >> "${LOGFILE}" 2>&1 & echo $! > "${PIDFILE}")
    sleep 2
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo " ✓ ComfyUI başlatıldı: http://127.0.0.1:${PORT}  (log: ${LOGFILE})"
    else
      echo " ✗ ComfyUI başlatılamadı; log: ${LOGFILE}" >&2
      exit 1
    fi
    ;;
  stop)
    if systemd_servisi_kurulu; then
      systemctl stop "${SERVICE}"
      echo " ✓ ComfyUI servisi durduruldu"
      exit 0
    fi
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      kill "$(cat "${PIDFILE}")"
      rm -f "${PIDFILE}"
      echo " ✓ ComfyUI durduruldu"
    else
      echo " ✓ ComfyUI zaten çalışmıyor"
    fi
    ;;
  status)
    if systemd_servisi_kurulu; then
      systemctl status "${SERVICE}" || true
      exit 0
    fi
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo " ✓ ComfyUI çalışıyor (pid $(cat "${PIDFILE}"), port ${PORT})"
    else
      echo " ✗ ComfyUI çalışmıyor (uyarı: COMFYUI_DIR=${COMFY_DIR}, port ${PORT})"
      exit 1
    fi
    ;;
  restart)
    "${PROJECT_ROOT}/scripts/comfyui.sh" stop
    "${PROJECT_ROOT}/scripts/comfyui.sh" start
    ;;
  *)
    echo "Kullanım: scripts/comfyui.sh {start|stop|status|restart}" >&2
    exit 2
    ;;
esac