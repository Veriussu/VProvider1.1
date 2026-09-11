#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti
#  Dosya:    scripts/remove.sh
#  Amaç:     VProvider'ı sistemden tamamen kaldırır
#  Mekanik:  - Servisi durdurur ve systemd birimini siler
#            - /usr/local/bin altındaki vprovider-* bağlarını kaldırır
#            - Sanal ortamı (.venv), Python önbelleklerini siler
#            - data/ (veritabanı, kullanıcılar) ve runtime loglarını temizler
#            - Varsayılan: models/ ve kaynak kod korunur
#            - --all: modeller + kaynak kod + .env dahil proje dizini silinir
#  Kullanım: scripts/remove.sh        (modeller ve kod korunur)
#            scripts/remove.sh --all  (projenin tamamı silinir)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="vprovider.service"

ALL="${1:-}"
if [ "${ALL}" = "--all" ]; then
  echo " ⚠  Tüm modeller, kaynak kod, .env ve veriler silinecek."
  echo "   3 saniye içinde iptal için Ctrl+C."
  sleep 3
fi

# 1) Servisi durdur ve kaldır
if command -v systemctl >/dev/null 2>&1 \
   && systemctl list-unit-files "${SERVICE}" >/dev/null 2>&1; then
  systemctl stop "${SERVICE}" 2>/dev/null || true
  systemctl disable "${SERVICE}" 2>/dev/null || true
  sudo rm -f "/etc/systemd/system/${SERVICE}" || rm -f "/etc/systemd/system/${SERVICE}" 2>/dev/null || true
  systemctl daemon-reload 2>/dev/null || true
  echo " ✓ systemd servisi kaldırıldı"
fi

# 2) Komut bağlarını kaldır (varsa)
if [ -d /usr/local/bin ]; then
  for name in start stop restart download remove; do
    link="/usr/local/bin/vprovider-${name}"
    if [ -L "${link}" ]; then
      sudo rm -f "${link}" || rm -f "${link}" 2>/dev/null || true
    fi
  done
  echo " ✓ /usr/local/bin vprovider-* bağları kaldırıldı (varsa)"
fi

# 3) Python/venv katmanını sil
if pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
fi
rm -rf "${PROJECT_ROOT}/.venv"
find "${PROJECT_ROOT}" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
echo " ✓ Sanal ortam ve önbellekler silindi"

# 4) Veri ve logları temizle (gitignore korumalı klasörlerin içeriği)
rm -rf "${PROJECT_ROOT}/data"/* "${PROJECT_ROOT}/runtime" 2>/dev/null || true
mkdir -p "${PROJECT_ROOT}/data"
touch "${PROJECT_ROOT}/data/.gitkeep"
echo " ✓ data/ ve runtime logları temizlendi"

# 5) Modeller ve proje dizini
if [ "${ALL}" = "--all" ]; then
  rm -rf "${PROJECT_ROOT}/models"/* 2>/dev/null || true
  echo " ✓ models/ içeriği silindi"
  # Güvenlik: proje dizinini kendimizi çalıştırarak değil, komut yoluyla sil
  # (kendi dosyamız zaten çalışıyor; silmeyi en son adıma bırakırız)
else
  echo " • models/ içeriği korundu (tam silme için: $0 --all)"
fi

# 6) Tüm proje dizinini kaldır (yalnızca --all)
if [ "${ALL}" = "--all" ]; then
  # Bulunduğumuz konum en az dosya sistemi kökü değilse güvenli olsun
  if [ "${PROJECT_ROOT}" = "/" ]; then
    echo " ✗ Güvenlik: proje kökü / olamaz; silme iptal edildi" >&2
    exit 1
  fi
  # Önce çalışma dizinini dışarı al (kendimizi silerken ayakta kalalım)
  cd /tmp
  rm -rf "${PROJECT_ROOT}"
  echo " ✓ Proje dizini silindi: ${PROJECT_ROOT}"
else
  echo " • Proje kaynak kodu korundu (kaldırmak için: $0 --all)"
fi

echo " ✓ VProvider kaldırma tamamlandı."