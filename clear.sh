#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 8 - Yönetim Scripti
#  Dosya:    clear.sh
#  Amaç:     scripts/remove.sh için geriye uyumlu takma ad
#  Mekanik:  Argümanları olduğu gibi remove.sh'e iletir.
#  Kullanım: clear.sh            (modeller korunur)
#            clear.sh --all      (modeller dahil silinir)
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_ROOT}/scripts/remove.sh" "$@"