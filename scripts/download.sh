#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti
#  Dosya:    scripts/download.sh
#  Amaç:     HuggingFace'ten GGUF modeli indirir
#  Mekanik:  Uygulamanın kendi hf_downloader modülünü kullanır;
#            dosya adı verilmezse repodaki GGUF dosyalarını listeler,
#            tek dosya verilirse o indirilir. models/ klasörüne iner
#            (MODELS_DIR .env'den okunur). HF_TOKEN .env'den alınır.
#  Kullanım: scripts/download.sh org/model
#            scripts/download.sh org/model ornek-Q4_K_M.gguf
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PY="${PROJECT_ROOT}/.venv/bin/python"

if ! command -v "${PY}" >/dev/null 2>&1; then
  echo " ✗ Sanal ortam bulunamadı: önce install.sh çalıştırılmalı" >&2
  exit 1
fi
if [ "$#" -lt 1 ]; then
  echo " Kullanım: $0 org/model [dosya.gguf]" >&2
  exit 1
fi

REPO_ID="$1"
FILENAME="${2:-}"

models_dir="$(grep -E '^MODELS_DIR=' "${PROJECT_ROOT}/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
models_dir="${models_dir:-models}"

"${PY}" - "${REPO_ID}" "${FILENAME}" "${models_dir}" <<'PYEOF'
import sys
import time

from app.hf_downloader import download_model, get_repo_files

repo_id, filename, models_dir = sys.argv[1], sys.argv[2], sys.argv[3]


def progress(received, total):
    if total:
        pct = received * 100 / total
        mb = received / 1_048_576
        print(f"\r  %5.1f%%  {mb:8.1f} MB / {total/1_048_576:8.1f} MB" % pct, end="", flush=True)


if not filename:
    try:
        files = get_repo_files(repo_id)
    except Exception as exc:
        sys.exit(f" ✗ Repo dosyaları alınamadı: {exc}")
    if not files:
        sys.exit(f" ✗ '{repo_id}' içinde GGUF dosyası bulunamadı.")
    print(f" '{repo_id}' reposundaki GGUF dosyaları:")
    for i, f in enumerate(files):
        mark = "  <-- önerilen" if "Q4_K_M" in f.filename else ""
        print(f"  {i+1}) {f.filename:60s} {f.size_bytes/1_048_576:8.1f} MB{mark}")
    print(" Kullanım: $0 org/model <dosya_adı>")
    sys.exit(0)

print(f" İndiriliyor: {repo_id} / {filename}")
start = time.time()
result = download_model(repo_id, filename, models_dir=models_dir, progress_cb=progress)
elapsed = time.time() - start
print(f"\n ✓ Tamamlandı: {result.path}  ({elapsed:.1f} sn)")
sys.exit(0 if not getattr(result, "error", None) else 1)
PYEOF