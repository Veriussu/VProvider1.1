#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 11 - ComfyUI Checkpoint İndirici
#  Dosya:    scripts/comfyui-checkpoint.sh
#  Amaç:     HuggingFace'teki bir checkpoint dosyasını ComfyUI'nin
#            models/checkpoints/ klasörüne indirir (SD/SDXL/FLUX).
#  Mekanik:  HF dosya URL'si (huggingface.co/.../resolve/main/...)
#            wget/curl ile, sürdürülebilir (resume) modda iner.
#  Kullanım: scripts/comfyui-checkpoint.sh <repo_id> <dosya_yolu> [hedef_adı]
#  Örnek:    scripts/comfyui-checkpoint.sh stabilityai/sdxl-1.5 single_file_stable_diffusion_xl_base_1.0.safetensors
# ─────────────────────────────────────────────────────────────

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Kullanım: scripts/comfyui-checkpoint.sh <repo_id> <dosya_yolu> [hedef_adı]" >&2
  echo "Örnek:    scripts/comfyui-checkpoint.sh stabilityai/sd-1.5 v1-5-pruned-emaonly.safetensors" >&2
  exit 2
fi

REPO_ID="$1"
FILE="$2"
TARGET="${3:-$(basename "${FILE}")}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMFY_DIR="${COMFY_DIR:-${HOME}/ComfyUI}"
if grep -qE '^COMFYUI_DIR=' "${PROJECT_ROOT}/.env" 2>/dev/null; then
  COMFY_DIR="$(grep -E '^COMFYUI_DIR=' "${PROJECT_ROOT}/.env" | tail -1 | cut -d= -f2-)"
  COMFY_DIR="${COMFY_DIR:-${HOME}/ComfyUI}"
fi

DEST_DIR="${COMFY_DIR}/models/checkpoints"
DEST="${DEST_DIR}/${TARGET}"
mkdir -p "${DEST_DIR}"

URL="https://huggingface.co/${REPO_ID}/resolve/main/${FILE}"

echo " ⇣ ${REPO_ID}/${FILE}"
echo "   → ${DEST}"

# sürdürebilir indirme: wget yoksa curl ile devam eder
if command -v wget >/dev/null 2>&1; then
  wget -c -O "${DEST}" "${URL}"
else
  curl -L -C - -o "${DEST}" "${URL}"
fi

echo " ✓ İndirme tamam: ${DEST}"
echo "   Panelde Görsel Üretim sekmesinde checkpoint'i yeniden yükleyin."