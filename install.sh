#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Kurulum Scripti
#  Dosya:    install.sh
#  Amaç:     Tek komutla VProvider kurulumu: donanım tespiti →
#            derleme bayrakları → venv + bağımlılıklar → llama-cpp-python
#            derleme → .env üretimi → systemd servisi (mümkünse)
#  Mekanik:  Donanım önceliği: CUDA → ROCm → SYCL → Vulkan → CPU.
#            CUDA derlemesi uzun sürdüğünden, llama_cpp içe aktarılabilir
#            durumdaysa yeniden derlenmez (--rebuild ile zorlanır).
#  Kullanım: bash install.sh                (otomatik tespit)
#            bash install.sh --cpu          (GPU yerine CPU derle)
#            bash install.sh --rebuild      (llama-cpp-python zorla derle)
#            bash install.sh --skip-build   (gerekse bile derlemeyi atla)
#            bash install.sh --no-systemd   (systemd kurulumunu atla)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

FORCE_CPU=0
FORCE_REBUILD=0
SKIP_BUILD=0
WITH_SYSTEMD=1
for arg in "$@"; do
  case "${arg}" in
    --cpu) FORCE_CPU=1 ;;
    --rebuild) FORCE_REBUILD=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --no-systemd) WITH_SYSTEMD=0 ;;
    -h|--help)
      echo " Kullanım: bash install.sh [--cpu] [--rebuild] [--skip-build] [--no-systemd]"; exit 0 ;;
    *) echo " Bilinmeyen bayrak: ${arg}" >&2; exit 1 ;;
  esac
done

# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────
LINE="--------------------------------------------------------------------------------"
banner() {
  printf '%s\n' "${LINE}"
  cat <<'ASCII'
   ██        ██  ████████████  ████████████    ████████    ██        ██  ████████████  ████████      ████████████  ████████████
   ██        ██  ██        ██  ██        ██  ██        ██  ██        ██      ██        ██        ██  ██            ██        ██
     ██    ██    ██        ██  ████████████  ██        ██    ██    ██        ██        ██        ██  ████████████  ████████████
     ██    ██    ████████████  ██  ██    ██  ██        ██    ██    ██        ██        ██        ██  ██            ██  ██    ██
       ████      ██            ██        ██    ████████        ████      ████████████  ████████      ████████████  ██        ██
ASCII
  printf '%s\n' "${LINE}"
  echo "  Hafif Yerel Yapay Zeka Model Sunucusu - Kurulum Başlıyor..."
  printf '%s\n' "${LINE}"
  echo "  Web:       https://veriussu.com"
  echo "  GitHub:    https://github.com/Veriussu/VProvider1.1"
  echo "  E-posta:   vprovider@veriussu.com  |  info@veriussu.com"
  printf '%s\n' "${LINE}"
}
banner

# ─────────────────────────────────────────────
# Adım 0 - Ön koşullar
# ─────────────────────────────────────────────
need() { command -v "$1" >/dev/null 2>&1 || { echo " ✗ Eksik bağımlılık: $1 (kurun ve tekrar deneyin)" >&2; exit 1; }; }
need python3
need pip
need git

PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [ "${PY_MAJOR}" -lt 3 ] || { [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -lt 10 ]; }; then
  echo " ✗ Python 3.10+ gerekli (mevcut ${PY_MAJOR}.${PY_MINOR})" >&2
  exit 1
fi
echo " ✓ Python ${PY_MAJOR}.${PY_MINOR} bulundu"

# ─────────────────────────────────────────────
# Adım 1 - Sanal ortam + bağımlılıklar
# ─────────────────────────────────────────────
if [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  echo " • Sanal ortam oluşturuluyor (.venv)..."
  python3 -m venv "${PROJECT_ROOT}/.venv"
fi
PY="${PROJECT_ROOT}/.venv/bin/python"
PIP="${PROJECT_ROOT}/.venv/bin/pip"
echo " ✓ Sanal ortam hazır"
"${PIP}" install --quiet --upgrade pip
"${PIP}" install --quiet -r requirements.txt -r requirements-dev.txt
echo " ✓ Bağımlılıklar kuruldu (requirements*.txt)"

# ─────────────────────────────────────────────
# Adım 2 - Donanım tespiti (CUDA -> ROCm -> SYCL -> Vulkan -> CPU)
# ─────────────────────────────────────────────
llama_kurulu_ve_iliskitli() {
  "${PY}" -c "import llama_cpp; print(llama_cpp.__version__)" >/dev/null 2>&1
}

# Kurulu derlemenin içerdiği backend adı (libggml-*.so dosyalarına bakılır)
llama_compiled_backend() {
  local libdir=${PROJECT_ROOT}/.venv/lib/python3.*/site-packages/llama_cpp/lib
  [ -d ${libdir} ] || { echo "cpu"; return; }
  for b in cuda:libggml-cuda.so rocm:libggml-hip.so sycl:libggml-sycl.so vulkan:libggml-vulkan.so; do
    local name="${b%%:*}"
    local so="${b##*:}"
    if [ -f ${libdir}/${so} ]; then echo "${name}"; return; fi
  done
  echo "cpu"
}

# Donanım tespitine göre istenen backend adı (pick_and_build ile aynı öncelik)
llama_desired_backend() {
  if [ "${FORCE_CPU}" -eq 1 ]; then echo "cpu"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then echo "cuda"; return; fi
  if command -v rocm-smi >/dev/null 2>&1; then echo "rocm"; return; fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qiE "vga.*intel|3d.*intel"; then echo "sycl"; return; fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qiE "vga.*(nvidia|amd)|3d.*(nvidia|amd)"; then echo "vulkan"; return; fi
  echo "cpu"
}

# İstenen backend ile kurulu backend uyumlu mu? (AMD/Intel için vulkan da geçer)
llama_backend_uyumlu() {
  local desired="$1"
  local compiled="$2"
  [ "${desired}" = "${compiled}" ] && return 0
  if [ "${desired}" = "cpu" ]; then
    return 1  # --cpu isteniyorsa yalnızca CPU derlemesi kabul edilir
  fi
  case "${desired}" in
    rocm|sycl) [ "${compiled}" = "vulkan" ] && return 0 ;;
  esac
  return 1
}

llama_derle() {
  local flags="$1"
  echo " • llama-cpp-python derleniyor (${flags}) — bu birkaç dakika sürebilir..."
  CMAKE_ARGS="${flags}" FORCE_CMAKE=1 "${PIP}" install --quiet --no-cache-dir --force-reinstall llama-cpp-python
}

pick_and_build() {
  local backend="CPU"
  if [ "${FORCE_CPU}" -eq 1 ]; then
    backend="CPU (--cpu bayrağı ile zorlandı)"
    llama_derle "-DGGML_CPU=on"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    backend="NVIDIA CUDA"
    llama_derle "-DGGML_CUDA=on" || { echo " • CUDA derlemesi başarısız, Vulkan deneniyor..."; llama_derle "-DGGML_VULKAN=on"; }
  elif command -v rocm-smi >/dev/null 2>&1; then
    backend="AMD ROCm"
    llama_derle "-DGGML_HIPBLAS=on -DAMDGPU_TARGETS=all" || { echo " • ROCm derlemesi başarısız, Vulkan deneniyor..."; llama_derle "-DGGML_VULKAN=on"; }
  elif command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qiE "vga.*intel|3d.*intel"; then
    backend="Intel (SYCL denemesi)"
    llama_derle "-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \
      || { echo " • SYCL derlemesi başarısız, Vulkan deneniyor..."; llama_derle "-DGGML_VULKAN=on"; }
  elif command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qiE "vga.*(nvidia|amd)|3d.*(nvidia|amd)"; then
    backend="GPU (Vulkan)"
    llama_derle "-DGGML_VULKAN=on"
  else
    backend="CPU (varsayılan)"
    llama_derle "-DGGML_CPU=on"
  fi
  echo " ✓ Motor: ${backend}"
}

# Derleme kararı: --rebuild zorlar; --skip-build atlatır; bunların dışında
# algılanan donanımla kurulu derleme uyumsuzsa otomatik olarak yeniden derlenir.
compiled="$(llama_compiled_backend)"
desired="$(llama_desired_backend)"
needs_build=0
build_reason=""
if [ "${FORCE_REBUILD}" -eq 1 ]; then
  needs_build=1
  build_reason="--rebuild bayrağı verildi"
elif [ "${SKIP_BUILD}" -eq 1 ]; then
  needs_build=0
elif ! llama_backend_uyumlu "${desired}" "${compiled}"; then
  needs_build=1
  build_reason="mevcut derleme (${compiled}) algılanan donanım (${desired}) ile uyumsuz"
fi

if [ "${needs_build}" -eq 1 ]; then
  echo " • Motor yeniden derlenecek: ${build_reason}"
  pick_and_build
elif llama_kurulu_ve_iliskitli; then
  echo " ✓ llama-cpp-python zaten kurulu ve uyumlu (motor yeniden derlenmedi)"
else
  pick_and_build
fi

# ─────────────────────────────────────────────
# Adım 3 - .env üretimi
# ─────────────────────────────────────────────
if [ ! -f "${PROJECT_ROOT}/.env" ]; then
  cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
  echo " • .env oluşturuldu (gerekirse değerleri düzenleyin)"
else
  echo " • Mevcut .env korundu"
fi

# ─────────────────────────────────────────────
# Adım 4 - systemd servisi (yetki varsa)
# ─────────────────────────────────────────────
host="$(grep -E '^HOST=' "${PROJECT_ROOT}/.env" | tail -1 | cut -d= -f2- || true)"
host="${host:-0.0.0.0}"
port="$(grep -E '^PORT=' "${PROJECT_ROOT}/.env" | tail -1 | cut -d= -f2- || true)"
port="${port:-9055}"

SUDO=""
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  SUDO="sudo"
fi

install_systemd() {
  local unit="/etc/systemd/system/vprovider.service"
  # Şablondaki yer tutucuları gerçek değerlerle değiştirip yetkili olarak yazar
  sed -e "s|{{USER}}|$(id -un)|g" \
      -e "s|{{PROJECT_ROOT}}|${PROJECT_ROOT}|g" \
      -e "s|{{HOST}}|${host}|g" \
      -e "s|{{PORT}}|${port}|g" \
      "${PROJECT_ROOT}/deploy/vprovider.service" | ${SUDO} tee "${unit}" >/dev/null
  ${SUDO} systemctl daemon-reload
  ${SUDO} systemctl enable --now vprovider.service
  echo " ✓ systemd servisi kuruldu ve başlatıldı: ${unit}"
}

if [ "${WITH_SYSTEMD}" -eq 1 ]; then
  if [ -n "${SUDO}" ]; then
    install_systemd
  else
    echo " • systemd kurulumu atlandı (parolasız sudo yok)."
    echo "   Elle kurulum: bash ${PROJECT_ROOT}/scripts/start.sh"
  fi
else
  echo " • systemd kurulumu --no-systemd ile atlandı"
fi

# ─────────────────────────────────────────────
# Adım 5 - Komut kısayolları (/usr/local/bin) — yetki varsa
# ─────────────────────────────────────────────
if [ -n "${SUDO}" ] && [ -d /usr/local/bin ]; then
  for name in start stop restart download remove; do
    ${SUDO} ln -sf "${PROJECT_ROOT}/scripts/${name}.sh" "/usr/local/bin/vprovider-${name}"
  done
  echo " ✓ Komut kısayolları: vprovider-start / -stop / -restart / -download / -remove"
else
  echo " • Komut kısayolları atlandı (yetki yok). Alternatif: scripts/vprovider-* doğrudan kullanılır."
fi

# ─────────────────────────────────────────────
# Adım 6 - Özet
# ─────────────────────────────────────────────
echo " ✔ Kurulum tamamlandı!"
echo "   - Panel:      http://${host}:${port}/     (ilk açılışta kurulum sihirbazı)"
echo "   - API:        http://${host}:${port}/v1/models   (OpenAI uyumlu)"
echo "   - Başlat:     ${PROJECT_ROOT}/scripts/start.sh"
echo "   - Durdur:     ${PROJECT_ROOT}/scripts/stop.sh"
echo "   - Test:       ${PROJECT_ROOT}/.venv/bin/pytest tests/ -q"
exit 0