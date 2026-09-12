# VProvider — Hafif Yerel Yapay Zeka Model Sunucusu

> **Lisans:** Bu yazılımın tüm hakları **Veriüssü (veriussu.com)** firmasına aittir.
> Ticarî kullanım **yasaktır**; ticarî kullanım için `info@veriussu.com` ile iletişime
> geçin. Ayrıntı: [LICENSE](LICENSE)

VProvider, **GGUF** formatındaki açık kaynak modelleri lokalinizde (veya LAN'ınızda)
çalıştıran hafif bir sunucudur. OpenAI uyumlu `/v1/*` API'si sayesinde Chatbox,
SillyTavern, OpenAI SDK'ları gibi araçlarla doğrudan çalışır; web yönetim paneliyle
de HuggingFace'ten tek tıkla model indirir, bellek kullanımını yönetirsiniz.

> Karnındaki motor **llama.cpp**'dir; amacı model desteğini sınırsız tutarken
> çekirdeği hafif bırakmaktır. NVIDIA (CUDA), AMD (ROCm/Vulkan), Intel (SYCL) ve
> CPU üzerinde çalışır — kurulum donanımı otomatik algılar.

---

## Özellikler

- **Otomatik GPU algılama** — `GPU_MODE=auto` ile NVIDIA/AMD/Intel donanımını
  `nvidia-smi`/`rocm-smi`/sysfs üzerinden algılar; kurulu derlemeye göre CUDA,
  ROCm, Vulkan, SYCL veya CPU'yu seçer. `install.sh` uyumsuzluğu fark edip
  motoru donanıma göre otomatik yeniden derler (ör. CUDA). `/panel/system/gpu`
  uç noktası anlık GPU durumunu verir.
- **Akıllı GPU katmanı hesabı** — `GPU_LAYERS=-1` (otomatik) seçiminde modelin
  başlığından katman sayısı ve **KV önbellek boyutu** okunur (bağlam dahil);
  VRAM'e sığan katman sayısı hesaplanır. VRAM yetmezse otomatik olarak CPU
  moduna düşülür, `gpu_layers` kullanıcıya bildirilir.
- **OpenAI uyumlu API** — `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses` (akışsız + SSE streaming, OpenAI hata yapısı)
- **Function / tool calling** — chat ve responses uç noktalarında araç çağrısı,
  `tool_choice` ve paralel tool call yanıtları
- **Web yönetim paneli** — ilk kurulum sihirbazı, şifreli giriş (session cookie),
  model listesi, yükle/boşalt, bellek modu, silme
- **HuggingFace entegrasyonu** — GGUF arama, dosya listeleme, kesintisiz devam
  edebilen (resumable) indirme, tek tıkla model yükleme
- **Dinamik bellek yönetimi** — `keep` (her daim hazır) ve `dynamic` (boşta boşalt)
  modları; modeller boştayken **0 MB VRAM**
- **Görsel üretim (ComfyUI köprüsü)** — OpenAI uyumlu `/v1/images/generations`,
  panelde "Görsel Üretim" sekmesi; LLM kullanım bitince GPU'yu bıraktığı için
  görsel motoruyla VRAM yarışmaz
- **Video üretim (AnimateDiff köprüsü)** — `/v1/videos/generations` + panelde
  "Video Üretim" sekmesi; kareler sunucuda GIF'e birleştirilir
- **Ses üretim (TTS köprüsü)** — OpenAI uyumlu `/v1/audio/speech` (edge-tts,
  Türkçe sesler) + panelde "Ses Üretim" sekmesi; GPU gerekmez, internet ister
- **Panel içi sohbet** — "Sohbet" sekmesiyle seçili modeli arayüzden test etme;
  `/panel/chat` uç noktası açık API anahtarı istemez, yalnızca panel oturumu
- **API Anahtarları sekmesi** — birden çok isimlendirilmiş API anahtarı oluşturma,
  listeleme, kopyalama ve silme; eski tek `settings.api_key` otomatik "Varsayılan"
  anahtarına taşınır
- **Güvenlik** — panel kullanıcı adı/şifre, `/v1/*` API anahtarı (`Bearer`) ile ayrı
  ayrı korunur; şifreler bcrypt ile saklanır
- **Değişmez sistem kimliği** — Proje adı, GitHub/domain/mail bağlantıları, logo ve
  favicon **kod içine gömülüdür** (`app/config.py` → `SITE_IDENTITY` + `static/logo.png`,
  `static/favicon.png`). Veritabanında saklanmaz, panelden değiştirilemez; her kurulum
  aynı kimliği taşır.
- **Yönetim scriptleri** — kurulum, başlat/durdur/yeniden başlat, model indir, tam silme
- **Farklı motorlar** — sahte motorla test (hızlı CI), gerçek llama.cpp motoruyla üretim

---

## Gereksinimler

- **Python 3.10+**
- **NVIDIA / AMD / Intel GPU** (opsiyonel — yoksa CPU modunda çalışır)
  - CUDA için: sürücü + CUDA Toolkit (llama.cpp derlemesi için)
- `git`, `python3`, `pip`

---

## İletişim

- **Web:** https://veriussu.com
- **GitHub:** https://github.com/Veriussu/VProvider1.1
- **E-posta:** vprovider@veriussu.com · info@veriussu.com

---

## Hızlı Kurulum

```bash
git clone https://github.com/Veriussu/VProvider1.1.git vprovider && cd vprovider
bash install.sh
```

`install.sh` aşağıdakileri sırayla yapar:

1. Donanım algılar (**CUDA → ROCm → SYCL → Vulkan → CPU**)
2. `.venv` sanal ortamını kurar ve bağımlılıkları yükler
3. Donanıma özel **llama-cpp-python** derler
4. `.env` oluşturur (yoksa)
5. Yetki varsa **systemd servisini** kurar ve başlatır (`vprovider.service`)
6. Yetki varsa `/usr/local/bin` altına `vprovider-*` kısayollarını bağlar

> Not: `llama-cpp-python` zaten kuruluysa yeniden derlenmez; zorlamak için
> `bash install.sh --rebuild`. CPU modunu zorlamak için `bash install.sh --cpu`.

Kurulum tamamlanınca tarayıcıda `http://<sunucu-ip>:9055/` adresini açın:
ilk açılışta kurulum sihirbazı yönetici hesabı ve API anahtarını üretir.

---

## Günlük Kullanım

| Komut | İşlev |
|---|---|
| `bash scripts/start.sh` | Sunucuyu başlatır |
| `bash scripts/stop.sh` | Sunucuyu durdurur |
| `bash scripts/restart.sh` | Sunucuyu yeniden başlatır |
| `bash scripts/download.sh org/model` | Repodaki GGUF dosyalarını listeler |
| `bash scripts/download.sh org/model dosya.gguf` | Modeli `models/` altına indirir |
| `scripts/remove.sh` | Sunucuyu, venv'i ve verileri kaldırır (modeller + kaynak kod korunur) |
| `scripts/remove.sh --all` | Projenin **tamamı** silinir: modeller, kaynak kod, `.env`, veriler |
| `clear.sh` | `remove.sh` için geriye uyumlu takma ad |

systemd kuruluysa scriptler `systemctl start/stop/restart vprovider` çağırır;
değilse uvicorn'u doğrudan yönetirler.

---

## Yapılandırma (`.env`)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `APP_NAME` | `VProvider` | Uygulama adı |
| `HOST` | `0.0.0.0` | Dinleme adresi (LAN için `0.0.0.0`) |
| `PORT` | `9055` | Dinleme portu |
| `MEMORY_MODE` | `dynamic` | `keep` / `dynamic` |
| `IDLE_TIMEOUT_MINUTES` | `0` | dynamic modda boşta kalma süresi; `0` = kullanım bitince anında GPU'dan boşalt (çoklu model), `5`+ = CLI/opencode kullanımında model Bellek'te kalır |
| `GPU_MODE` | `auto` | `auto` / `cuda` / `rocm` / `sycl` / `vulkan` / `cpu` — backend ve katman seçimini belirler; `auto` donanımı algılar |
| `MODELS_DIR` | `models` | GGUF klasörü |
| `DATA_DIR` | `data` | Veritabanı klasörü |
| `DATABASE_PATH` | `data/vprovider.db` | SQLite dosyası |
| `CONTEXT_SIZE` | `4096` | Bağlam penceresi uzunluğu |
| `GPU_LAYERS` | `-1` | GPU'ya taşınan katman sayısı (`-1` = otomatik hesapla) |
| `THREADS` | `0` | CPU iş parçacığı (`0` = otomatik) |
| `HF_TOKEN` | *(boş)* | Gated HuggingFace repoları için |
| `COMFYUI_ENABLED` | `false` | Görsel üretim köprüsünü açar |
| `COMFYUI_HOST` | `127.0.0.1` | ComfyUI API adresi |
| `COMFYUI_PORT` | `8188` | ComfyUI API portu |
| `COMFYUI_DIR` | *(boş)* | ComfyUI kurulum dizini (`scripts/comfyui.sh` ve checkpoint indirici kullanır) |
| `COMFYUI_DEFAULT_CHECKPOINT` | *(boş)* | İstekte model belirtilmezse kullanılan checkpoint |
| `COMFYUI_DEFAULT_NEGATIVE` | `blur, ugly, low quality, watermark` | İstekte negatif prompt verilmezse kullanılır |
| `TTS_ENABLED` | `false` | Ses üretim (TTS) modülünü açar |
| `TTS_ENGINE` | `edge` | `edge` — edge-tts (çevrimiçi Microsoft motoru) |
| `TTS_VOICE` | `tr-TR-EmelNeural` | Varsayılan Türkçe ses |
| `TTS_VOICE_RATE` | `+0%` | Varsayılan hız (`-10%` yavaş, `+20%` hızlı) |

---

## API Kullanımı

Tüm `/v1/*` istekleri API anahtarı ister:

```bash
API_KEY="panelden-kopyalanan-anahtar"

# Model listesi
curl http://localhost:9055/v1/models \
  -H "Authorization: Bearer $API_KEY"

# Sohbet (akışsız)
curl http://localhost:9055/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-0.5b-instruct-q2_k",
       "messages": [{"role": "user", "content": "Merhaba!"}]}'

# Sohbet (stream / SSE)
curl -N http://localhost:9055/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-0.5b-instruct-q2_k",
       "messages": [{"role": "user", "content": "1 ve 1 kaç eder?"}],
       "stream": true}'
```

### Uç Noktalar

| Bölge | Yöntem | Yol | Açıklama |
|---|---|---|---|
| API | `GET` | `/v1/models` | Yüklü modellerin listesi |
| API | `POST` | `/v1/chat/completions` | Chat (akışsız / SSE) — tool calling destekli |
| API | `POST` | `/v1/responses` | OpenAI Responses API (Codex gibi araçlar için) |
| API | `POST` | `/v1/completions` | Metin tamamlama |
| API | `POST` | `/v1/images/generations` | Görsel üretim (ComfyUI), `url` veya `b64_json` |
| API | `GET` | `/v1/images/file/{prompt_id}/{i}` | Üretilen görsel (API anahtarı gerekir) |
| API | `GET` | `/v1/images/comfy/checkpoints` | ComfyUI checkpoint listesi |
| API | `POST` | `/v1/videos/generations` | Video üretim (AnimateDiff → GIF), url modu |
| API | `GET` | `/v1/videos/file/{prompt_id}` | Üretilen video (API anahtarı gerekir) |
| API | `GET` | `/v1/videos/comfy/checkpoints` | Video için checkpoint listesi |
| API | `POST` | `/v1/audio/speech` | Metni seslendirir → ham MP3 döner (OpenAI ile aynı) |
| API | `GET` | `/v1/audio/voices` | Türkçe ses listesi |
| API | `GET` | `/v1/audio/status` | TTS durumu (açık mı, kurulu mu) |
| Panel | `GET` | `/panel/status` | Kurulum durumu |
| Panel | `POST` | `/panel/setup` | İlk kayıt (yalnızca hiç kullanıcı yokken) |
| Panel | `POST` | `/panel/login` | Giriş (session cookie) |
| Panel | `GET` | `/panel/status` | `needs_setup` (kayıt mı giriş mi) sinyali |
| Panel | `GET` | `/panel/models` | Model yönetimi listesi |
| Panel | `POST` | `/panel/models/{id}/load` | Modeli belleğe yükle |
| Panel | `POST` | `/panel/models/{id}/unload` | Modeli boşalt |
| Panel | `POST` | `/panel/models/{id}/mode` | `keep` / `dynamic` |
| Panel | `POST` | `/panel/models/{id}/delete` | Diskten sil |
| Panel | `POST` | `/panel/chat` | Seçili modelle sohbet (açık API anahtarı gerekmez) |
| Panel | `GET` | `/panel/models/search` | HuggingFace arama |
| Panel | `POST` | `/panel/models/download` | Model indirme başlat |
| Panel | `GET` | `/panel/models/download/status` | İndirme ilerlemesi |
| Panel | `GET` | `/panel/apis` | API anahtarlarını listele (isimlendirilmiş) |
| Panel | `POST` | `/panel/apis` | Yeni isimlendirilmiş API anahtarı oluştur |
| Panel | `DELETE` | `/panel/apis/{id}` | API anahtarını sil |
| Panel | `GET` | `/panel/comfy/status` | ComfyUI köprüsü durumu (GPU bilgisi) |
| Panel | `GET` | `/panel/comfy/checkpoints` | ComfyUI checkpoint listesi |
| Panel | `POST` | `/panel/comfy/generate` | Görsel üret (panel içi, session cookie) |
| Panel | `GET` | `/panel/comfy/image/{prompt_id}/{i}` | Panel içinde üretilen görsel |
| Panel | `POST` | `/panel/video/generate` | Video üret (GIF, session cookie) |
| Panel | `GET` | `/panel/comfy/video/{prompt_id}` | Panel içinde üretilen video |
| Panel | `GET` | `/panel/tts/status` | TTS durumu |
| Panel | `POST` | `/panel/tts/generate` | Metni seslendir |
| Panel | `GET` | `/panel/tts/audio/{key}` | Üretilen sesi oynat |
| Panel | `GET` | `/` | Web yönetim paneli |
| Diğer | `GET` | `/health` | Sağlık kontrolü |

---

## Bellek Modları

- **keep** — Model ilk istekte yüklenir ve bellekte kalır; sonraki istekler
  gecikmesizdir. Bellekten kaldırmak için panelden "Boşalt" denir.
- **dynamic** — Model istek üzerine yüklenir ve kullanım bitince GPU'dan
  ayrılır:
  - `IDLE_TIMEOUT_MINUTES=0` (varsayılan): yanıt tamamlanır tamamlanmaz anında
    boşaltılır → tek GPU'da birden çok model arasında serbestçe geçiş yapılır,
    VRAM sürekli açıktır. **Chartbox/TTY yerine CLI (ör. opencode) kullanılıyorsa**
    yanıtların yüklenmesi için her istekte yeniden yükleme yaşanır; bu durumda
    `IDLE_TIMEOUT_MINUTES=5` (veya üstü) önerilir.
  - `IDLE_TIMEOUT_MINUTES>0`: o süre boyunca yeni istek gelmezse boşaltılır →
    **boşta 0 MB VRAM**.

---

## Testler

```bash
.venv/bin/pytest tests/ -q            # tüm birim + API + motor testleri
.venv/bin/pytest tests/test_e2e.py    # gerçek model + gerçek motor (uçtan uca)
VPROVIDER_E2E_GPU_LAYERS=0 .venv/bin/pytest tests/test_e2e.py   # CPU mod teyidi
```

Testler `llama-cpp-python` ve gerçek bir GGUF modeli yoksa otomatik atlanır;
sahte motorlarla hızlı çalışır, gerçek veritabanına asla dokunmaz.

---

## Görsel Üretim (ComfyUI)

VProvider görsel üretimi harici bir **ComfyUI** kurulumuyla (ayrı bir süreç,
varsayılan `127.0.0.1:8188`) konuşarak yapar. LLM `IDLE_TIMEOUT_MINUTES=0` ile
kullanım biter bitmez GPU'yu boşalttığından, görsel motoruyla VRAM yarışmaz.

Kısaca:

```bash
# 1) ComfyUI'yi kendiniz kurun (ayrıntı: deploy/comfyui-rehber.md)
# 2) .env'de COMFYUI_ENABLED=true (ve COMFYUI_DIR) ayarlayıp sunucuyu yeniden başlatın
# 3) scripts/comfyui.sh start        # motoru başlat
# 4) scripts/comfyui-checkpoint.sh stabilityai/sd-1.5 v1-5-pruned-emaonly.safetensors
# 5) Panel > Görsel Üretim sekmesinden veya /v1/images/generations ile üretin
```

```bash
# OpenAI uyumlu çağrı (url modu)
curl http://localhost:9055/v1/images/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "v1-5-pruned-emaonly.safetensors",
       "prompt": "güneşli orman, fotogerçekçi", "size": "512x512"}'
```

Yanıt `data[0].url` ile görseli `/v1/images/file/{prompt_id}/0` adresinden verir
(`response_format: "b64_json"` ile base64 de alınır). Üretilen görseller in-memory
depoda tutulur (en fazla 40 iş) — sunucu yeniden başlarsa eski görseller silinir.

**Not:** Varsayılan üretim workflow'u SD/SDXL ailesi içindir. FLUX gibi farklı
mimariler özel workflow gerektirir. RL/yerleşik yüksek performans için 8 GB VRAM
yeterlidir (512x512, 20 adım); büyük çözünürlükler veya video (SVD/Wan) sınırlıdır.

### Video Üretim (AnimateDiff)

Aynı köprü üzerinden kısa animasyonlar:

```bash
curl http://localhost:9055/v1/videos/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "v1-5-pruned-emaonly.safetensors",
       "prompt": "mavi gökyüzünde süzülen beyaz bulutlar",
       "size": "512x512", "frames": 16}'
```

Yanıttaki `data[0].url` (`/v1/videos/file/{prompt_id}`) animasyonlu GIF'i döner;
`mime_type`, `width`, `height`, `frames` meta bilgileri de gelir.

Video üretimi için ayrıca **ComfyUI Manager → "AnimateDiff Evolved"** node paketi
ve **hareket modülü** (`models/animate_diff/mm_sd_v15_v2.ckpt`, SD 1.5) gerekir —
ayrıntılar `deploy/comfyui-rehber.md`'deki Video bölümünde. 8 GB VRAM için
512x512 ve 16-24 kare önerilir; üretim görsele göre birkaç dakika sürebilir.

### Ses Üretim (TTS)

OpenAI'nin `/v1/audio/speech` imzasıyla metni konuşmaya çevirir; yanıt ham
MP3'tür (JSON değil).

```bash
# Ham ses döner; dosyaya kaydetmek için -o kullanın
curl http://localhost:9055/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "edge-tts", "input": "Merhaba! Bugün hava çok güzel.",
       "voice": "tr-TR-EmelNeural", "response_format": "mp3"}' \
  -o selam.mp3
```

Alanlar: `input` (zorunlu, ≤3000 karakter), `voice` (boşsa `TTS_VOICE`),
`speed` (0.5-2.0), `response_format` (`mp3`). Sesler: `GET /v1/audio/voices`
(Türkçe: `tr-TR-EmelNeural`, `tr-TR-AhmetNeural`). Edge-tts çevrimiçi bir
motordur — GPU gerektirmez ama internet bağlantısı ister.

---

## Proje Yapısı

```
vprovider/
├── app/                  # Uygulama kaynak kodu
│   ├── main.py           # Uygulama + router bağlama + panel ön yüzü
│   ├── config.py         # .env okuma + sabit sistem kimliği (SITE_IDENTITY)
│   ├── model_manager.py  # Bellek modları + yükleme/boşaltma
│   ├── gpu_detect.py     # Donanım/backend algılama + akıllı katman hesabı
│   ├── llama_backend.py  # llama.cpp sarmalayıcı (gerçek motor)
│   ├── auth.py           # Kullanıcı/şifre + API anahtarı
│   ├── user_store.py     # SQLite (kullanıcı, oturum, ayarlar, API anahtarları)
│   ├── hf_downloader.py  # HuggingFace arama + resumable indirme
│   ├── openai_api.py     # /v1/* router
│   ├── admin_api.py      # /panel/* router
│   ├── comfy_client.py   # ComfyUI HTTP istemcisi (görsel + video)
│   ├── comfy_api.py      # /v1/images/* + /v1/videos/*
│   ├── tts_backend.py    # Ses üretim motoru (edge-tts)
│   └── tts_api.py        # /v1/audio/*
├── static/index.html     # Yönetim paneli (tek dosya)
├── static/logo.png       # Gömülü proje logosu (değiştirilemez)
├── static/favicon.png    # Gömülü tarayıcı simgesi (değiştirilemez)
├── scripts/              # start/stop/restart/download/remove
├── scripts/comfyui.sh          # ComfyUI motoru başlat/durdur/durum (çalışmazsa 1 döner)
├── scripts/comfyui-checkpoint.sh  # HF checkpoint indirici
├── deploy/vprovider.service   # systemd şablonu
├── deploy/comfyui.service     # ComfyUI systemd şablonu (opsiyonel)
├── deploy/Caddyfile           # dış ağ/HTTPS örneği (opsiyonel)
├── deploy/caddy-rehber.md     # Caddy kurulum + sorun giderme rehberi
├── deploy/comfyui-rehber.md   # ComfyUI kurulum + checkpoint + sistem rehberi
├── tests/                # pytest (birim + API + uçtan uca)
├── models/               # GGUF modelleri
├── data/                 # SQLite veritabanı
├── install.sh            # Otomatik kurulum
├── clear.sh              # remove.sh takma adı
├── LICENSE               # Sınırlı kullanım lisansı (Veriüssü)
└── .env.example          # Ayarlar şablonu
```

---

## Kaldırma

```bash
bash scripts/remove.sh        # sunucu + venv + veriler kaldırılır; modeller ve kaynak kod kalır
bash scripts/remove.sh --all  # projenin tamamı: modeller, kaynak kod, .env, veriler silinir
```

İlki systemd servisini, `/usr/local/bin` kısayollarını, `.venv`'i, `data/` içeriğini ve
`runtime/` loglarını temizler; kaynak kod ve `models/` yerinde kalır. `--all` ile diskten
proje dizininin tamamı kaldırılır.

---

## Lisans

Bu yazılımın **tüm hakları [Veriüssü](https://veriussu.com) firmasına aittir**
(© 2026 Veriüssü). Ayrıntılı koşullar [LICENSE](LICENSE) dosyasında yer alır:

- **İzin verilen:** kişisel, eğitim ve ticarî olmayan kullanım.
- **YASAK:** ticarî kullanım — ürün/hizmet içinde, SaaS/barındırma olarak,
  danışmanlık dahil. Ticarî kullanım için `info@veriussu.com` adresinden yazılı
  onay alınmalıdır.
- Model ağırlıkları ilgili açık kaynak lisanslarına tabidir (HuggingFace'ten
  indirilir); bu lisans yalnızca yazılım kodunu kapsar.

## Notlar

- Sunucu varsayılanda `0.0.0.0:9055` dinler; dış ağa açmadan önce mutlaka panel
  kurulumunu tamamlayın ve API anahtarınızı koruyun.
- Dış ağ/domain + otomatik HTTPS için hazır rehber: `deploy/caddy-rehber.md`
  (örnek `deploy/Caddyfile` ile birlikte).