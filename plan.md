# VProvider — Hafif Yerel Yapay Zeka Model Sunucusu (Plan v2)

## Proje Tanımı

Bu proje; Ollama veya LocalAI gibi yüksek kaynak tüketen araçlara bağımlı kalmadan, tamamen hafif (lightweight), yüksek performanslı ve donanıma duyarlı bir Yerel Yapay Zeka Model Sunucusu projesidir.

Yaklaşım: **Motor hafif, model desteği sınırsız.** Çekirdek (llama.cpp) hafif kalır; metin/chat/kodlama/düşünme/resim-anlama/ses tanıma modellerinin tamamı GGUF ile tek motorda çalışır. Görsel/video/müzik **üretim** modelleri ise ayrı, isteğe bağlı modüller olarak aynı panel ve API üzerinden sunulur.

Projenin temel amacı; yerel ağdaki (LAN) tüm cihazların (telefon, tablet, farklı bilgisayarlar) domain adına ihtiyaç duymadan, yalnızca IP adresi üzerinden OpenAI standartlarıyla erişebileceği bağımsız bir backend altyapısı sunmaktır. İsteğe bağlı olarak statik IP + domain + reverse proxy ile dış ağda da kullanılabilir.

## Temel Mimari Sütunlar

### 1. Donanıma Duyarlı Akıllı Kurulum (`install.sh`)
- Kurulum esnasında `nvidia-smi`, `rocm-smi` ve `lspci` taramaları yaparak NVIDIA (CUDA), AMD (ROCm/HIP), Intel (SYCL) veya CPU olup olmadığını otomatik tespit eder.
- `llama-cpp-python` kütüphanesini ilgili donanımın matris optimizasyon bayraklarıyla derler (`CMAKE_ARGS` üzerinden).
- **Vulkan fallback:** CUDA/ROCm SDK'sı kurulu değilse veya kurulum başarısız olursa `GGML_VULKAN` ile tek derleme NVIDIA + AMD + Intel kartlarını kapsar. (Vulkan SDK, GB boyutlu CUDA/ROCm SDK'larına göre çok daha hafiftir ve bağımlılık sorununu aşar.)
- Ekran kartı yoksa veya sürücü eksikse güvenli bir şekilde CPU modunda çalışır.
- Kurulum sonunda systemd servisini (örn. `vprovider.service`) oluşturur ve başlatır.

### 2. Tam Donanımlı Temizlik (`clear.sh`)
- Sistemde oluşturulan systemd arka plan servisini durdurur ve siler.
- Python sanal ortamını (venv) kaldırarak arkasında gereksiz yük bırakmadan sistemi temizler.

### 3. İki Farklı Bellek Modu (Kaynak Verimliliği)
- **Her Daim Hazır (Keep in Memory):** Model belleğe yüklenir ve sürekli hazır bekler. Sıfıra yakın yanıt gecikmesi sunar.
- **Kullanırken Yükle (Dynamic Offload - 0 MB Boşta VRAM):** İstek geldiğinde modeli VRAM/RAM'e alır, işlem bitip belirlenen süre (örn. 5 dakika) boyunca yeni istek gelmezse modeli bellekten tamamen siler. Sistem boşta iken 0 MB VRAM tüketir.
  - **Uygulama şekli (kritik detay):** `llama-cpp-python`'ın hazır OpenAI server'ı modeli her zaman bellekte tutar. Dinamik boşaltma için `model_manager.py`, her model için tembel (lazy) bir worker process yönetir: istek geldiğinde worker'ı başlatır/model yükler, idle timer dolunca `llama_backend_free()` + garbage collection ile modeli ve worker'ı kapatır. Belleğe kalıcı (keep) modda ise worker sürekli ayakta kalır.

### 4. OpenAI Standartlarında API
- `/v1/models`, `/v1/chat/completions` ve `/v1/completions` uç noktaları sunulur.
- Akışlı yanıtlar (Streaming / SSE) desteklenir.
- Çeşitli üçüncü taraf yazılımlar (Chatbox, Jan, SillyTavern, VS Code eklentileri vb.) OpenAI uyumu sayesinde sorunsuz bağlanır.
- **`GET /health`** uç noktası (kullanılabilirlik/monitoring) eklenir.

### 5. Güvenlik (v3 — kullanıcı tabanlı)
- **İlk kurulumda kullanıcı oluşturma:** Uygulama ilk kez çalıştırıldığında yönetici kullanıcı adı/şifre istenir (basit setup adımı). Şifreler tek yönlü hash (örn. bcrypt/argon2) ile saklanır, düz metin asla tutulmaz.
- **Panel girişi:** Web yönetim paneline yalnızca kullanıcı adı + şifre ile (session cookie tabanlı) giriş yapılır. Panel dış ağ açıkken de korunur — dış ağa açılacaksa HTTPS + rate limiting ek öneri.
- **API (`/v1/*`) koruması:** OpenAI uyumlu istemciler (Chatbox, SillyTavern vb.) kullanıcı/şifre girişi yapamaz; onlar için ayrı API anahtarı doğrulaması (`Authorization: Bearer` / `x-api-key`) kullanılır. Panelin "API" sekmesinden **birden çok isimlendirilmiş** anahtar oluşturulur, kopyalanır ve silinir (Faz 15).
- **Yönetim paneli ve API açıkları ayrı tutulur:** Panel (kullanıcı/şifre) ile API (token/anahtar) farklı uç noktalar üzerinden korunur.
- CORS yalnızca yönetim paneli farklı origin'den açılacaksa gerekir; `/v1` istemcileri sunucu tarafında olduğundan etkilenmez.

### 6. Dış Ağ Yayınlama (opsiyonel, v2 eklentisi)
- Statik IP + domain varsa: Caddy (otomatik TLS/Let's Encrypt) veya nginx reverse proxy ile `https://domain/v1/*` yayınlanır. Bu bağımsız bir rehber/betik olarak dokümante edilir (`deploy/` klasörü). Yalnızca lokal kullanacaksa bu adım hiç kurulmaz.

### 7. Model Kategorileri (v4 — tam uyumluluk)
Çekirdek hafif kalır, kategoriler modüler eklenir. Model **boyutu** önemli değildir; hangi model olursa olsun yalnızca donanım hızı belirler.

| Kategori | Örnekler | Motor |
|----------|----------|-------|
| Metin / Chat | Llama, Mistral, Qwen, Gemma, Phi | llama.cpp (GGUF) — çekirdek |
| Kodlama | CodeLlama, DeepSeek-Coder, Qwen-Coder, Yi-Coder | llama.cpp (GGUF) — çekirdek |
| Düşünme (Reasoning) | DeepSeek-R1, Qwen3-Reasoning, GLM-Z1 | llama.cpp (GGUF) — çekirdek |
| Resim Anlama (Vision) | LLaVA, Qwen2-VL, Llama-3.2-Vision, Gemma3 | llama.cpp (GGUF) — çekirdek |
| Ses Tanıma (ASR) | Whisper (GGUF) | llama.cpp (whisper) — çekirdek |
| Ses Üretim (TTS) | GGUF TTS modelleri (chat-tts, Kokoro vb.) | GGUF çekirdek / opsiyonel modül |
| Görsel Üretim | Stable Diffusion, FLUX, SDXL | Opsiyonel ComfyUI modülü |
| Video Üretim | SVD, Wan, HunyuanVideo, AnimateDiff | Opsiyonel ComfyUI modülü |
| Müzik/Ses Üretim | Diffusers tabanlı ses modelleri | Opsiyonel ayrı modül |

- Çekirdek modüller varsayılan; üretim modülleri isteğe bağlı (ağır oldukları için ayrı süreç/ayrı kurulum, aynı panel + API).
- **Tam uyumluluk garantisi:** llama.cpp sürekli güncel tutulur (yeni mimariler eklenir); HuggingFace GGUF kütüphanesi tüm açık kaynak modellere erişim sağlar.

### 8. Dahili Web Yönetim Paneli
- Sunucunun `http://<IP-ADRESİ>:9055` adresinde çalışan hafif bir arayüz bulunur.
- HuggingFace üzerindeki açık kaynak GGUF modelleri tek tıkla indirilebilir, silinebilir ve bellek modu değiştirilebilir.
- Yönetici paneli varsayılan olarak sadece `localhost` üzerinden veya API anahtarı korumalı olarak sunulur.

## Konfigürasyon
- Tüm ayarlar `.env` dosyasından okunur (`config.py` üzerinden):
  - `HOST`, `PORT` (varsayılan 9055), `API_KEY`
  - `MEMORY_MODE` (keep / dynamic), `IDLE_TIMEOUT` (dakika)
  - `MODELS_DIR` (varsayılan `./models`)
  - `GGUF` model parametreleri: context boyutu, GPU layer sayısı, thread sayısı

## Proje Dosya Yapısı

```
vprovider/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI sunucusu ve OpenAI API endpoint'leri + /health
│   ├── config.py          # .env tabanlı port, IP, bellek stratejisi, API key
│   ├── model_manager.py   # Worker process + dinamik yükleme/boşaltma mantığı
│   ├── auth.py            # Kullanıcı/şifre (panel) + API anahtar doğrulaması
│   ├── user_store.py      # Kullanıcı kaydı ve şifre hash (ilk kurulum + giriş)
│   └── hf_downloader.py   # HuggingFace model indirme entegrasyonu
├── static/
│   └── index.html         # Web yönetim paneli
├── models/                # İndirilen GGUF modellerinin depolandığı klasör
├── deploy/
│   └── Caddyfile          # Opsiyonel: dış ağ / HTTPS yayınlama örneği
├── install.sh             # Donanım tespitli otomatik kurulum + systemd servisi
├── clear.sh               # Servis ve sistem temizleme betiği
├── .env.example
├── requirements.txt       # Python bağımlılıkları (llama-cpp-python dahil değil; install.sh derler)
└── tests/                 # pytest: auth, endpoint, streaming, model_manager birim testleri
```

> Not: `llama-cpp-python` normal `pip` ile kurulursa CPU derlemesi gelir. GPU desteği için `install.sh` içinde `CMAKE_ARGS`/`GGML_*` değişkenleri ile derleme yapılmalı (requirements.txt'e düz `llama-cpp-python` konmaz).

## Belirlenen Yol Haritası (Roadmap)

| Adım | Açıklama | Durum |
|------|----------|-------|
| 1 | Proje mimarisi ve dizin yapısının tasarlanması | Tamamlandı |
| 2 | Donanım duyarlı `install.sh` ve `clear.sh` betiklerinin hazırlanması | Tamamlandı (systemd adımıyla birlikte) |
| 3 | Dinamik bellek yöneticisinin (`model_manager.py`) kodlanması | Tamamlandı |
| 4 | OpenAI uyumlu FastAPI backend (`main.py`), streaming ve `/health` oluşturulması | Tamamlandı |
| 5 | Kullanıcı oluşturma (ilk kurulum) + panel şifreli giriş (`auth.py`, `user_store.py`) | Tamamlandı |
| 6 | HuggingFace entegreli web yönetim arayüzünün (`index.html`) tasarlanması | Tamamlandı |
| 7 | systemd servis kurulum betiğinin `install.sh` içinde sonlandırılması | Tamamlandı (Faz 8) |
| 7b | Yönetim scriptleri: `start.sh`, `stop.sh`, `restart.sh`, `download.sh` (model indirme), `remove.sh` (tam silme, `models/` korunur veya `--all` ile silinir), `clear.sh` (takma ad) | Tamamlandı (Faz 8) |
| 8 | Test yazımı (pytest: auth, endpoint'ler, streaming, model_manager) | Tamamlandı (69 test yeşil) |
| 9 | Uçtan uca doğrulama + README (Faz 9 kapanış) | Tamamlandı |
| 9b | Ek istek: paralel çoklu indirme + kullanım-bitince-anında GPU boşaltma (`IDLE_TIMEOUT_MINUTES=0`, varsayılan) — tek GPU'da modelden modele geçiş | Tamamlandı (73 test yeşil, canlı GPU doğrulandı) |
| 10 | Opsiyonel: Caddy reverse proxy + domain ile dış ağ yayınlama rehberi | Tamamlandı (`deploy/Caddyfile` + `deploy/caddy-rehber.md`, `caddy validate` doğrulandı) |
| 11 | Opsiyonel Faz: Görsel üretim modülü (ComfyUI entegrasyonu) | Tamamlandı (95 test yeşil; `/v1/images/*` + panel "Görsel Üretim", `deploy/comfyui-rehber.md`, `scripts/comfyui.sh` ve `comfyui-checkpoint.sh`, `deploy/comfyui.service`) |
| 12 | Opsiyonel Faz: Video üretim modülü (ComfyUI) | Tamamlandı (108 test yeşil; `/v1/videos/*`, panel "Video Üretim", AnimateDiff workflow + sunucuda GIF birleştirme, `Pillow` bağımlılığı) |
| 13 | Opsiyonel Faz: Müzik/ses üretim modülü | Tamamlandı (125 test yeşil; `/v1/audio/speech` OpenAI uyumlu edge-tts TTS, panel "Ses Üretim", `app/tts_backend.py`+`tts_api.py`) |
| 14 | Ön yüz tamamlama + canlı doğrulama | Tamamlandı (130 test yeşil; panel "Sohbet" sekmesi + `/panel/chat` uç noktası; kum havuzu sunucusunda arayüzün kullandığı tümuçlarla canlı akış doğrulandı: kurulum→giriş→model listeleme→yükleme→gerçek çıkarım→HF arama→indirme (105 MB SmolLM2-Q4_K_M)→kullanma→silme; `hf_downloader` durum metinleri düzeltildi) |
| 15 | API Anahtarları sekmesi + giriş/kayıt sayfası düzenlemeleri | Tamamlandı (124 test yeşil; "Ayarlar" sekmesi kaldırılıp yerine "API" sayfası: birden çok **isimlendirilmiş** anahtar oluşturma/listeleme/kopyalama/silme (`/panel/apis`); eski tek `settings.api_key` "Varsayılan" anahtarı olarak otomatik taşınır; giriş/kayıt sayfalarına ortalanmış düzen + GitHub/firma altbilgisi; tüm sekmeler navbar'a taşındı ve ortalandı, mobil uyumlu; **Modeller** sayfası yeniden tasarlandı: model yoksa sayfanın ortasında "Burası boş" boş durumu, varsa alt alta liste — model adı, parametre etiketi, dosya boyutu, kategori (metin/görsel/ses/video/müzik) + ikon butonlar: sürekli aktif (keep), istek ile aktif (dynamic), modeli sil (üzerine gelince ipucu); indirilen modeller varsayılan `dynamic` moda atanır; sayfa sağ üstündeki "Ekle" popup'ı açılışta **tüm GGUF modellerini listeler** (yeni `GET /panel/models/browse`: "gguf" adlı repolar geniş tarama + eşzamanlı dosya doğrulama, sonuç önbellekli; 150 model, tüm pipeline'lar), arama + kategori filtresi yerelde anlık çalışır, model seçince altında parametre/dosya seçimi ve "Kur" ile indirme + canlı ilerleme) |