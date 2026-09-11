# VProvider — Proje Yol Haritası ve Adım Adım İlerleme Planı

> Bu dosya, VProvider projesinin tamamının **planını, kullanılan dilleri/altyapıyı ve sıralı ilerleme planını** içerir.
> Her aşamanın sonunda "Doğrulama" adımı vardır; bir aşama doğrulanmadan sıradakine geçilmez.

---

## 1. Proje Özeti

**Hedef:** Ollama/LocalAI gibi ağır araçlara bağımlı olmayan, hafif ve profesyonel bir yerel yapay zeka model sunucusu.

**Temel ilke: Motor hafif, model desteği sınırsız.**

- Yalnızca LAN'da IP ile çalışabilir VEYA statik IP + domain + proxy ile dış ağda.
- OpenAI standartlarında API (`/v1/*`) ile tüm üçüncü taraf istemcilere uyumlu.
- Donanımı otomatik algılar (NVIDIA/AMD/Intel/CPU), ona göre derlenir ve çalışır.
- 0.5B'den 70B+'ye tüm GGUF modelleri (metin, chat, kodlama, düşünme, resim anlama, ses tanıma, TTS).
- Opsiyonel: görsel/video/müzik üretimi (ComfyUI tabanlı ayrı modüller).
- Web yönetim paneli: kullanıcı/şifre girişi + HuggingFace'ten tek tıkla model indirme + bellek modu yönetimi.
- Dockersız, hafif: `venv` + `systemd`.

---

## 2. Teknoloji ve Altyapı Seçimleri

| Bölüm | Seçim | Neden |
|-------|-------|-------|
| **Dil (Backend)** | Python 3.11+ | `llama-cpp-python` ve AI ekosistemi Python tabanlı; geliştirme hızlı ve okunaklı |
| **Web çatısı** | FastAPI + Uvicorn | Async, hızlı, OpenAI/SSE desteği doğal, otomatik dokümantasyon |
| **Model motoru** | `llama.cpp` (`llama-cpp-python`) | Tüm GGUF modeller; CPU/GPU karışık çalıştırma; hafif; motorda sıfır Python bağımlılığı |
| **Model formatı** | GGUF (quantized) | Endüstri standardı; HuggingFace'te binlerce model |
| **Model kaynağı** | HuggingFace Hub (`huggingface-hub`) | Ollama kütüphanesinden kat kat geniş; tüm açık kaynak |
| **Frontend (Panel)** | Tek dosya HTML/CSS/JS (vanilla) | Sıfır derleme adımı, sıfır node bağımlılığı → gerçekten hafif |
| **Veritabanı** | SQLite (stdlib `sqlite3`) | Kullanıcı/session depolama; ayar ve servis gerekmez; hafif |
| **Konfigürasyon** | `.env` + `pydantic-settings` | Ortam değişkenleriyle esnek ayar, kopya kolaylığı |
| **Şifre/session** | `bcrypt` + kendi session token'imiz | Şifre hash güvenli; harici ağır kimlik servisleri yok |
| **Süreç yönetimi** | `systemd` (servis) + `venv` | Dockersız, açılışta otomatik başlar, izlenebilir |
| **Dış ağ (opsiyonel)** | Caddy (otomatik HTTPS) | Let's Encrypt otomatik sertifika, tek dosya yapılandırma |
| **Üretim modülleri (opsiyonel)** | ComfyUI (harici süreç) | Görsel/video üretimi; çekirdekten ayrı, isteğe bağlı |
| **Test** | `pytest` + `httpx` | Hızlı, FastAPI ile doğal uyum |

### Bağımlılık listesi (requirements.txt)
```
fastapi
uvicorn[standard]
pydantic-settings
huggingface-hub
bcrypt
httpx
```
> `llama-cpp-python` bilinçli olarak requirements'e **konmaz**: donanıma göre özel derleme bayraklarıyla `install.sh` içinde kurulur.

---

## 3. Genel Mimari

```
                    ┌─────────────────────────────────────────┐
                    │            VProvider Sunucusu           │
                    │              (FastAPI)                 │
                    │                                         │
  İstemciler ─────► │  /v1/*  (OpenAI API + API anahtarı)     │
  (Chatbox, Jan,   │  /panel (admin API + kullanıcı/şifre)    │
   SillyTavern...)  │  /      (statik yönetim paneli)          │
                    │                                         │
                    └──────────────┬──────────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
     ┌──────────────┐    ┌────────────────┐   ┌──────────────────┐
     │ model_manager│───►│ llama_backend  │   │ hf_downloader     │
     │ (bellek modu)│    │ (GGUF çalıştır)│   │ (model indirme)   │
     └──────────────┘    └────────────────┘   └──────────────────┘
              │                                              │
              ▼                                              ▼
     ┌──────────────┐                                 ┌──────────────┐
     │   models/    │                                 │ HuggingFace  │
     │ (GGUF dosyal)│                                 │  Hub (internet)│
     └──────────────┘                                 └──────────────┘

     Veri kalıcılığı: data/users.db (SQLite) — kullanıcı, session,
     isimlendirilmiş API anahtarları (api_keys), site_info, settings
```

**İki arayüz ayrı korunur:**
- `/v1/*` → API anahtarı (OpenAI istemcileri için)
- `/panel/*`, `/` → kullanıcı adı + şifre (tarayıcı panel için)

### Proje Kimliği ve Site Bilgileri (site_info tablosu)

Uygulama kimliği ve site bilgileri tek tabloda saklanır; ön yüz (giriş/kayıt
altbilgisi) bunları `/panel/status` üzerinden okur.

| Alan | Açıklama |
|------|----------|
| `project_name` | Proje adı (varsayılan: VProvider) |
| `logo` | Logo görseli (BLOB) |
| `favicon` | Favicon görseli (BLOB) |
| `github_url` | GitHub adresi |
| `developer_domain` | Geliştirici firma domaini |
| `docs_url` | Dokümantasyon site adresi |
| `contact_email` | İletişim e-posta adresi |

- Logo ve favicon, veritabanında BLOB olarak saklanır (dosya yolu sorunları çıkmaz).
- Servis uç noktaları: `GET /logo`, `GET /favicon.ico` DB'den anlık döner.
- Faz 15 itibarıyla **panelden düzenleme kaldırıldı**: yalnızca `/panel/status`
  üzerinden okunur (giriş/kayıt altbilgisi GitHub + firma bağlantılarını buradan alır).

---

## 4. Kod Standartları

1. **Her dosyanın başında** o dosyanın *amacını, yapısını ve mekaniğini* anlatan **Türkçe yorum bloğu** bulunur (örnek aşağıda).
2. Tüm yorum ve dokümantasyon **Türkçe** yazılır.
3. Kod okunaklı ve anlaşılır olur: anlamlı değişken isimleri, kısa fonksiyonlar, tip ipuçları.
4. Fonksiyonların başına kısa Türkçe docstring (ne yapar, ne döner).
5. Tek sorumluluk ilkesi: her modül tek iş yapar.

**Örnek dosya başı yorum formatı:**
```python
# ─────────────────────────────────────────────────────────────
#  Bölüm:    ...       (hangi katmana ait)
#  Dosya:    ...
#  Amaç:     Bu dosya ne işe yarar
#  Mekanik:  Nasıl çalışır (akış, bağımlılıklar, alt fonksiyonlar)
#  Kullanım: Kim tarafından, nasıl çağrılır
# ─────────────────────────────────────────────────────────────
```

---

## 5. Dosya Yapısı (Hedef)

```
vprovider/
├── app/
│   ├── __init__.py          # Paket tanımı
│   ├── main.py              # Uygulama oluşturma, router bağlama, başlangıç
│   ├── config.py            # .env okuma ve tüm ayarlar
│   ├── model_manager.py     # Bellek modları + model yükleme/boşaltma yöneticisi
│   ├── llama_backend.py     # llama-cpp-python sarmalayıcı (LLM + streaming)
│   ├── auth.py              # Kullanıcı/şifre + API anahtarı doğrulama
│   ├── user_store.py        # SQLite: kullanıcı, session, API anahtarı
│   ├── hf_downloader.py     # HuggingFace'ten GGUF indirme + arama
│   ├── openai_api.py        # /v1/* router (OpenAI uyumlu endpoint'ler)
│   └── admin_api.py         # Panel API router'ı (login, modeller, ayarlar)
├── static/
│   └── index.html           # Yönetim paneli (tek dosya, vanilla JS)
├── models/                  # İndirilen GGUF modelleri (çalışma klasörü)
├── data/                    # SQLite veritabanı (kullanıcı, session)
├── deploy/
│   ├── vprovider.service      # systemd servis şablonu (User/WorkingDirectory/Restart)
│   └── Caddyfile              # Opsiyonel dış ağ örneği
├── tests/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_openai_api.py
│   ├── test_model_manager.py
│   ├── test_llama_backend.py
│   ├── test_hf_downloader.py
│   ├── test_admin_api.py
│   └── test_e2e.py
├── install.sh               # Donanım tespitli kurulum + systemd
├── clear.sh                 # remove.sh'e yönlendirme (geriye uyumlu)
├── scripts/
│   ├── start.sh             # servisi başlat
│   ├── stop.sh              # servisi durdur
│   ├── restart.sh           # servisi yeniden başlat
│   ├── download.sh          # model indirme (HF GGUF)
│   └── remove.sh            # tam silme
├── runtime/                 # çalışma zamanı logları
├── .env.example
├── requirements.txt
└── README.md                # Kısa kurulum/anlatım
```

---

## 6. Yol Haritası — Fazlar ve Adımlar

Her faz sonunda "Doğrulama" kutusu vardır. Onay verilmeden sonraki faza geçilmez.

### FAZ 0 — Planlama (Tamamlandı ✅)
- [x] Proje tanımı ve plan.md
- [x] Mimari kararlar ve teknoloji seçimi

### FAZ 1 — Proje İskeleti ve Konfigürasyon
**Amaç:** Çalışan boş bir uygulama + tüm ayarların altyapısı.
- [x] `requirements.txt` oluşturulur
- [x] `.env.example` + `.gitignore` + `.env` (yerel kopya)
- [x] `app/config.py` — .env okuma, tüm ayarlar (port, host, bellek modu, idle süre, model klasörü, API anahtarı)
- [x] `app/main.py` — FastAPI uygulaması oluşturulur, health endpoint'i `/health`
- [x] `app/__init__.py`

**Doğrulama:** `uvicorn app.main:app` açılır → `/health` → `200 OK`.

### FAZ 2 — Kimlik Doğrulama (user_store.py + auth.py)
**Amaç:** İlk kurulumda kullanıcı oluşturma, panele şifreli giriş, API anahtarı ve site bilgileri.
- [x] `app/user_store.py` — SQLite tabloları: `users` (id, username, password_hash), `sessions` (token, user_id, expires), `settings` (api_key, setup_done), `site_info` (project_name, github_url, developer_domain, docs_url, contact_email, logo BLOB, favicon BLOB)
- [x] İlk çalıştırmada kullanıcı oluşturma mantığı (setup durumu yoksa panel kurulumu açar)
- [x] `app/auth.py` — şifre doğrulama (bcrypt), session token üretme, `Authorization: Bearer` API anahtarı doğrulama, FastAPI dependency fonksiyonları
- [x] `/logo` ve `/favicon.ico` uç noktaları (DB'den BLOB okur)

**Doğrulama:** `pytest tests/test_auth.py` → kullanıcı oluşturma, giriş, geçersiz şifre, geçersiz API anahtarı testleri yeşil.

### FAZ 3 — Model Yöneticisi (model_manager.py)
**Amaç:** Modellerin tespiti, bellek modları (keep / dynamic), yükleme ve boşaltma.
- [x] `models/` klasörünü tarayan GGUF bulucu
- [x] Model kayıt yapısı: her model için (motor, kilit, son kullanım, mod)
- [x] **keep mode:** model yüklenir, kalıcı kalır
- [x] **dynamic mode:** istek üzerine yüklenir, `idle_timeout` sonunda tamamen boşaltılır → 0 MB VRAM
- [x] Eşzamanlılık: her model için `asyncio.Lock` (aynı modelin istekleri seri işlenir)
- [x] Bellek modu değiştirme API'si (model bazlı + küresel)
- [x] `llama_backend.py` sözleşme (arayüz) katmanı

**Doğrulama:** `pytest tests/test_model_manager.py` → sahte motorlarla 10 test yeşil ✅ (toplam 19 test).

### FAZ 4 — LLM Motoru (llama_backend.py)
**Amaç:** GGUF modelini çalıştıran sarmalayıcı: chat ve completion çağrıları + akışlı (streaming) yanıtlar.
- [x] Model yükleme: context boyutu, GPU layer sayısı, thread sayısı (config'ten)
- [x] `run_chat(messages)` ve `stream_chat(messages)` fonksiyonları (LlamaCppEngine)
- [x] Streaming: parça parça (chunk) üretici (generator)
- [x] `unload`: model nesnesi silinir + GC -> VRAM/RAM anında bırakılır
- [x] `create_engine`: llama-cpp-python yoksa anlaşılır hata

**Doğrulama:** Gerçek Qwen2-0.5B GGUF modeliyle (models/smoke) üretilen yanıt + stream ✅
- `pytest tests/test_llama_backend.py` → 5 test yeşil ✅ (toplam 24 test)
- **GPU doğrulaması (RTX 4060):** llama-cpp-python CUDA ile derlendi
  (`CMAKE_ARGS="-DGGML_CUDA=on"`); `llama_supports_gpu_offload()` = True.
  VRAM ölçümü: boşta 3690 MiB → model yüklü 4419 MiB → boşaltınca 3800 MiB
  → model gerçekten ekran kartında çalışıyor. `install.sh` bu derlemeyi
  donanım tespitiyle otomatik yapacak (Faz 8).

### FAZ 5 — OpenAI Uyumlu API (openai_api.py + main.py)
**Amaç:** Standart `/v1/*` endpoint'leri.
- [x] `GET /v1/models` — yüklü mevcut modellerin listesi
- [x] `POST /v1/chat/completions` — chat + streaming (SSE)
- [x] `POST /v1/completions` — completion + streaming (SSE)
- [x] Hata biçimleri OpenAI standartlarında (404 model yok, 401 anahtar yok)
- [x] main.py'e router bağlama

**Doğrulama:** Chatbox/curl ile `/v1/chat/completions` normal + stream test.

### FAZ 6 — Model İndirici (hf_downloader.py)
**Amaç:** HuggingFace'ten GGUF modelleri arama ve indirme.
- [x] HuggingFace API arama (GGUF içeren repo'ları filtreleme, kategori/pipeline filtresi)
- [x] Parça parça (resumable) indirme + indirme ilerlemesi (Range destekli, `.part` atomik bitiş, ilerleme kaydı)
- [x] İndirme tamamlanınca model listesine ekleme (disk taramasıyla otomatik)
- [x] Model silme (önce boşaltma, ardından dosya temizliği)

**Doğrulama:** Panel üzerinden gerçek bir model aranır, indirilir, listelenir.

### FAZ 7 — Yönetim Paneli (admin_api.py + static/index.html)
**Amaç:** Kurulum (kullanıcı oluşturma), giriş, model yönetimi ve bellek ayarları.
- [x] `admin_api.py`: setup, login, logout, model listesi, model indir/sil, bellek modu değiştir
- [x] `admin_api.py`: site bilgileri yönetimi (`GET/POST /api/settings`) + logo/favicon yükleme (BLOB)
- [x] `index.html`: giriş sayfası, model listesi/kartları, indirme butonu, bellek modu anahtarı, durum paneli
- [x] Oturum çerezine göre panel koruması
- [x] Kurulum tamamlanmadıysa otomatik kurulum ekranına yönlendirme

**Doğrulama:** Tarayıcıdan katma değer: ilk kurulum → giriş → model indir → bellek modu değiştir. `pytest tests/test_admin_api.py` → 16 test, tümü yeşil ✅ (toplam 62 test).

### FAZ 8 — Kurulum, Yönetim ve Temizlik Scriptleri (systemd)
**Amaç:** Tek komutla kurulum, otomatik başlatma; çalıştırma/durdurma/yeniden başlatma/indirme/tam silme için ayrı scriptler (`scripts/`).
- [x] `install.sh`: donanım tespiti → derleme bayrakları → venv + pip → llama-cpp-python derleme → geri kalan bağımlılıklar → `.env` üretimi → systemd servisi kurma/başlatma
- [x] `install.sh`: çalıştırıldığında terminalde büyük **VProvider** başlığı (`---` ayraçlarıyla, aşağıdaki bölümdeki görsel)
- [x] `start.sh`: servisi başlatır (servis yoksa uyarır)
- [x] `stop.sh`: servisi durdurur
- [x] `restart.sh`: servisi yeniden başlatır
- [x] `download.sh`: model indirme yardımcı betiği (model id + HF adresi argümanı, `models/` altına GGUF iner)
- [x] `remove.sh`: **tamamen silme** — servisi durdur/sil, venv'i kaldır, config/log verilerini temizle, sistemden komut bağlarını kaldır; `models/` korunur ya da `--all` ile silinir
- [x] `vprovider.service` şablonu (Restart=on-failure, WorkingDirectory, User)
- [x] `clear.sh`: `remove.sh`'e yönlendirme (geriye uyumlu takma ad)

**Doğrulama:** Temiz kurulum başlar, servis otomatik ayakta; start/stop/restart döngüsü sorunsuz; download.sh model indirir; remove.sh her şeyi geri alır. — Yerel doğrulama: `bash install.sh --no-systemd` → banner + venv + tüm adımlar tamam; start/stop/restart (uvicorn fallback) + `download.sh ggml-org/models-moved tinyllamas/stories260K.gguf` (1.1 MB, 1.1 sn) başarılı; systemd/symlink adımları SuperUser yetkisi olan sistemde devreye girer.

### FAZ 9 — Testler ve Kapanış
**Amaç:** Her şeyin garantisi.
- [x] Tüm birim testleri (auth, openai_api, model_manager, hf_downloader)
- [x] Uçtan uca test: gerçek model indir → yükle → chat → stream → panel
- [x] CPU ve (varsa) GPU modda deneme
- [x] README.md son hali

**Doğrulama:** `pytest` tamamen yeşil + gerçek model senaryosu başarılı.
- `pytest tests/ -q` → **69 test, tümü yeşil** ✅ (birim + API + motor + panel + uçtan uca)
- `tests/test_e2e.py` → gerçek GGUF modeli + gerçek llama.cpp motoru, **GPU mod** (`gpu_layers=-1`) ve **CPU mod** (`VPROVIDER_E2E_GPU_LAYERS=0`) ayrı ayrı çalıştırıldı, ikisinde de 6/6 ✅
  - Akış gerçekleşti: panel listesi → /v1/models → /v1/chat/completions (akışsız + SSE tek id) → /v1/completions → panel yükle/boşalt/mode
- `README.md` son hali yazıldı: kurulum, scriptler, API referansı, .env tablosu, test rehberi

### FAZ 10 — Opsiyonel Dış Ağ
- [x] Caddyfile örneği + domain/sertifika rehberi
- [x] API anahtarlı açık erişim, HTTPS, (öneri) rate limiting

**Doğrulama:** `deploy/Caddyfile` örneği `caddy:2` (v2.11.4) imajıyla
`caddy validate` + `caddy fmt` → **valid**. `rate_limit` direktifinin Caddy
çekirdeğinde olmadığı öğrenildi (topluluk modülü `mholt/caddy-ratelimit`,
`xcaddy` ile derlenir) — bu yüzden Caddyfile'da yorum olarak bırakıldı.
LAN-only (IP) ve domain (Let's Encrypt) olmak üzere iki örnek verildi;
rehber `deploy/caddy-rehber.md` (DNS, port yönlendirme, systemd, CORS,
sorun giderme tablosu).

### FAZ 11 — Opsiyonel Üretim Modülleri (ComfyUI)
- [x] Görsel üretim modülü (ComfyUI entegrasyonu) — `/v1/images/generations` +
      `/v1/images/comfy/checkpoints` (OpenAI uyumlu), panelde "Görsel Üretim"
      sekmesi, `app/comfy_client.py` + `app/comfy_api.py`, scriptler
      (`comfyui.sh`, `comfyui-checkpoint.sh`), systemd şablonu ve
      `deploy/comfyui-rehber.md`. (95 test yeşil)
- [x] Video üretim modülü — `/v1/videos/generations` + panelde "Video Üretim";
      AnimateDiff (ADE) workflow'u, kareler sunucuda Pillow ile GIF'e
      birleştirilir, `Pillow>=10` bağımlılığı eklendi. (108 test yeşil, Faz 12)
- [x] Müzik/ses üretim modülü — `/v1/audio/speech` (OpenAI uyumlu) edge-tts ile
      metin-sesleme, Türkçe sesler, panelde "Ses Üretim" sekmesi. `edge-tts`
      bağımlılığı eklendi. (125 test yeşil, Faz 13)
- [x] Ön yüz tamamlama + canlı doğrulama — panelde "Sohbet" sekmesi ve
      `/panel/chat` uç noktası (API anahtarı gerekmez, oturum korumalı);
      modeli arayüzden yükle/sohbet et, HF arama→indirme→kullanma→silme akışı
      kum havuzu sunucusunda gerçek modelle uçtan uca doğrulandı. (130 test
      yeşil, Faz 14)
- [x] API Anahtarları sekmesi — "Ayarlar" sekmesi kaldırılıp yerine "API"
      sayfası geldi: birden çok isimlendirilmiş anahtar oluşturma, listeleme,
      kopyalama ve silme (`GET/POST/DELETE /panel/apis`). Eski tek
      `settings.api_key`, ilk başlatmada otomatik olarak "Varsayılan"
      isimlendirilmiş anahtara taşınır; `require_api_key` miras anahtarı da
      kabul eder. Giriş/kayıt sayfaları ortalandı, GitHub + üretici firma
      altbilgisi eklendi. Modeller sayfası yeniden tasarlandı: model yoksa
      ortada "Burası boş", varsa alt alta liste (ad, parametre, boyut, kategori
      metin/görsel/ses/video/müzik) + ikon butonlar — sürekli aktif (keep),
      istek ile aktif (dynamic), modeli sil (hover ipucu). İndirilen modeller
      varsayılan `dynamic` moda atanır. Sağ üstteki "Ekle" popup'ı açılışta
      HuggingFace'in tüm GGUF kataloğunu listeler (`/panel/models/browse`:
      "gguf" repoları geniş tarama + thread pool ile eşzamanlı doğrulama,
      ~150 model, önbellekli); metin araması ve kategori/pipeline filtresi
      yerelde anlık çalışır; model seçilince parametre/dosya seçimi ve "Kur"
      ile indirme + ilerleme çubuğu sunar. Modeller listesi canlı yenilenir.
      (125 test yeşil, Faz 15)

---

## 7. Donanım Tespit Matrisi (install.sh)

| Tespit | Bayrak / Eylem |
|--------|----------------|
| `nvidia-smi` çalışıyor | `CMAKE_ARGS="-DGGML_CUDA=on"` → CUDA derleme |
| `rocm-smi` çalışıyor | `-DGGML_HIPBLAS=on` → ROCm derleme |
| CUDA/ROCm SDK yok ama GPU var | `-DGGML_VULKAN=on` → Vulkan (tüm markalar) |
| Hiçbiri yok / sürücü yok | CPU modu (varsayılan derleme) |
| `lspci` Intel | SYCL derleme denemesi, başarısızsa Vulkan |

> Öncelik: CUDA → ROCm → SYCL → Vulkan → CPU. Bir tanesi bile çalışmazsa sıradaki devreye girer; **her zaman** en az CPU çalışır.

---

## 8. Bellek Modları Mekaniği

**Her Daim Hazır (keep):**
- İlk kullanımda model yüklenir, `Llama` nesnesi bellekte kalır.
- İkinci istekte hazır → gecikme sıfıra yakın.
- Panelden "bellekten kaldır" denirse boşaltılır.

**Kullanırken Yükle (dynamic):**
1. İstek gelir → model yüklü değilse yüklenir.
2. Yanıt üretilir.
3. Arka planda bir zamanlayıcı başlar (`idle_timeout`, örn. 5 dk).
4. Süre içinde yeni istek gelirse sayaç sıfırlanır.
5. Süre dolarsa model boşaltılır (`del` + `llama_backend_free()` + `gc.collect()`) → **boşta 0 MB VRAM**.

---

## 9. API Tasarımı

**Herkese açık:**
- `GET /health` → `{status: "ok", version: ...}`
- `GET /logo` → DB'deki logo görseli (BLOB)
- `GET /favicon.ico` → DB'deki favicon görseli (BLOB)

**OpenAI uyumlu (API anahtarı zorunlu, `Authorization: Bearer <anahtar>`):**
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

**Panel API (oturum çerezi zorunlu):**
- `POST /api/setup` (ilk kurulum: kullanıcı oluştur)
- `POST /api/login` / `POST /api/logout`
- `GET  /api/models` (yerel modeller + durum)
- `POST /api/models/download` (HuggingFace'ten indir)
- `DELETE /api/models/{model}` (model sil)
- `POST /api/models/{model}/load` (belleğe al)
- `POST /api/models/{model}/unload` (bellekten kaldır)
- `GET  /api/config` / `POST /api/config` (bellek modu, idle süresi vb.)
- `GET  /api/settings` / `POST /api/settings` (site bilgileri)
- `POST /api/settings/logo` / `POST /api/settings/favicon` (görsel yükleme - BLOB)

---

## 10. install.sh — VProvider Başlık Bandı (Banner)

`install.sh` çalıştırıldığında terminale büyük VProvider başlığı ve `---` ayraçları basılır:

```
----------------------------------------------------------------------------------------------------------------
   ██   ██ ██████   ██████  ██   ██  ██████  ██████  ██████   ██████   ██████
   ██   ██ ██   ██ ██   ██ ██   ██    ██   ██   ██   ██  ██ ██      ██   ██
   ██   ██ ██████  ██████  ██   ██    ██   ██   ██   ██████  ██████  ██████
    ██ ██  ██      ██  ██   ██ ██     ██   ██   ██   ██  ██ ██      ██  ██
     ███   ██████  ██   ██   ███    ██████  ██████  ██   ██ ██████  ██   ██
----------------------------------------------------------------------------------------------------------------
  Hafif Yerel Yapay Zeka Model Sunucusu - Kurulum Başlıyor...
----------------------------------------------------------------------------------------------------------------
```

> Kontrol: Bu görsel ile install.sh'teki banner birebir aynı olmalı (Komut: `bash install.sh`).

---

## 11. Riskler ve Çözümleri

| Risk | Çözüm |
|------|-------|
| Çok yeni mimarili model llm.cpp'de yok | llama.cpp'i düzenli güncelle; GGUF çevirisi HF'te yayınlanınca destek gelir |
| GPU derlemesi SDK eksikliğiyle patlar | Vulkan fallback + CPU son çare; kurulum asla başarısız kalmaz |
| Dinamik modda ilk istek gecikir | Idle süresi ayarlanabilir; "her daim hazır" moduyla istenirse hiç gecikmez |
| İnternet'e açıkken kötüye kullanım | Zorunlu API anahtarı + kullanıcı/şifre panel + HTTPS (Caddy) |
| Şifre sızıntısı | Yalnızca bcrypt hash saklanır; anahtar panelde üretilir |
| Aynı anda çok istek | Model başına asyncio.Lock; seri işleme, kilitlenme yok |
| İndirme yarım kalır | Parça parça/özetlenebilir indirme (huggingface-hub) |

---

## 12. Sıradaki Adım

**Faz 1 — Proje İskeleti ve Konfigürasyon** ile başlanır.
Sonrasında her fazın "Doğrulama" adımı tamamlanıp onay alındıkça ilerlenir.