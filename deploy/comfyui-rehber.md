# ComfyUI Görsel Üretim Rehberi (Faz 11)

VProvider görsel üretimi **harici bir ComfyUI** süreciyle yapar. Bu rehber ComfyUI'yi
kurup internetten görsel üreten (txt2img) ilk checkpoint'inizi gösterir.

> Mimari: VProvider (Python/FastAPI) → HTTP → `ComfyUI` (`127.0.0.1:8188`).
> LLM `IDLE_TIMEOUT_MINUTES=0` sayesinde kullanım biter bitmez GPU'yu boşalttığı için
> görsel motoruyla VRAM yarışmaz.

---

## 1. ComfyUI Kurulumu

ComfyUI bağımlılıklarını kendi `~/.venv`'inde tutar; sistem Python'una dokunmaz.

```bash
# git yoksa: sudo apt install -y git
git clone https://github.com/comfyanonymous/ComfyUI.git ${HOME}/ComfyUI
cd ${HOME}/ComfyUI

# Python sanal ortamı (kendi venv'i)
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# Üretim (SD/SDXL) için yeterli olan bağımlılıklar:
.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
```

> CPU ile denemek için `--index-url .../cpu` kullanabilirsiniz (çok yavaş olur).

## 2. İlk Checkpoint (SD 1.5)

```bash
# VProvider repo'sundaki indiriciyi kullan (sürdürülebilir indirme, doğru klasör):
bash scripts/comfyui-checkpoint.sh stabilityai/sd-1.5 v1-5-pruned-emaonly.safetensors
```

Dilerseniz SDXL (daha kaliteli, daha yavaş):

```bash
bash scripts/comfyui-checkpoint.sh stabilityai/sdxl-1.5 single_file_stable_diffusion_xl_base_1.0.safetensors
```

## 3. Motoru Başlatma

```bash
bash scripts/comfyui.sh start     # başlat (log: runtime/comfyui.log)
bash scripts/comfyui.sh status    # durum
bash scripts/comfyui.sh stop      # durdur
```

Doğrulama:

```bash
curl -s http://127.0.0.1:8188/system_stats | jq .devices[0].name
```

## 4. VProvider'ı Köprüye Bağlama

`.env` içinde (dosya yoksa `install.sh` oluşturur):

```env
COMFYUI_ENABLED=true
COMFYUI_HOST=127.0.0.1
COMFYUI_PORT=8188
COMFYUI_DIR=/home/<kullanici>/ComfyUI
COMFYUI_DEFAULT_CHECKPOINT=v1-5-pruned-emaonly.safetensors
COMFYUI_DEFAULT_NEGATIVE=blur, ugly, low quality, watermark
```

Sonra sunucuyu yeniden başlatın:

```bash
bash scripts/restart.sh
```

Panelin **Görsel Üretim** sekmesi artık motor durumunu, GPU bilgisini ve checkpoint
listesini gösterir; üretilen görseller panel içinde (VEYA API ile) sunulur.

## 5. API ile Üretim

```bash
API_KEY="panelden-kopyalanan-anahtar"

curl http://localhost:9055/v1/images/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "v1-5-pruned-emaonly.safetensors",
       "prompt": "kırmızı ayakkabılı sarı kedi, yağlı boya tarzı",
       "size": "512x512", "steps": 20, "n": 2,
       "seed": -1, "response_format": "url"}'

# json içindeki data[0].url → görsel dosyası
# veya response_format: "b64_json" ile data[0].b64_json (base64)
```

İstek alanları: `prompt` (zorunlu), `n`, `size`, `model` (checkpoint adı),
`response_format` (`url`/`b64_json`), `negative_prompt`, `steps`, `seed`.
Step/cfg ayrıntıları `app/comfy_client.py` içinde `build_txt2img_workflow`
fonksiyonundadır.

## 6. systemd ile Otomatik Başlatma (opsiyonel)

`deploy/comfyui.service` şablonunu düzenleyin:

```bash
sed -e "s|{{USER}}|$(whoami)|g" \
    -e "s|{{COMFY_DIR}}|/home/$(whoami)/ComfyUI|g" \
    deploy/comfyui.service | sudo tee /etc/systemd/system/comfyui.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now comfyui
bash scripts/comfyui.sh status   # artık systemd üzerinden raporlar
```

> Not: `comfyui.sh` systemd servisi kuruluysa `systemctl` kullanır; aksi halde
> nohup ile çalışır.

## 7. Önerilen Model Ayarları (8 GB VRAM, RTX 4060)

| Model | Boyut | Süre (512x512, 20 adım) | Not |
|---|---|---|---|
| SD 1.5 (`v1-5-pruned`) | ~4 GB | ~15-30 sn | Başlangıç için ideal |
| SDXL Base | ~6,5 GB | ~40-90 sn | Kalite; `--highvram` yerine auto önerilir |
| FLUX.1 (schnell/dev) | >10 GB | — | Özel workflow + daha fazla VRAM gerekir |
| SVD / Wan (video) | ~2-14 GB | — | Faz 12 kapsamı; VRAM sınırlı |

Yetersiz VRAM'de "CUDA out of memory" hatası görürseniz: adım sayısını azaltın,
512x512 kullanın, `n`'yi 1 yapın ve LLM'in `IDLE_TIMEOUT_MINUTES=0` (varsayılan)
olduğundan emin olun.

## 8. Sorun Giderme

| Belirti | Çözüm |
|---|---|
| `/panel/comfy/status` → "Motor yanıt vermiyor" | `scripts/comfyui.sh status` ve `runtime/comfyui.log`; port çakışmasına bakın |
| `curl 127.0.0.1:8188/system_stats` boş | ComfyUI başlamadı; torch kurulumunu kontrol edin |
| "Port already in use" | `ss -ltnp | grep 8188`; eski süreci durdurun |
| CUDA out of memory | N=1, 512x512, adım 20'den başlayın; LLM boşta olduğundan emin olun |
| Checkpoint listede yok | `/models/checkpoints/{ad}.safetensors` yolunu doğrulayın, panelde yenileyin |
| FLUX grid dışı çıktı | FLUX özel workflow ister (varsayılan dizayn SD/SDXL içindir) |

---

## 9. Video Üretimi (Faz 12, AnimateDiff)

Video, **AnimateDiff Evolved (ADE)** node paketiyle üretilen karelerin sunucuda
GIF'e birleştirilmesiyle çalışır (`/v1/videos/generations`).

### 9.1 Gereksinimler

1. **ComfyUI Manager** kurun (ComfyUI kurulumunuza):
   ```bash
   cd ${HOME}/ComfyUI/custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   cd ../ && .venv/bin/python -m pip install -r custom_nodes/ComfyUI-Manager/requirements.txt
   ```
2. Manager üzerinden **"AnimateDiff Evolved"** paketini kurun (ya da elle
   `custom_nodes/AnimateDiff-Evolved` klonlayın + gereksinimleri kurun).
3. **Hareket modülünü** indirin ve doğru klasöre koyun:
   ```
   ComfyUI/models/animate_diff/mm_sd_v15_v2.ckpt   (~1,7 GB)
   ```
   Kaynak: HuggingFace'te `guoyww/animatediff` repo'sunda `mm_sd_v15_v2.ckpt`,
   `comfyui/models/` yolundadır.

Checkpoint indiricimiz yalnızca `models/checkpoints` içine yazar; hareket modülü
için `ComfyUI/models/animate_diff/` klasörüne manuel taşıma gerekir.

### 9.2 Doğrulama (ComfyUI arayüzünde)

Idle (ara) arayüzde aşağıdaki düğüm bağlantısı çalışmalıdır:
```
CheckpointLoaderSimple ─ model → ADE_AnimateDiffLoaderWithContext ─ model → KSampler
       ├── clip → CLIPTextEncode (pozitif)  ─ positive → KSampler
       └── clip → CLIPTextEncode (negatif)  ─ negative → KSampler
EmptyLatentImage (batch = kare sayısı) ─ latent_image → KSampler
```

VProvider'ın `build_video_workflow`'u aynı grafiği JSON olarak kurar
(`app/comfy_client.py`). Çalıştıktan sonra `SaveImage` her kare için ayrı PNG
üretir; VProvider bunları `Pillow` ile GIF yapar.

### 9.3 Kullanım

```bash
curl http://localhost:9055/v1/videos/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "v1-5-pruned-emaonly.safetensors",
       "prompt": "uçan kuş sürüsü", "size": "512x512", "frames": 16}'
```

### 9.4 Video Ayarları (8 GB VRAM)

| Biçim | Öneri | Süre (yaklaşık) |
|---|---|---|
| 512x512, 16 kare | Başlangıç; ~20-40 sn | 1-3 dk |
| 512x512, 24 kare | Dengeli | 2-5 dk |
| 512x768, 24+ kare | Ağır; VRAM'a göre ayarlayın | 4-8 dk |

> CUDA out of memory: kare sayısını düşürün (8/16), 512x512'de kalın. LLM'in
> `IDLE_TIMEOUT_MINUTES=0` (varsayılan) olduğundan emin olun ki GPU boşalsın.

---