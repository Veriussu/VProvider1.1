# VProvider Dış Ağ Yayını — Caddy Rehberi

> **Amaç:** VProvider'ı `http://192.168...` yerine **alan adı + otomatik HTTPS (Let's Encrypt)** üzerinden
> internete güvenle açmak. Yalnızca LAN kullanacaksanız bu rehberi **atlayabilirsiniz** — proje zaten
> IP üzerinden çalışır; Caddy yalnızca isteğe bağlı bir katmandır.

Hazır örnek: `deploy/Caddyfile` (alan adınızı kendinize göre değiştirin).

---

## 0. Önce güvenlik temeli (zorunlu)

Dış ağa açmadan önce:

1. `.env` içinde `HOST=127.0.0.1` yapın (VProvider artık yalnızca yerelde dinler;
   internete yalnızca Caddy açılır).
2. Panelde **ilk kurulumu tamamlayın** (yönetici kullanıcı + şifre) — gerekli kimlik doğrulama
   katmanı budur.
3. `/v1/*` çağrılarının tümünde **API anahtarı** (Bearer) zorunludur; anahtarı paylaşmayın.
   Örnek: `Authorization: Bearer <API_KEY>`.

## 1. Caddy kurulumu

```bash
# Debian/Ubuntu resmi paketi (yalnızca HTTP/HTTPS proxy için yeterlidir):
sudo apt install caddy

# Daha yeni sürüm (rate_limit için 2.7+ gerekir) resmi repo önerilir:
# sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
# curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
# ... (detay: https://caddyserver.com/docs/install)
```

Doğrulama: `caddy version` → v2.x çalışıyor demektir (temel proxy/HTTPS için yeterli).

## 2. Caddyfile kurulumu

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile     # alan adınızı yazın
sudo systemctl reload caddy
```

`deploy/Caddyfile` içinde:

- **`handle /v1/*`** — OpenAI uyumlu uçlar. `flush_interval -1` sayesinde **SSE stream anında** akar
  (önbellek gecikmesi olmaz).
- **`handle /panel*`** — yönetim paneli. Panelde zaten kullanıcı adı + şifre vardır; dış ağda bir
  katman daha isterseniz `basic_auth` bloğunu doldurun:
  ```bash
  caddy hash-password          # çıkan satırı Caddyfile'daki $2a$... yerine koyun
  ```
- **`handle { ... }`** — anasayfa, `/health`, indirilen dosyalar vb. için genel yakalayıcı.

**Hız sınırı (öneri):** `rate_limit` direktifi Caddy çekirdeğine ait **değildir**; bir topluluk
modülüdür (`github.com/mholt/caddy-ratelimit`). Kullanmak isterseniz:
```bash
xcaddy build --with github.com/mholt/caddy-ratelimit
```
derleyip ortaya çıkan Caddy ikilisini kurun, ardından site bloğunun üstüne şunu ekleyin
(Caddyfile sonundaki "İsteğe bağlı ekler" bölümünde de vardır):
```
rate_limit {
    zone v1 {
        key {remote_host}
        events 60      # dakikada en çok istek
        window 1m
    }
}
```
Modül olmadan open-source edisyonlarda `caddy validate` bu satıra hata verir (bilinen durum).

## 3. DNS + port yönlendirme

- Alan adınızın **DNS A kaydını** sunucunun **genel IP** adresine yönlendirin
  (örn. `modelim.ornek-domain.com  A  1.2.3.4`).
- Modem/yönlendiricide **80 ve 443** TCP portlarını bu sunucuya (VProvider'ın çalıştığı makine)
  yönlendirin. Caddy otomatik sertifika için 80'i HTTP-01 challenge için kullanır.
- Güvenlik duvarında da bu iki portun açık olduğundan emin olun:
  ```bash
  sudo ufw allow 80,443/tcp
  ```

## 4. systemd (caddy + vprovider birlikte)

```bash
sudo systemctl enable --now caddy         # Caddy açılışta başlar
sudo systemctl status caddy --no-pager    # durum
```

VProvider servisi zaten `install.sh` ile kurulduysa ikisi bağımsız çalışır. Önerilen düzen:

```
İnternet --> :80/:443 (Caddy) --> 127.0.0.1:9055 (VProvider, sadece localhost)
```

## 5. Doğrulama

```bash
# Sunucu sağlığı
curl https://modelim.ornek-domain.com/health

# API anahtarı ile açık uç
curl https://modelim.ornek-domain.com/v1/models \
     -H "Authorization: Bearer <API_KEY>"

curl https://modelim.ornek-domain.com/v1/chat/completions \
     -H "Authorization: Bearer <API_KEY>" -H "Content-Type: application/json" \
     -d '{"model":"<model_id>","messages":[{"role":"user","content":"merhaba"}]}'

# Panel
# Tarayıcıda https://modelim.ornek-domain.com/panel  -> kullanıcı adı/şifre ile giriş
```

Certificate (sertifika) kontrolü:
```bash
curl -vI https://modelim.ornek-domain.com/health 2>&1 | grep -i subject
```

## 6. Olası sorunlar

| Belirti | Çözüm |
|---|---|
| `Certificate obtained` yok, 80'de redirect döngüsü | 80/443 port yönlendirmesi kapalıdır; `sudo ss -tlnp | grep :80` kontrol edin |
| `too many open files` | `LimitNOFILE=65536` satırı `deploy/vprovider.service` içinde hazırdır |
| Web arayüzden (tarayıcı) `/v1` çağrısı CORS hatası | `deploy/Caddyfile` sonundaki "İsteğe bağlı ekler" CORS satırlarını açın |
| Çok büyük context kesiliyor | `request_timeout 10m` ve `read_timeout 0` (Caddyfile sonundaki ekler) |
| `rate_limit` hatası (unrecognized directive) | Beklenen: modül çekirdekte yok; yukarıdaki `xcaddy` bölümünü uygulayın |