# ─────────────────────────────────────────────────────────────
#  Bölüm:    Kimlik Doğrulama Katmanı
#  Dosya:    app/auth.py
#  Amaç:     Şifre hash'leme, oturum yönetimi ve API anahtarı
#            doğrulamasını sağlar; FastAPI dependency'leri sunar.
#  Mekanik:  - Şifreler bcrypt ile hash'lenir (düz metin asla saklanmaz).
#            - Panele giriş: kullanıcı adı/shifre doğrulanır, rastgele
#              oturum token'i çereze yazılır.
#            - Docker/OpenAI istemcileri kullanıcı/shifre girişi yapamaz;
#              onlar API anahtarı (Bearer token) kullanır.
#            - get_current_session ve require_api_key fonksiyonları
#              FastAPI dependency olarak route'lara eklenir.
#  Kullanım: from app.auth import hash_password, require_api_key
# ─────────────────────────────────────────────────────────────

import hmac
import secrets

import bcrypt
from fastapi import Header, HTTPException, Request, status

from app.user_store import get_store

# Panel oturum çerezinin adı
SESSION_COOKIE = "vprovider_session"


# ------------------------------------------------------------------
# Şifre işlemleri (bcrypt)
# ------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Düz şifreyi bcrypt ile hash'ler ve metin olarak döner."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    """Verilen şifrenin, saklanan hash ile eşleşip eşleşmediğini döner."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Geçersiz hash biçimi -> güvenlik için eşleşmez kabul edilir
        return False


# ------------------------------------------------------------------
# Token / anahtar yardımcıları
# ------------------------------------------------------------------

def generate_token() -> str:
    """Kriptografik olarak rastgele bir token üretir (oturum veya API anahtarı)."""
    return secrets.token_hex(32)


def verify_api_key(candidate: str, stored: str) -> bool:
    """API anahtarını sabit zamanlı (timing attack korumalı) karşılaştırır."""
    if not stored or not candidate:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), stored.encode("utf-8"))


def authenticate_user(username: str, password: str) -> dict | None:
    """Kullanıcı adı ve şifreyi doğrular; geçerliyse kullanıcı kaydını döner."""
    user = get_store().get_user_by_username(username)
    if user is None:
        return None
    if not check_password(password, user["password_hash"]):
        return None
    return user


def create_session_for_user(user_id: int) -> str:
    """Kullanıcı için yeni bir oturum token'i üretir ve depolar."""
    token = generate_token()
    get_store().create_session(token, user_id)
    return token


# ------------------------------------------------------------------
# FastAPI bağımlılıkları (dependency)
# ------------------------------------------------------------------

def get_current_session(request: Request) -> dict:
    """Panel istekleri için oturum doğrulaması yapar (çerez kontrolü).

    Çerez yok veya oturum süresi dolmuşsa 401 hatası fırlatır.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Giriş yapılmamış. Önce panele giriş yapın.",
        )
    session = get_store().get_session(token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Oturum geçersiz veya süresi dolmuş. Yeniden giriş yapın.",
        )
    return session


def require_api_key(authorization: str | None = Header(default=None)) -> str:
    """OpenAI uyumlu /v1/* uç noktaları için API anahtarı doğrulaması (Bearer).

    Anahtar yoksa veya eşleşmiyorsa 401 hatası fırlatır; geçerliyse anahtarı döner.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz yetkilendirme. 'Authorization: Bearer <anahtar>' gönderin.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    candidate = authorization.split(" ", 1)[1].strip()
    store = get_store()
    if any(verify_api_key(candidate, k) for k in store.get_all_api_keys()):
        return candidate
    # Miras uyumluluğu: eski tek settings.api_key hâlâ geçerli olsun
    if store.get_api_key() and verify_api_key(candidate, store.get_api_key()):
        return candidate
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="API anahtarı geçersiz.",
        headers={"WWW-Authenticate": "Bearer"},
    )