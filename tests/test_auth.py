# ─────────────────────────────────────────────────────────────
#  Bölüm:    Kimlik Doğrulama Testleri
#  Dosya:    tests/test_auth.py
#  Amaç:     Kullanıcı oluşturma, şifre doğrulama, oturum yönetimi,
#            API anahtarı ve site bilgileri mekanizmasını test eder.
#  Mekanik:  Testler "store" fixture'ı ile geçici veritabanı kullanır;
#            gerçek proje verisine dokunmaz.
# ─────────────────────────────────────────────────────────────

from fastapi import HTTPException
import pytest

from app import auth
from app.user_store import KEY_API_KEY, KEY_SETUP_DONE


# ------------------------------------------------------------------
# Kullanıcı ve şifre
# ------------------------------------------------------------------

def test_create_user_and_verify_password(store):
    """Kullanıcı oluşturulur ve doğru şifreyle oturum açılabilir."""
    hashed = auth.hash_password("gizli-sifre-123")
    assert store.create_user("admin", hashed) is True

    # Oluşturulan kullanıcının hash'i saklanır
    user = store.get_user_by_username("admin")
    assert user is not None
    assert user["password_hash"] != "gizli-sifre-123"  # düz metin saklanmaz

    # Doğru şifre kabul edilir
    assert auth.check_password("gizli-sifre-123", user["password_hash"]) is True
    # Yanlış şifre reddedilir
    assert auth.check_password("yanlis-sifre", user["password_hash"]) is False


def test_duplicate_username_rejected(store):
    """Aynı kullanıcı adı iki kez oluşturulamaz."""
    hashed = auth.hash_password("sifre")
    assert store.create_user("admin", hashed) is True
    assert store.create_user("admin", hashed) is False


def test_authenticate_user_bad_credentials(store):
    """Olmayan kullanıcı veya hatalı şifreyle giriş başarısız olur."""
    hashed = auth.hash_password("dogru-sifre")
    store.create_user("admin", hashed)

    assert auth.authenticate_user("admin", "dogru-sifre") is not None
    assert auth.authenticate_user("admin", "yanlis-sifre") is None
    assert auth.authenticate_user("yok-olan", "dogru-sifre") is None


# ------------------------------------------------------------------
# Oturum yönetimi
# ------------------------------------------------------------------

def test_session_roundtrip(store):
    """Oluşturulan oturum okunur ve silinebilir."""
    hashed = auth.hash_password("sifre")
    store.create_user("admin", hashed)
    user = store.get_user_by_username("admin")

    token = auth.create_session_for_user(user["id"])
    session = store.get_session(token)
    assert session is not None
    assert session["user_id"] == user["id"]

    # Çıkış: oturum silinir
    store.delete_session(token)
    assert store.get_session(token) is None


def test_expired_session_invalid(store):
    """Süresi dolmuş oturum geçersiz sayılır ve silinir."""
    hashed = auth.hash_password("sifre")
    store.create_user("admin", hashed)
    user = store.get_user_by_username("admin")

    # Geçmişte biten bir oturum oluştur
    store.create_session("eski-token", user["id"], ttl_days=-1)
    assert store.get_session("eski-token") is None  # süresi dolmuş kabul edilir


# ------------------------------------------------------------------
# API anahtarı
# ------------------------------------------------------------------

def test_api_key_set_and_verify(store):
    """API anahtarı kaydedilir ve sabit zamanlı karşılaştırma ile doğrulanır."""
    store.set_api_key("s3cret-anahtar")
    assert store.get_api_key() == "s3cret-anahtar"
    assert auth.verify_api_key("s3cret-anahtar", store.get_api_key()) is True
    assert auth.verify_api_key("yanlis-anahtar", store.get_api_key()) is False


def test_api_key_requires_set_value(store):
    """Boş/ayarlanmamış anahtarla doğrulama her zaman başarısız olur."""
    assert store.get_api_key() == ""
    assert auth.verify_api_key("herhangi-bir-sey", store.get_api_key()) is False


def test_init_preserves_existing_api_key(store):
    """Başlatma (init) ayarları sıfırlamamalı, mevcut API anahtarını korumalı."""
    store.set_api_key("kalici-anahtar")
    store.init()  # uygulama yeniden başlarken init çağrılır
    assert store.get_api_key() == "kalici-anahtar"


def test_init_does_not_import_legacy_key_as_named(store):
    """Kullanıcı anahtarı kendisi oluşturur; miras ayar isimlendirilmiş anahtara aktarılmaz."""
    store.set_api_key("miras-anahtar")
    store.init()
    assert store.list_api_keys() == []  # otomatik 'Varsayılan' anahtar artık üretilmez
    with pytest.raises(HTTPException) as exc:
        auth.require_api_key(authorization="Bearer miras-anahtar")
    assert exc.value.status_code == 401


def test_multiple_named_keys_verified(store):
    """Birden çok isimlendirilmiş anahtar require_api_key ile doğrulanır."""
    store.create_api_key("bir", "anahtar-1")
    store.create_api_key("iki", "anahtar-2")
    assert auth.require_api_key(authorization="Bearer anahtar-2") == "anahtar-2"
    assert auth.require_api_key(authorization="Bearer anahtar-1") == "anahtar-1"
    with pytest.raises(HTTPException) as exc:
        auth.require_api_key(authorization="Bearer yanlis")
    assert exc.value.status_code == 401


# ------------------------------------------------------------------
# Ayarlar ve setup durumu
# ------------------------------------------------------------------

def test_setup_done_flow(store):
    """İlk kurulum durumu beklenen şekilde değişir."""
    assert store.is_setup_done() is False
    store.mark_setup_done()
    assert store.is_setup_done() is True


# ------------------------------------------------------------------
# Site bilgileri (proje kimliği + logo/favicon)
# ------------------------------------------------------------------

def test_site_info_defaults(store):
    """Site kimliği kod içine gömülü sabit değerlerden gelir."""
    info = store.get_site_info()
    assert info["project_name"] == "VProvider"
    assert info["github_url"] == "https://github.com/Veriussu/"
    assert info["developer_domain"] == "https://veriussu.com"
    assert info["contact_email"] == "vprovider@veriussu.com"


def test_site_info_is_immutable(store):
    """Sistem ayarları sabittir: store üzerinden değiştirilemez."""
    info = store.get_site_info()
    assert not hasattr(store, "save_site_info")
    assert not hasattr(store, "set_image")
    # Her çağrıda aynı sabit değerler döner
    assert store.get_site_info() == info
    assert info["project_name"] == "VProvider"
    assert info["contact_email"] == "vprovider@veriussu.com"