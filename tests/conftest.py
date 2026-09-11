# ─────────────────────────────────────────────────────────────
#  Bölüm:    Test Altyapısı
#  Dosya:    tests/conftest.py
#  Amaç:     Testlerde kullanılacak ortak nesneleri hazırlar
#  Mekanik:  Her test için geçici bir dizinde yeni bir veritabanı
#            (UserStore) oluşturur; auth modülünün singleton
#            erişimi (get_store) bu geçici örneğe yönlendirilir.
#            Böylece testler gerçek proje verisine dokunmaz.
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import pytest

# Proje kökünü import yoluna ekle (app paketine erişim için)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import auth
from app.user_store import UserStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Her test için temiz, geçici bir UserStore döner ve auth'a bağlar."""
    db = UserStore(tmp_path / "test_vprovider.db")
    db.init()
    # auth modülündeki singleton erişimini test veritabanına yönlendir
    monkeypatch.setattr(auth, "get_store", lambda: db)
    return db