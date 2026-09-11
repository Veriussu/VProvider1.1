# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 2 - Veri Katmanı
#  Dosya:    app/user_store.py
#  Amaç:     Kullanıcı, oturum, uygulama ayarları ve site bilgilerini
#            SQLite veritabanında tutar.
#  Mekanik:  - UserStore sınıfı tek bir .db dosyasını iş parçacığı
#              güvenli (kilitli) şekilde yönetir.
#            - Tablolar: users (kullanıcılar), sessions (oturumlar),
#              settings (anahtar-değer ayarlar: api_key, setup_done),
#              site_info (proje adı, github/domain/mail, logo/favicon BLOB).
#            - Logo ve favicon, dosya yolu yerine BLOB olarak saklanır;
#              böylece klasör taşıma sorunları oluşmaz.
#  Kullanım: get_store() tek örneği (singleton) verir; tüm modüller
#            bu fonksiyon üzerinden erişir.
# ─────────────────────────────────────────────────────────────

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import SITE_IDENTITY, settings

# Oturum geçerlilik süresi (gün)
SESSION_TTL_DAYS = 7

# Site bilgileri için varsayılanlar (ilk oluşturmada kullanılır)


# settings tablosundaki özel anahtarlar
KEY_API_KEY = "api_key"
KEY_SETUP_DONE = "setup_done"


class UserStore:
    """SQLite üzerinde çalışan basit ve güvenli veri deposu."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()  # birden fazla istek aynı anda yazmasın

    # ------------------------------------------------------------------
    # İç yardımcılar
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Veritabanı bağlantısını açar; klasörü gerekirse oluşturur."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _run(self, sql: str, params: tuple = (), fetch: str | None = None) -> Any:
        """Kilitli şekilde tek sorgu çalıştırır; sonucu döner."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(sql, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def _now(self) -> str:
        """UTC zamanını ISO biçiminde üretir."""
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Kurulum (şema + varsayılanlar)
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Tablo yapısını oluşturur ve varsayılan kayıtları ekler."""
        self._run(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
            """
        )
        self._run(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        self._run(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self._run(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                key        TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        # Varsayılan ayar anahtarları: yalnızca YOKSA eklenir,
        # mevcut değerler (örn. API anahtarı) asla sıfırlanmaz
        self._run("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (KEY_API_KEY, ""))
        self._run("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (KEY_SETUP_DONE, "0"))
        # Miras geçiş: eski tek settings.api_key varsa ve isimlendirilmiş anahtar
        # tablosu boşsa, eski anahtar "Varsayılan" adıyla içe aktarılır. Böylece
        # mevcut kurulumlar anahtarlarını kaybetmeden yeni API sayfasını görür.
        legacy = self.get_setting(KEY_API_KEY)
        if legacy:
            existing = self._run("SELECT COUNT(*) AS n FROM api_keys", fetch="one")
            if existing and existing["n"] == 0:
                self._run(
                    "INSERT INTO api_keys (name, key, created_at) VALUES (?, ?, ?)",
                    ("Varsayılan", legacy, self._now()),
                )

    # ------------------------------------------------------------------
    # Kullanıcılar
    # ------------------------------------------------------------------

    def user_count(self) -> int:
        """Veritabanındaki kullanıcı sayısını döner."""
        row = self._run("SELECT COUNT(*) AS n FROM users", fetch="one")
        return int(row["n"]) if row else 0

    def create_user(self, username: str, password_hash: str) -> bool:
        """Yeni kullanıcı oluşturur. Kullanıcı adı daha önce varsa False döner."""
        try:
            self._run(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, self._now()),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Kullanıcı adına göre kullanıcı kaydını sözlük olarak döner."""
        row = self._run(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
            fetch="one",
        )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Oturumlar
    # ------------------------------------------------------------------

    def create_session(self, token: str, user_id: int, ttl_days: int = SESSION_TTL_DAYS) -> str:
        """Kullanıcı için yeni bir oturum kaydı oluşturur."""
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(days=ttl_days)).isoformat()
        self._run(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires),
        )
        return token

    def get_session(self, token: str) -> Optional[dict]:
        """Geçerli (süresi dolmamış) oturumu döner; süresi dolmuşsa siler."""
        row = self._run("SELECT * FROM sessions WHERE token = ?", (token,), fetch="one")
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            self.delete_session(token)
            return None
        return dict(row)

    def delete_session(self, token: str) -> None:
        """Oturumu siler (çıkış işlemi)."""
        self._run("DELETE FROM sessions WHERE token = ?", (token,))

    # ------------------------------------------------------------------
    # Anahtar-değer ayarları
    # ------------------------------------------------------------------

    def set_setting(self, key: str, value: str) -> None:
        """Tek bir ayarı yazar (yoksa ekler, varsa günceller)."""
        self._run(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_setting(self, key: str, default: str = "") -> str:
        """Tek bir ayarı okur; yoksa varsayılanı döner."""
        row = self._run("SELECT value FROM settings WHERE key = ?", (key,), fetch="one")
        return row["value"] if row else default

    def is_setup_done(self) -> bool:
        """İlk kurulumun yapılıp yapılmadığını döner."""
        return self.get_setting(KEY_SETUP_DONE, "0") == "1"

    def mark_setup_done(self) -> None:
        self.set_setting(KEY_SETUP_DONE, "1")

    def set_api_key(self, api_key: str) -> None:
        """OpenAI uyumlu API anahtarını kaydeder."""
        self.set_setting(KEY_API_KEY, api_key)

    def get_api_key(self) -> str:
        """Kayıtlı API anahtarını döner."""
        return self.get_setting(KEY_API_KEY)

    # ------------------------------------------------------------------
    # İsimlendirilmiş API anahtarları (birden çok /v1* anahtarı)
    # ------------------------------------------------------------------

    def list_api_keys(self) -> list[dict]:
        """Tüm isimlendirilmiş API anahtarlarını (ad, anahtar, tarih) döner."""
        rows = self._run("SELECT id, name, key, created_at FROM api_keys ORDER BY id", fetch="all")
        return [dict(r) for r in rows]

    def create_api_key(self, name: str, key: str) -> dict:
        """Yeni isimlendirilmiş API anahtarı ekler ve kaydı döner."""
        row_id = self._run(
            "INSERT INTO api_keys (name, key, created_at) VALUES (?, ?, ?)",
            (name, key, self._now()),
        )
        row = self._run(
            "SELECT id, name, key, created_at FROM api_keys WHERE id = ?", (row_id,), fetch="one"
        )
        return dict(row)

    def get_api_key_by_id(self, key_id: int) -> Optional[dict]:
        """Anahtar kimliğine göre kaydı döner; yoksa None."""
        row = self._run(
            "SELECT id, name, key, created_at FROM api_keys WHERE id = ?", (key_id,), fetch="one"
        )
        return dict(row) if row else None

    def delete_api_key(self, key_id: int) -> None:
        """İsimlendirilmiş anahtarı siler."""
        self._run("DELETE FROM api_keys WHERE id = ?", (key_id,))

    def get_all_api_keys(self) -> list[str]:
        """Doğrulama için tüm isimlendirilmiş anahtar değerlerini döner."""
        rows = self._run("SELECT key FROM api_keys", fetch="all")
        return [r["key"] for r in rows]

    # ------------------------------------------------------------------
    # Site bilgileri (proje kimliği + logo/favicon)
    # ------------------------------------------------------------------

    def get_site_info(self) -> dict:
        """Proje kimliğini döner.

        Değerler DB'den değil, kod içine gömülü SITE_IDENTITY sabitinden
        gelir; sistem ayarları değiştirilemez olarak tasarlanmıştır.
        """
        return dict(SITE_IDENTITY)


# ------------------------------------------------------------------
# Singleton erişimi: tüm modüller bu tek örnek üzerinden çalışır.
# İlk çağrıda veritabanı dosyası oluşturulur ve şema hazırlanır.
# ------------------------------------------------------------------

_default_store: Optional[UserStore] = None
_store_lock = threading.Lock()


def get_store() -> UserStore:
    """Uygulama genelinde tek UserStore örneğini döner."""
    global _default_store
    if _default_store is None:
        with _store_lock:
            if _default_store is None:
                _default_store = UserStore(settings.database_file)
                _default_store.init()
    return _default_store