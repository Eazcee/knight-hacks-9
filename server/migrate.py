"""Apply committed migrations using Neon's direct connection: python migrate.py."""
import hashlib
import os
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict

from database_config import load_database_env

MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def direct_url():
    load_database_env()
    url = os.environ.get("DATABASE_URL_UNPOOLED")
    if not url:
        raise RuntimeError("Set DATABASE_URL_UNPOOLED to the direct Neon connection string")
    if "-pooler" in conninfo_to_dict(url).get("host", ""):
        raise ValueError("Migrations require a direct URL, not a -pooler endpoint")
    return url


def apply_migrations(url):
    applied = []
    with psycopg.connect(url, connect_timeout=15) as conn:
        # Transaction-scoped lock also serializes simultaneous deploys.
        conn.execute("SELECT pg_advisory_xact_lock(728194031)")
        conn.execute("CREATE TABLE IF NOT EXISTS beacon_migrations ("
                     "version TEXT PRIMARY KEY, checksum TEXT NOT NULL,"
                     "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        existing = dict(conn.execute("SELECT version, checksum FROM beacon_migrations").fetchall())
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            script = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(script.encode()).hexdigest()
            if path.name in existing:
                if existing[path.name] != checksum:
                    raise RuntimeError(f"Applied migration has changed: {path.name}")
                continue
            conn.execute(script, prepare=False)
            conn.execute("INSERT INTO beacon_migrations (version, checksum) VALUES (%s,%s)",
                         (path.name, checksum))
            applied.append(path.name)
    return applied


if __name__ == "__main__":
    versions = apply_migrations(direct_url())
    print("Applied: " + ", ".join(versions) if versions else "Schema is up to date")
