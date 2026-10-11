"""Load the database configuration: Neon when DATABASE_URL is set, local SQLite otherwise."""
import os
from pathlib import Path


def load_database_env():
    try:
        from dotenv import load_dotenv
    except ImportError:          # python-dotenv not installed: only real environment variables count
        return

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env.local", override=False)
    load_dotenv(root / ".env", override=False)


def use_local_storage():
    """True when Beacon keeps everything on this machine (SQLite + image folders) instead of Neon.

    BEACON_STORAGE=local forces it even if .env.local has Neon credentials
    (the tests use it so they can never touch a real Neon database).
    """
    if os.environ.get("BEACON_STORAGE", "").lower() == "local":
        return True
    load_database_env()
    return not os.environ.get("DATABASE_URL")


def open_database(data_dir="data"):
    if use_local_storage():
        from sqlite_database import SQLiteDatabase

        os.makedirs(data_dir, exist_ok=True)
        return SQLiteDatabase(os.path.join(data_dir, "beacon.db"))
    from neon_database import NeonDatabase

    return NeonDatabase(os.environ["DATABASE_URL"])
