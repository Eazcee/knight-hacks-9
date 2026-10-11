"""Load Neon CLI's local environment without overriding exported variables."""
import os
from pathlib import Path


def load_database_env():
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env.local", override=False)
    load_dotenv(root / ".env", override=False)


def open_database(sqlite_path):
    from db import DB

    load_database_env()
    url = os.environ.get("DATABASE_URL")
    if url:
        from postgres_db import PostgresDB

        return PostgresDB(url)
    return DB(sqlite_path)
