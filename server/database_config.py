"""Load and validate the Neon database configuration."""
import os
from pathlib import Path


def load_database_env():
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env.local", override=False)
    load_dotenv(root / ".env", override=False)


def open_database():
    load_database_env()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is required. Run `npx neon@latest env --write .env.local` "
            "or copy .env.example and add your Neon connection string."
        )
    from neon_database import NeonDatabase

    return NeonDatabase(url)
