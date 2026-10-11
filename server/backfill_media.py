"""Upload locally retained Beacon images referenced by Neon Postgres."""
from pathlib import Path

from database_config import open_database
from media_storage import MediaStorage

HERE = Path(__file__).resolve().parent


def local_file(path):
    normalized = Path(str(path).replace("\\", "/"))
    candidates = [normalized] if normalized.is_absolute() else [Path.cwd() / normalized, HERE / normalized]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def main():
    db = open_database()
    media = MediaStorage()
    uploaded = skipped = missing = 0
    try:
        existing = set(media.list_keys())
        for path in db.media_paths():
            key = media.key(path)
            if key in existing:
                skipped += 1
                continue
            source = local_file(path)
            if source is None:
                missing += 1
                continue
            media.put_bytes(path, source.read_bytes())
            existing.add(key)
            uploaded += 1
    finally:
        db.close()
    print(f"Uploaded {uploaded}; already stored {skipped}; local files missing {missing}")


if __name__ == "__main__":
    main()
