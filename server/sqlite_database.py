"""Local SQLite storage for Beacon, used when no Neon DATABASE_URL is configured.

Same queries as the Neon adapter (DatabaseRepository); embeddings are stored as
raw float32 blobs and matched with NumPy.
"""
from contextlib import contextmanager
import sqlite3
import threading

import numpy as np

from database_repository import DatabaseRepository, atomic

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    notes       TEXT DEFAULT '',
    is_watch    INTEGER DEFAULT 0,
    watch_label TEXT DEFAULT '',
    crop        TEXT,
    created_at  REAL,
    last_seen   REAL
);
CREATE TABLE IF NOT EXISTS embeddings (
    person_id   TEXT,
    vec         BLOB,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   TEXT,
    ts          REAL,
    node_id     INTEGER,
    x           REAL,
    y           REAL,
    sim         REAL,
    crop        TEXT
);
CREATE TABLE IF NOT EXISTS categories (
    name        TEXT PRIMARY KEY,
    sort        INTEGER DEFAULT 0,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    x           REAL,
    y           REAL,
    keyframe    TEXT,
    qr          TEXT,
    created_at  REAL,
    visits      INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS edges (
    a           INTEGER,
    b           INTEGER,
    dist        REAL,
    crowd       REAL DEFAULT 0,
    blocked     INTEGER DEFAULT 0,
    PRIMARY KEY (a, b)
);
"""

# Columns added after the first version. Added in place so existing data survives.
MIGRATIONS = {
    "embeddings": {"crop": "TEXT", "quality": "REAL DEFAULT 0", "source": "TEXT DEFAULT 'auto'"},
    "sightings": {"frame": "TEXT", "box": "TEXT", "quality": "REAL DEFAULT 0", "lat": "REAL", "lng": "REAL"},
    "nodes": {"lat": "REAL", "lng": "REAL", "floor": "INTEGER DEFAULT 1"},
    "people": {"met": "INTEGER DEFAULT 0", "met_at": "REAL", "met_lat": "REAL", "met_lng": "REAL", "category": "TEXT"},
}


class SQLiteDatabase(DatabaseRepository):
    backend = "sqlite"

    def __init__(self, path="data/beacon.db"):
        # Autocommit; transaction() groups statements explicitly.
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        # WAL + normal sync: commits stop waiting on the disk every time (big win on Windows)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.RLock()
        self._depth = 0
        with self.lock:
            self.conn.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():
                have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                for col, typ in cols.items():
                    if col not in have:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    @contextmanager
    def transaction(self):
        with self.lock:
            outer = self._depth == 0
            if outer:
                self.conn.execute("BEGIN")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outer:
                    self.conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self.conn.execute("COMMIT")

    def _run(self, sql, args=(), fetch=None):
        with self.lock:
            cur = self.conn.execute(sql, args)
            if fetch == "one":
                r = cur.fetchone()
                return dict(r) if r else None
            if fetch == "all":
                return [dict(r) for r in cur.fetchall()]
            return cur.lastrowid

    def nearest_people(self, vec, limit=2):
        vec = self.validate_embedding(vec)
        if limit < 1:
            raise ValueError("limit must be positive")
        pids, mat, _ = self.embeddings()
        if not pids:
            return []
        norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
        sims = (mat @ vec) / np.where(norms > 0, norms, 1)
        best = {}
        for pid, sim in zip(pids, sims.tolist()):
            if sim > best.get(pid, -2.0):
                best[pid] = sim
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [{"person_id": pid, "sim": sim} for pid, sim in ranked]

    @atomic
    def clear_map(self):
        """Forget all places (keeps people). Sightings keep their time, lose their place."""
        self._run("DELETE FROM nodes")
        self._run("DELETE FROM edges")
        self._run("UPDATE sightings SET node_id=NULL")
        self._run("DELETE FROM sqlite_sequence WHERE name='nodes'")

    @atomic
    def wipe(self):
        for t in ("people", "embeddings", "sightings", "nodes", "edges", "categories"):
            self._run(f"DELETE FROM {t}")
        self._run("DELETE FROM sqlite_sequence")  # place ids restart at 1
