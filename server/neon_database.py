"""Neon Postgres storage for Beacon."""
from contextlib import contextmanager
import re
import threading

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from database_repository import DatabaseRepository, atomic


class NeonDatabase(DatabaseRepository):
    backend = "postgres"

    def __init__(self, url):
        self.lock = threading.RLock()
        self.conn = psycopg.connect(url, autocommit=True, row_factory=dict_row,
                                   connect_timeout=15, prepare_threshold=None)
        try:
            ready = self.conn.execute("SELECT to_regclass('beacon_migrations') AS name").fetchone()
            if not ready["name"]:
                raise RuntimeError("Beacon schema missing; run python server/migrate.py first")
            version = self.conn.execute("SELECT 1 FROM beacon_migrations WHERE version=%s",
                                        ("001_beacon.sql",)).fetchone()
            if not version:
                raise RuntimeError("Beacon schema incomplete; run python server/migrate.py first")
            register_vector(self.conn)
        except BaseException:
            self.conn.close()
            raise

    @contextmanager
    def transaction(self):
        with self.lock, self.conn.transaction():
            yield

    @staticmethod
    def _row(row):
        if row is None:
            return None
        row = dict(row)
        # Match the repository's snapshot format for undo/restore and the NumPy gallery.
        if "vec" in row:
            value = row["vec"]
            if hasattr(value, "to_numpy"):
                value = value.to_numpy()
            row["vec"] = np.asarray(value, dtype=np.float32).tobytes()
        # The embedding storage ID is exposed only when explicitly requested.
        row.pop("rowid", None)
        return row

    def _run(self, sql, args=(), fetch=None):
        # Shared repository statements use qmark parameters; values stay separately bound.
        # These statements contain no literal question marks or %-format operators.
        sql = sql.replace("?", "%s")
        args = tuple(np.frombuffer(a, dtype=np.float32).copy()
                     if isinstance(a, (bytes, memoryview)) else a for a in args)
        insert = re.match(r"INSERT INTO (embeddings|sightings|nodes)\b", sql, re.I)
        returning = insert is not None and fetch is None
        if returning:
            sql += " RETURNING " + ("rowid" if insert.group(1).lower() == "embeddings" else "id")
        with self.lock, self.conn.cursor() as cur:
            cur.execute(sql, args)
            if fetch == "one":
                return self._row(cur.fetchone())
            if fetch == "all":
                return [self._row(row) for row in cur.fetchall()]
            if returning:
                return next(iter(cur.fetchone().values()))
            return None

    def nearest_people(self, vec, limit=2):
        vec = self.validate_embedding(vec)
        if limit < 1:
            raise ValueError("limit must be positive")
        # Exact search preserves matching thresholds and distinct runner-up people.
        # ANN is unnecessary for the current gallery size and could miss a match.
        return self._run("SELECT person_id, MAX(1 - (vec <=> ?)) AS sim FROM embeddings"
                         " GROUP BY person_id ORDER BY sim DESC, person_id LIMIT ?",
                         (vec, limit), fetch="all")

    @atomic
    def ensure_model(self, model):
        self._run("INSERT INTO beacon_metadata (key,value) VALUES ('face_model',?)"
                  " ON CONFLICT (key) DO NOTHING", (model,))
        stored = self._run("SELECT value FROM beacon_metadata WHERE key='face_model'", fetch="one")
        if stored["value"] != model:
            raise ValueError(f"Database uses {stored['value']}; BEACON_MODEL must match")

    @atomic
    def clear_map(self):
        self._run("UPDATE sightings SET node_id=NULL")
        self._run("DELETE FROM edges")
        self._run("DELETE FROM nodes")
        # Keep IDs monotonic so undo cannot attach sightings to unrelated new places.

    @atomic
    def wipe(self):
        for table in ("embeddings", "sightings", "edges", "nodes", "people", "categories"):
            self._run(f"DELETE FROM {table}")
        # Model metadata and migration history are configuration, not user records.
