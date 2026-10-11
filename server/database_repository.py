"""Database operations shared by Beacon's Neon repository.

Everything the glasses learn lives here. Embeddings ("face fingerprints") are
stored with the photo they came from so every person's training set remains
inspectable and correctable.
"""
import time
from functools import wraps

import numpy as np


def atomic(method):
    """Keep multi-statement mutations together, including nested operations."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.transaction():
            return method(self, *args, **kwargs)
    return wrapped

class DatabaseRepository:
    """Backend-neutral Beacon queries implemented by the Neon adapter."""

    def close(self):
        with self.lock:
            self.conn.close()

    def _run(self, sql, args=(), fetch=None):
        raise NotImplementedError

    # ---- people --------------------------------------------------------
    def add_person(self, pid, name=None, notes="", crop=None, is_watch=0, watch_label=""):
        now = time.time()
        self._run(
            "INSERT INTO people (id,name,notes,is_watch,watch_label,crop,created_at,last_seen)"
            " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (id) DO UPDATE SET"
            " name=excluded.name, notes=excluded.notes, is_watch=excluded.is_watch,"
            " watch_label=excluded.watch_label, crop=excluded.crop, last_seen=excluded.last_seen",
            (pid, name, notes, is_watch, watch_label, crop, now, now),
        )

    def person(self, pid):
        return self._run("SELECT * FROM people WHERE id=?", (pid,), fetch="one")

    def people(self):
        """Everyone, newest first, with how many training photos and sightings they have."""
        return self._run(
            "SELECT p.*,"
            " (SELECT COUNT(*) FROM embeddings e WHERE e.person_id=p.id) AS n_train,"
            " (SELECT COUNT(*) FROM sightings s WHERE s.person_id=p.id) AS n_seen,"
            " (SELECT e.crop FROM embeddings e WHERE e.person_id=p.id AND e.crop IS NOT NULL"
            "   ORDER BY e.quality DESC LIMIT 1) AS best_crop"
            " FROM people p ORDER BY p.last_seen DESC",
            fetch="all",
        )

    def update_person(self, pid, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._run(f"UPDATE people SET {cols} WHERE id=?", (*fields.values(), pid))

    def touch_person(self, pid, ts):
        self._run("UPDATE people SET last_seen=CASE WHEN COALESCE(last_seen,0) < ?"
                  " THEN ? ELSE last_seen END WHERE id=?", (ts, ts, pid))

    def find_person_by_name(self, name):
        return self._run("SELECT * FROM people WHERE lower(name)=lower(?)", (name,), fetch="one")

    def person_snapshot(self, pid):
        """Everything about one person, so a delete can be undone."""
        p = self.person(pid)
        if not p:
            return None
        return {"person": p,
                "emb": self._run("SELECT rowid AS rid, * FROM embeddings WHERE person_id=?", (pid,), fetch="all"),
                "sig": self._run("SELECT * FROM sightings WHERE person_id=?", (pid,), fetch="all")}

    @atomic
    def restore_person(self, snap):
        p = snap["person"]
        if self.person(p["id"]):
            return False
        cols = list(p.keys())
        self._run(f"INSERT INTO people ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", tuple(p[c] for c in cols))
        for e in snap["emb"]:
            e = dict(e)
            rid = e.pop("rid")
            c2 = list(e.keys())
            self._run(f"INSERT INTO embeddings (rowid,{','.join(c2)}) VALUES (?,{','.join('?' * len(c2))})", (rid, *[e[c] for c in c2]))
        for r in snap["sig"]:
            c3 = list(r.keys())
            self._run(f"INSERT INTO sightings ({','.join(c3)}) VALUES ({','.join('?' * len(c3))})", tuple(r[c] for c in c3))
        return True

    @atomic
    def delete_person(self, pid):
        self._run("DELETE FROM embeddings WHERE person_id=?", (pid,))
        self._run("DELETE FROM sightings WHERE person_id=?", (pid,))
        self._run("DELETE FROM people WHERE id=?", (pid,))

    @atomic
    def merge_people(self, src, dst):
        """Fold person `src` into `dst`: all training photos and sightings move over.
        Returns a snapshot that undo_merge() can restore, or False."""
        a, b = self.person(src), self.person(dst)
        if not a or not b or src == dst:
            return False
        snap = {"src": a, "dst": b,
                "emb": [r["id"] for r in self._run("SELECT rowid AS id FROM embeddings WHERE person_id=?", (src,), fetch="all")],
                "sig": [r["id"] for r in self._run("SELECT id FROM sightings WHERE person_id=?", (src,), fetch="all")]}
        fields = {}
        if not b["name"] and a["name"]:
            fields["name"] = a["name"]
        if a["notes"]:
            fields["notes"] = (b["notes"] + " · " if b["notes"] else "") + a["notes"]
        if a["is_watch"] and not b["is_watch"]:
            fields["is_watch"], fields["watch_label"] = 1, a["watch_label"]
        fields["last_seen"] = max(a["last_seen"] or 0, b["last_seen"] or 0)
        self.update_person(dst, **fields)
        self._run("UPDATE embeddings SET person_id=? WHERE person_id=?", (dst, src))
        self._run("UPDATE sightings SET person_id=? WHERE person_id=?", (dst, src))
        self._run("DELETE FROM people WHERE id=?", (src,))
        return snap

    @atomic
    def undo_merge(self, snap):
        """Put a merged person back exactly as they were."""
        a, b = snap["src"], snap["dst"]
        if self.person(a["id"]) or not self.person(b["id"]):
            return False
        cols = list(a.keys())
        self._run(f"INSERT INTO people ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", tuple(a[c] for c in cols))
        bcols = [c for c in b.keys() if c != "id"]
        self._run(f"UPDATE people SET {', '.join(c + '=?' for c in bcols)} WHERE id=?", (*[b[c] for c in bcols], b["id"]))
        for eid in snap["emb"]:
            self._run("UPDATE embeddings SET person_id=? WHERE rowid=?", (a["id"], eid))
        for sid in snap["sig"]:
            self._run("UPDATE sightings SET person_id=? WHERE id=?", (a["id"], sid))
        return True

    def prune_anonymous(self, older_than_s):
        """Forget strangers nobody annotated. Keeps the store small and honest."""
        cutoff = time.time() - older_than_s
        rows = self._run(
            "SELECT id FROM people WHERE (name IS NULL OR name='') AND notes='' AND is_watch=0"
            " AND last_seen < ?",
            (cutoff,),
            fetch="all",
        )
        for r in rows:
            self.delete_person(r["id"])
        return len(rows)

    # ---- training photos (embeddings) ----------------------------------
    def add_embedding(self, pid, vec, crop=None, quality=0.0, source="auto"):
        vec = self.validate_embedding(vec)
        return self._run(
            "INSERT INTO embeddings (person_id, vec, created_at, crop, quality, source) VALUES (?,?,?,?,?,?)",
            (pid, np.asarray(vec, dtype=np.float32).tobytes(), time.time(), crop, quality, source),
        )

    @staticmethod
    def validate_embedding(vec):
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (512,) or not np.isfinite(vec).all() or not np.linalg.norm(vec):
            raise ValueError("An embedding must contain 512 finite values and have nonzero length")
        return vec

    def nearest_people(self, vec, limit=2):
        raise NotImplementedError

    def embeddings(self):
        """Returns (person_ids, matrix, embedding_ids) for every stored embedding."""
        rows = self._run("SELECT rowid AS id, person_id, vec FROM embeddings", fetch="all")
        if not rows:
            return [], np.zeros((0, 512), dtype=np.float32), []
        pids = [r["person_id"] for r in rows]
        mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return pids, mat, [r["id"] for r in rows]

    def embeddings_for(self, pid):
        return self._run(
            "SELECT rowid AS id, person_id, crop, quality, source, created_at FROM embeddings"
            " WHERE person_id=? ORDER BY quality DESC",
            (pid,),
            fetch="all",
        )

    def embedding(self, eid):
        return self._run("SELECT rowid AS id, * FROM embeddings WHERE rowid=?", (eid,), fetch="one")

    def delete_embedding(self, eid):
        self._run("DELETE FROM embeddings WHERE rowid=?", (eid,))

    def count_embeddings(self, pid):
        r = self._run("SELECT COUNT(*) AS n FROM embeddings WHERE person_id=?", (pid,), fetch="one")
        return r["n"] if r else 0

    def worst_embedding(self, pid):
        return self._run(
            "SELECT rowid AS id, quality, source FROM embeddings WHERE person_id=?"
            " ORDER BY (source='auto') DESC, quality ASC LIMIT 1",
            (pid,),
            fetch="one",
        )

    # ---- sightings -----------------------------------------------------
    def add_sighting(self, pid, ts, node_id, x, y, sim, crop=None, frame=None, box=None, quality=0.0,
                     lat=None, lng=None):
        return self._run(
            "INSERT INTO sightings (person_id, ts, node_id, x, y, sim, crop, frame, box, quality, lat, lng)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, ts, node_id, x, y, sim, crop, frame, box, quality, lat, lng),
        )

    def first_sighting(self, pid, located=False):
        """Where/when you first saw someone (optionally the first one with a map position)."""
        return self._run(
            "SELECT s.*, n.name AS place FROM sightings s LEFT JOIN nodes n ON n.id=s.node_id"
            " WHERE s.person_id=?" + (" AND s.lat IS NOT NULL" if located else "") + " ORDER BY s.ts ASC LIMIT 1",
            (pid,), fetch="one",
        )

    # ---- categories ----------------------------------------------------
    def categories(self):
        return self._run(
            "SELECT c.name, c.sort, (SELECT COUNT(*) FROM people p WHERE p.category=c.name) AS count"
            " FROM categories c ORDER BY c.sort, c.created_at", fetch="all")

    @atomic
    def add_category(self, name):
        n = self._run("SELECT COUNT(*) AS n FROM categories", fetch="one")["n"]
        self._run("INSERT INTO categories (name, sort, created_at) VALUES (?,?,?)"
                  " ON CONFLICT (name) DO NOTHING", (name, n, time.time()))

    @atomic
    def rename_category(self, old, new):
        self._run("UPDATE categories SET name=? WHERE name=?", (new, old))
        self._run("UPDATE people SET category=? WHERE category=?", (new, old))

    @atomic
    def delete_category(self, name):
        self._run("DELETE FROM categories WHERE name=?", (name,))
        self._run("UPDATE people SET category=NULL WHERE category=?", (name,))

    def last_located_sighting(self, pid):
        """Most recent sighting with a map position."""
        return self._run(
            "SELECT s.*, n.name AS place FROM sightings s LEFT JOIN nodes n ON n.id=s.node_id"
            " WHERE s.person_id=? AND s.lat IS NOT NULL ORDER BY s.ts DESC LIMIT 1",
            (pid,), fetch="one",
        )

    def sighting(self, sid):
        return self._run(
            "SELECT s.*, n.name AS place FROM sightings s LEFT JOIN nodes n ON n.id=s.node_id WHERE s.id=?",
            (sid,),
            fetch="one",
        )

    def delete_sighting(self, sid):
        self._run("DELETE FROM sightings WHERE id=?", (sid,))

    def media_paths(self):
        """Every object-storage path currently referenced by application data."""
        rows = self._run(
            "SELECT crop AS path FROM people WHERE crop IS NOT NULL"
            " UNION SELECT crop FROM embeddings WHERE crop IS NOT NULL"
            " UNION SELECT crop FROM sightings WHERE crop IS NOT NULL"
            " UNION SELECT frame FROM sightings WHERE frame IS NOT NULL"
            " UNION SELECT keyframe FROM nodes WHERE keyframe IS NOT NULL",
            fetch="all",
        )
        return [row["path"] for row in rows if row["path"]]

    def media_path_is_referenced(self, path):
        row = self._run(
            "SELECT EXISTS(SELECT 1 FROM people WHERE crop=?"
            " UNION ALL SELECT 1 FROM embeddings WHERE crop=?"
            " UNION ALL SELECT 1 FROM sightings WHERE crop=? OR frame=?"
            " UNION ALL SELECT 1 FROM nodes WHERE keyframe=?) AS found",
            (path, path, path, path, path), fetch="one",
        )
        return bool(row and row["found"])

    def last_sighting(self, pid, before_ts=None):
        if before_ts is None:
            return self._run(
                "SELECT s.*, n.name AS place FROM sightings s LEFT JOIN nodes n ON n.id=s.node_id"
                " WHERE person_id=? ORDER BY ts DESC LIMIT 1",
                (pid,),
                fetch="one",
            )
        return self._run(
            "SELECT s.*, n.name AS place FROM sightings s LEFT JOIN nodes n ON n.id=s.node_id"
            " WHERE person_id=? AND ts<? ORDER BY ts DESC LIMIT 1",
            (pid, before_ts),
            fetch="one",
        )

    def sightings(self, pid=None, limit=200):
        if pid:
            return self._run(
                "SELECT s.*, n.name AS place FROM sightings s LEFT JOIN nodes n ON n.id=s.node_id"
                " WHERE person_id=? ORDER BY ts DESC LIMIT ?",
                (pid, limit),
                fetch="all",
            )
        return self._run(
            "SELECT s.*, n.name AS place, p.name AS person_name FROM sightings s"
            " LEFT JOIN nodes n ON n.id=s.node_id LEFT JOIN people p ON p.id=s.person_id"
            " ORDER BY ts DESC LIMIT ?",
            (limit,),
            fetch="all",
        )

    # ---- places --------------------------------------------------------
    def add_node(self, name, x, y, keyframe=None, qr=None, lat=None, lng=None):
        return self._run(
            "INSERT INTO nodes (name, x, y, keyframe, qr, created_at, lat, lng) VALUES (?,?,?,?,?,?,?,?)",
            (name, x, y, keyframe, qr, time.time(), lat, lng),
        )

    def node(self, nid):
        return self._run("SELECT * FROM nodes WHERE id=?", (nid,), fetch="one")

    def nodes(self):
        return self._run("SELECT * FROM nodes ORDER BY id", fetch="all")

    def node_by_qr(self, qr):
        return self._run("SELECT * FROM nodes WHERE qr=?", (qr,), fetch="one")

    def node_by_name(self, name):
        name = (name or "").strip()
        n = self._run("SELECT * FROM nodes WHERE lower(name)=lower(?)", (name,), fetch="one")
        if n is None and name.lower().startswith("place ") and name[6:].strip().isdigit():
            n = self.node(int(name[6:].strip()))
        return n

    def update_node(self, nid, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._run(f"UPDATE nodes SET {cols} WHERE id=?", (*fields.values(), nid))

    def bump_visits(self, nid):
        self._run("UPDATE nodes SET visits=visits+1 WHERE id=?", (nid,))

    @atomic
    def add_edge(self, a, b, dist, crowd=0.0):
        if a == b:
            return
        for (u, v) in ((a, b), (b, a)):
            existing = self._run("SELECT * FROM edges WHERE a=? AND b=?", (u, v), fetch="one")
            if existing:
                # keep the shorter measured distance, average the crowd estimate
                dist2 = min(existing["dist"], dist)
                crowd2 = (existing["crowd"] + crowd) / 2
                self._run("UPDATE edges SET dist=?, crowd=? WHERE a=? AND b=?", (dist2, crowd2, u, v))
            else:
                self._run(
                    "INSERT INTO edges (a, b, dist, crowd, blocked) VALUES (?,?,?,?,0)",
                    (u, v, dist, crowd),
                )

    @atomic
    def set_edge(self, a, b, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        for (u, v) in ((a, b), (b, a)):
            self._run(f"UPDATE edges SET {cols} WHERE a=? AND b=?", (*fields.values(), u, v))

    def edges(self):
        return self._run("SELECT * FROM edges", fetch="all")
