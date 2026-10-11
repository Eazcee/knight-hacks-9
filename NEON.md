# Neon migration handoff

## Goal

Move Beacon's application data from the local SQLite file to Neon Postgres so people, face embeddings, sightings, places, and map edges are stored in a shared cloud database.

Neon will store application records. InsightFace will continue to run in the Python server and perform face detection and recognition. The Android app will continue sending camera frames to the Python server.

```text
Glasses -> Android app -> Python/FastAPI server -> Neon Postgres
                                |
                                +-> InsightFace recognition
```

## Current state

The repository's Git branch is `neon` and tracks `origin/neon`.

Neon infrastructure is set up:

- Project: `Recall`
- Project ID: `old-moon-99463704`
- Region: `aws-us-east-2`
- PostgreSQL: version 18
- Current Neon database branch: `production`
- `neon.ts` declares the base Postgres configuration.
- `neon deploy` completed successfully.
- Neon CLI skills and project dependencies are committed.
- `.neon`, `.env.local`, and `node_modules/` are ignored by Git.
- Psycopg 3 is installed in the local `server/.venv`.

The application has not been migrated yet. `server/db.py` still uses SQLite, and Neon does not yet contain Beacon's tables or face data.

Git branches and Neon database branches are separate. The Git branch happens to be named `neon`; the linked Neon database branch is currently named `production`.

## Continue setup on another computer

Check out the handoff branch and install the committed Node dependencies:

```sh
git switch neon
npm install
```

Install and authenticate the Neon CLI if needed, then link this checkout:

```sh
npm i -g neon@latest
neon login
neon link --project-id old-moon-99463704 --branch production -y
```

Linking writes the ignored `.neon` file and pulls connection variables into the ignored `.env.local`. Access to the Neon project must be granted separately; credentials must never be committed.

Create a Python environment:

```sh
cd server
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "psycopg[binary]"
```

A dependency file still needs to be added, so this manual Psycopg command is temporary.

## Work remaining

1. Add a Python dependency file such as `server/requirements.txt`, including `psycopg[binary]` and the server's other runtime dependencies.
2. Create a Neon development branch before changing the schema. Do not test the migration directly on `production`.

   ```sh
   neon checkout dev-neon-migration --create
   ```

   This updates the ignored local Neon pointer and pulls that branch's environment variables.
3. Add versioned SQL migration files for the existing SQLite tables:
   - `people`
   - `embeddings`
   - `sightings`
   - `nodes`
   - `edges`
4. Translate SQLite-specific schema features to Postgres:
   - `BLOB` to `BYTEA` for face embeddings
   - SQLite autoincrement IDs to Postgres identity columns
   - SQLite migration pragmas to normal versioned migrations
   - SQLite upsert syntax to Postgres `ON CONFLICT`
5. Rewrite `server/db.py` to connect with Psycopg:
   - Use pooled `DATABASE_URL` for normal server traffic.
   - Use direct `DATABASE_URL_UNPOOLED` for schema migrations.
   - Change SQLite `?` parameters to Psycopg `%s` parameters.
   - Return dictionary-like rows so the existing server code continues working.
   - Keep transaction and thread behavior safe for FastAPI's worker threads.
6. Update remaining SQLite-specific SQL in `server/server.py`, including queries that use `rowid` or `?` placeholders.
7. Test the complete flow on the Neon development branch:
   - Create, rename, merge, and delete people.
   - Store and reload 512-dimensional face embeddings.
   - Add, move, and delete sightings.
   - Add places and map edges.
   - Restart the server and confirm recognition data reloads correctly.
8. Decide whether existing SQLite data needs to be migrated. If it does, write a one-time migration that preserves IDs, embeddings, timestamps, and relationships.
9. Decide where JPEG crops, full frames, and keyframes should live. They currently remain under `server/data/` on one computer. Neon Postgres stores database records, not those local files. Neon Object Storage or another object store is needed if multiple server machines must access them.
10. After testing, apply the schema migration to `production`, switch the server to the production connection, and run an end-to-end test from glasses to Android to Python to Neon.

## Definition of done

The switch is complete when:

- Starting the Python server requires `DATABASE_URL` and no longer creates or opens `data/beacon.db`.
- All database tests pass against a disposable Neon branch.
- The production schema is created through committed migration files.
- A person enrolled from the glasses remains available after restarting the Python server.
- Another authorized developer can clone the repository, install dependencies, link Neon, and run the server without undocumented setup.
- No database passwords, connection strings, face photographs, or embeddings are committed to Git.
