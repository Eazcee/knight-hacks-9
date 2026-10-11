# Neon setup

Beacon uses Neon Postgres as its only application database. Face detection and
embedding generation run in the Python server; Neon stores people, 512-D
pgvector embeddings, sightings, categories, places, and map edges.

```text
Glasses -> Android app -> Python/FastAPI + InsightFace -> Neon Postgres
                                                   \-> Neon Object Storage
```

## Project

- Neon project: `Recall` (`rough-math-87664804`)
- Database branch: `production`
- Region: `aws-us-east-2`
- Postgres: version 18 with `vector`
- Private bucket: `recall-images`

Git branches and Neon database branches are separate. The Git branch is named
`neon`; the linked Neon database branch is named `production`.

## Set up a checkout

```sh
git switch neon
npm install
npx -y neon@latest login
npx -y neon@latest link --project-id rough-math-87664804 --branch production -y
npx -y neon@latest env --write .env.local

cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python migrate.py
uvicorn server:app --host 0.0.0.0 --port 8000
```

`.env.local`, `.neon`, model files, and runtime data are local-only and must
never be committed. Server startup fails when `DATABASE_URL` is absent; there
is no hidden SQLite fallback.

Use the pooled `DATABASE_URL` for the application and the direct
`DATABASE_URL_UNPOOLED` for migrations. Migrations are versioned under
`server/migrations/` and validated by checksum.

## Files

- `server/database_repository.py` — shared Beacon data operations
- `server/neon_database.py` — Psycopg and pgvector implementation
- `server/database_config.py` — required connection configuration
- `server/migrate.py` — migration runner
- `server/migrations/` — versioned Postgres schema
- `neon.ts` — Neon service policy, including the private bucket

JPEG crops, frames, and keyframes still live under `server/data/`. The private
bucket is provisioned and verified, but wiring those files into Object Storage
is the next storage task.

## Testing safety

The end-to-end scripts reset their database. Never point them at production.
Create a disposable Neon database branch and export its pooled URL as
`BEACON_TEST_DATABASE_URL` before running them. The scripts refuse to start
without that explicit variable.
