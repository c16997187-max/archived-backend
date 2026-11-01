# ArchivED backend

Offline student enrollment and records API. See `docs/backend-guide.md` for the full spec.

## Local setup (Python 3.12)

```bash
py -V:3.12 -m venv .venv          # Linux/macOS: python3.12 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # then edit JWT_SECRET
```

## Database

**With Docker:** `docker compose up -d db` starts PostgreSQL 17 on `127.0.0.1:5432`
using the `POSTGRES_*` values from `.env`.

**Without Docker** (host PC): install PostgreSQL 17 natively, create a user and
database, and set `DATABASE_URL` in `.env` to match. Then run `alembic upgrade head`.

**Fake data for UI work:** `python -m scripts.seed` adds one admin (ID number
`admin`, password from `SEED_ADMIN_PASSWORD` or printed once), the current school
year, a section per grade and 20 fake learners. It only runs on an empty database.
To start over: `alembic downgrade base && alembic upgrade head`.

## Run

- API: `uvicorn app.main:app --reload` (docs at http://127.0.0.1:8000/docs)
- Tests: `pytest -q` (needs the database running; uses a separate `archived_test` database)
- Lint/format: `ruff check . && ruff format .`
- New migration: `alembic revision --autogenerate -m "msg"`

CI (`.github/workflows/ci.yml`) runs Ruff and pytest on every pull request.
