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

## Run

- API: `uvicorn app.main:app --reload` (docs at http://127.0.0.1:8000/docs)
- Tests: `pytest -q`
- Lint/format: `ruff check . && ruff format .`
- New migration: `alembic revision --autogenerate -m "msg"`

CI (`.github/workflows/ci.yml`) runs Ruff and pytest on every pull request.
