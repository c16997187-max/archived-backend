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

**First admin (development and production):** `python -m scripts.create_admin
--id-number <ID> --full-name "<Name>"` asks for the password twice. For an
install script, pipe the password in with `--password-stdin`. There is no public
sign-up endpoint; admins create the other accounts.

## Settings

All settings are in `.env.example`. The ones that change behaviour:

- `APP_ENV=production` hides `/docs` and refuses to start without a strong
  `JWT_SECRET` (32+ characters).
- `CORS_ORIGINS` (comma-separated): browser origins allowed to call the API with
  cookies. Unset means `http://localhost:5173` in development and none in
  production, where the API serves the frontend from the same origin.
- `COOKIE_SECURE=true` once the host PC serves HTTPS.
- `ACCESS_TOKEN_MINUTES`, `REFRESH_TOKEN_HOURS`, `REMEMBER_ME_DAYS`: login lifetimes.

## For the frontend: login

- `POST /api/auth/login` with `{id_number, password, remember_me}` returns an
  `access_token` (30 minutes) and sets an HTTP-only `refresh_token` cookie.
  Send the token as `Authorization: Bearer <token>`.
- Keep the access token in memory (a variable or store), never in
  `localStorage` or `sessionStorage`.
- On page load, call `POST /api/auth/refresh` to restore the session; it also
  gives a new access token whenever the old one expires. Send requests with
  credentials (`fetch(..., {credentials: "include"})`) so the cookie goes along.
- On a 401 from `/api/auth/refresh`, retry once before sending the user to login.
- If a response has `must_change_password: true`, show the change-password
  screen (`POST /api/auth/change-password`); other routes return 403
  `password_change_required` until then.
- `POST /api/auth/logout` ends the session and clears the cookie.
- Errors always look like `{"error": "code", "fields": {"field": "message"}}`.
  Login errors: `invalid_credentials` (401), `account_disabled` (403),
  `account_locked` (423, `fields.locked_until`; 5 wrong passwords lock the
  account for 15 minutes).

## Run

- API: `uvicorn app.main:app --reload` (docs at http://127.0.0.1:8000/docs)
- Tests: `pytest -q` (needs the database running; uses a separate `archived_test` database)
- Lint/format: `ruff check . && ruff format .`
- New migration: `alembic revision --autogenerate -m "msg"`

CI (`.github/workflows/ci.yml`) runs Ruff and pytest on every pull request.
