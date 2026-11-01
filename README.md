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

## For the frontend: phone capture

- "Scan using your phone": `POST /api/capture-sessions` (registrar or admin) returns
  `id`, `capture_url`, `reference_no` and `expires_at`. Draw `capture_url` as a QR
  code with a library bundled into the frontend build (no CDN; the host PC is
  offline). The URL holds a secret and is only returned here, so don't log or store it.
- Poll `GET /api/capture-sessions/{id}` every 2 seconds for `status`, `upload_count`,
  `duplicate_count` and `documents`. Polling does not keep a session alive.
- `POST /api/capture-sessions/{id}/close` ends the session and locks the phone out.
  Closing again, or closing an expired session, returns 200 and changes nothing.
- `status` is `open` (QR not scanned), `claimed` (phone paired), `closed` or `expired`.
  Both of the last two are final: start a new session for a new QR code.
- A session expires after 10 minutes with no phone activity and holds 50 uploads.
- The phone side is served by the backend at `/m/<token>`; there is nothing to build.

## Network setup for phone capture (Sison)

Phones load the capture page directly from the host PC, so it must be reachable
from the teachers' Wi-Fi.

1. **Fixed address.** Give the host PC a static LAN IP or a DHCP reservation. QR
   codes contain this address.
2. **Listen on the LAN.** `uvicorn app.main:app --host 0.0.0.0 --port 8000`, without
   `--reload`. The default binds only to 127.0.0.1, which phones cannot reach.
3. **`.env` on the host PC.** `PUBLIC_BASE_URL=http://<host-ip>:8000` (without it, a QR
   made in a browser on the host PC points at localhost). `COOKIE_SECURE=false` while
   on plain HTTP (with `true` the phone never keeps its upload cookie and every upload
   gets 401). An absolute path for `STORAGE_DIR`.
4. **Host firewall.** On Windows, set the network profile to Private and allow inbound
   TCP 8000 from the school subnets only. Keep PostgreSQL (5432) closed to the LAN.
5. **Wi-Fi and Layer 3 device.** Turn off client/AP isolation on the teachers' SSID if
   the host PC is on the same network, and allow the Wi-Fi subnet to reach the host's
   subnet on TCP 8000. No port forwarding to the internet.
6. **Phones.** This Wi-Fi has no internet, so Android and iOS may quietly use mobile
   data instead, and the page will not load. Choose "stay connected" when the phone
   asks, or turn mobile data off while capturing.
7. **Power.** Stop the host PC sleeping during school hours.

**To test:** on a phone on the teachers' Wi-Fi, open `http://<host-ip>:8000/api/health`,
then scan a real QR code and send one photo. Plain HTTP works because the page uses
the phone's own camera app through a file input, not live camera access.

## Run

- API: `uvicorn app.main:app --reload` (docs at http://127.0.0.1:8000/docs)
- Tests: `pytest -q` (needs the database running; uses a separate `archived_test` database)
- Lint/format: `ruff check . && ruff format .`
- New migration: `alembic revision --autogenerate -m "msg"`

CI (`.github/workflows/ci.yml`) runs Ruff and pytest on every pull request.
