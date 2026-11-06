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

## OCR models

The worker reads forms with PP-OCRv6 on CPU with ONNX Runtime: text detection, printed
text and, by default, handwriting. TrOCR base-handwritten can read the handwritten fields
instead (`OCR_HANDWRITING_MODEL=trocr`). The worker loads models only from `models/` and
never downloads anything; every file is checked against `models/manifest.json` when the
worker starts.

**Why PP-OCRv6 reads handwriting by default.** The Phase 1 plan named TrOCR for
handwriting. The benchmark below changed that:

- **Speed.** 9.6 s per form against 34.1 s. Phase 9 batches of 300 forms take about 50
  minutes instead of nearly 3 hours.
- **Memory.** 1.1 GB peak against 2.9 GB, which leaves the API and PostgreSQL far more of
  the host's 16 GB.
- **Accuracy on generated forms.** 99.5% of fields exact against 54.3%. TrOCR also invents
  text in blank space and adds stray full stops.
- **License.** Everything PP-OCRv6 uses is Apache-2.0. TrOCR carries the IAM caveat below.

The accuracy figures come from a computer script font, not real handwriting, where TrOCR
may do better. TrOCR stays installed and selectable so both can be scored on the Step 0
forms. Switch only if it wins there by enough to justify the time, the memory and the
license question.

**Getting the models (once, on a machine with internet).** The export tools are large and
pin an older transformers, so they get their own virtual environment:

```bash
py -V:3.12 -m venv .venv-models
.venv-models\Scripts\pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-models.txt
.venv-models\Scripts\python -m scripts.download_models
```

This downloads PP-OCRv6 (small tier; add `--tiers tiny,small,medium` for the others) and
TrOCR, each pinned to a commit and checked against a SHA-256 in the script. A changed
file is refused. Then it exports TrOCR to ONNX: the decoder is quantized to INT8, and the
encoder stays FP32 because every INT8 encoder tried read noticeably worse. Running it
again skips models that are already intact. With the small tier `models/` is about
650 MB, of which TrOCR is 620 MB. With `--skip-trocr` it is about 30 MB, enough for the
default `ppocr`. `models/` is gitignored: copy it to the host PC with the install.
`models/_downloads/` is only a cache and can be deleted.

**Settings** (`.env`): `OCR_HANDWRITING_MODEL` (`ppocr`, the default, or `trocr`), `OCR_DET_TIER`,
`OCR_CONFIDENCE_THRESHOLD`, `OCR_THREADS` (default: one per physical core, so the API
keeps the remaining hyper-threads).

**Comparing the two handwriting models:**

- Speed and memory on this machine: `python -m scripts.bench_ocr --forms 20`
- Accuracy on the Step 0 test set: run `python -m scripts.ocr_debug --images-dir
  private_data/forms --recognize --handwriting trocr --predictions private_data/trocr.csv`,
  then the same with `ppocr`. Score each with `python -m scripts.eval_ocr --labels
  private_data/labels.csv --predictions private_data/trocr.csv`.

**Measured on 2026-11-07** (development laptop: Intel Core i5-12500H, 12 cores / 16
threads, 24 GB RAM, Windows 11; 10 generated forms each, PP-OCRv6 small, 12 threads):

| Handwriting | Load | Seconds per form (p95) | of which preprocessing | Peak memory | CER* | Fields exact* |
| --- | --- | --- | --- | --- | --- | --- |
| `trocr` | 6.5 s | 34.1 (36.4) | 6.8 s | 2.9 GB | 16.3% | 54.3% |
| `ppocr` | 1.3 s | 9.6 (10.2) | 7.1 s | 1.1 GB | 0.1% | 99.5% |

\* The generated forms use a computer script font, not handwriting, so these accuracy
numbers only rank the models on typed-looking text and catch regressions. TrOCR's time
goes into its FP32 encoder (about 1.2 s per line on this laptop), which is why fields where
the detector finds no writing are not sent to it. Re-run on the host PC and on the Step 0
forms before choosing; laptop timings vary with heat.

### Licenses

ArchivED is meant to be sold to schools, so every model and library it ships must allow
commercial use. Checked on 2026-11-07:

| Component | Used for | License |
| --- | --- | --- |
| PP-OCRv6 weights (`PaddlePaddle/PP-OCRv6_*_onnx`) | Detection, printed text, `ppocr` handwriting | Apache-2.0 |
| PaddleOCR (pre/post-processing reimplemented from it) | | Apache-2.0 |
| TrOCR base-handwritten (`microsoft/trocr-base-handwritten`) | `trocr` handwriting | MIT tag on the model card; code (microsoft/unilm) MIT. **See the IAM caveat below.** |
| ONNX Runtime | Running the models | MIT |
| tokenizers | TrOCR's tokenizer | Apache-2.0 |
| huggingface_hub | Pulled in by tokenizers; never imported at runtime | Apache-2.0 |
| psutil | Physical core count, benchmark memory | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | Image processing | Apache-2.0. The wheel bundles FFmpeg under LGPL-2.1: ship the wheel unmodified. |
| NumPy | Arrays | BSD-3-Clause |
| Pillow | Image loading | MIT-CMU (HPND) |
| PyYAML | Reading PP-OCRv6 configs | MIT |
| PyTorch, transformers, optimum, optimum-onnx, onnx | Export only (`requirements-models.txt`); not shipped | BSD-3-Clause; Apache-2.0 for the rest |

**IAM caveat (TrOCR).** TrOCR base-handwritten was fine-tuned on the IAM Handwriting
Database. IAM's terms of use have historically allowed only non-commercial research use.
Its site refused connections when this was checked, so check the current terms yourself. The MIT
tag on the weights may not settle whether a model trained on IAM may be used commercially.
Until that is confirmed in writing (from the IAM maintainers, or by legal review), treat
TrOCR as fine for the pilot and the thesis but not for paid deployments.
**`OCR_HANDWRITING_MODEL=ppocr`, the default, is the license-safe option for commercial
deployments:** everything it uses is Apache-2.0 or more permissive. A third option is to
fine-tune a model only on consented school crops (Phase 7, "Measure and tune").

## Run

- API: `uvicorn app.main:app --reload` (docs at http://127.0.0.1:8000/docs)
- Tests: `pytest -q` (needs the database running; uses a separate `archived_test` database)
- Lint/format: `ruff check . && ruff format .`
- New migration: `alembic revision --autogenerate -m "msg"`

CI (`.github/workflows/ci.yml`) runs Ruff and pytest on every pull request.
