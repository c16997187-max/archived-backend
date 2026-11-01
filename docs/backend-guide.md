# ArchivED Backend Development Guide

Oct 31, 2026 · @Goy

## Overview

The backend is a single offline server on the school's host PC that receives phone photos, runs OCR, holds the database, and serves every screen in the prototype. Build it in 14 phases: foundation first (stack, database, login, CRUD), then the capture and OCR pipeline, then the safety features (review, audit, backup, exports), then testing and handoff.

**What the backend owns (Valera)**

- The REST API behind every screen in Figures 1 to 9: login, dashboard, Encode Record, Batch Import, Search Records, Record Details, New Enrollment, QR pairing, mobile capture.
- The OCR pipeline and the confidence flags that highlight fields like the red Address and Contact Number boxes in Figure 3.
- The database, audit trail, backups, and DepEd-ready exports.

**Where it meets the rest of the team**

- **Fermin (UX/UI):** agree on an API contract early (Phase 5). Publish it as an OpenAPI spec so the frontend can build against mock data before the backend is done.
- **Sison (network/deployment):** agree on the host PC's static IP, ports, HTTPS certificate approach, and the service install method (Phase 14).

**Constraints from your test feedback that shape every decision**

- 100% offline: no cloud APIs, no CDN, no outside model downloads at runtime. Every dependency and model weight ships with the install.
- The strongest host PC has 16 GB RAM and may have no GPU. OCR must run on CPU in acceptable time.
- A Layer 3 device handles local routing, so phones reach the host PC over the school LAN or Wi-Fi.
- Users have low desktop-software comfort (the Ma'am Teresa persona). The backend must never lose work, and must say clearly what needs checking.

## System architecture

Everything runs on the school's host PC: one API process, one OCR worker process, a PostgreSQL database and a folder of images. Phones and office PCs reach it over the school LAN; nothing leaves the building.

&#91;embedded content: ArchivED backend architecture · 2 clients, 4 server parts, 1 backup\]

The API never runs OCR itself. It saves the photo, queues a job, and answers at once; the worker does the slow OCR work separately, so screens stay fast even during a 300-form batch.

## Phase 1: Choose the tech stack

Use Python for the whole backend, because the OCR models are Python-native and one language keeps a three-person team fast. Pick the stack in week 1 and do not change it later.

| Layer | Recommended | Why | Alternative |
| --- | --- | --- | --- |
| Language | Python 3.11 or 3.12 | Same language as the OCR libraries | None worth the split |
| Web framework | FastAPI + Uvicorn | Async uploads, auto-generated OpenAPI docs for Fermin | Django REST Framework (more built-in admin, heavier) |
| Database | PostgreSQL | Full-text search, JSONB for raw OCR output, solid concurrent writes | SQLite (simpler, fine for 1 school, weaker with many phones uploading at once) |
| ORM + migrations | SQLAlchemy 2 + Alembic | Versioned schema changes | Django ORM if you pick Django |
| Background jobs | A worker process reading a jobs table in PostgreSQL | No extra server to install offline | Redis + RQ or Celery (one more service to maintain) |
| Text detection | PaddleOCR PP-OCRv6 (tiny or small tier) | Light, CPU-friendly, finds text regions on the form | Template-based cropping only (works if forms never change) |
| Handwriting recognition | TrOCR base-handwritten, exported to ONNX or OpenVINO INT8 | Strong on handwriting, runs on CPU after export | PP-OCRv6 recognizer alone (faster, weaker on handwriting) |
| Image processing | OpenCV + Pillow | Deskew, crop, contrast fixes | None needed |
| Exports | openpyxl | Excel files matching SF1 columns | CSV only |
| Testing | pytest + httpx | Standard, works with FastAPI | None needed |

**Notes on the OCR choice**

- PP-OCRv6 was released in June 2026 in three tiers: tiny (1.5M parameters), small (7.7M) and medium (34.5M). Its paper reports the tiny tier running 3.9× faster than PP-OCRv5\_mobile on an Intel Xeon CPU. [source](https://arxiv.org/html/2606.13108)
- TrOCR base-handwritten has about 334M parameters. It runs on CPU but is slow unless exported to ONNX or quantized with OpenVINO. [source](https://ai.gitcode.com/gcw_C8PI9e90/trocr-base-handwritten-npu)
- Neither model was trained on Filipino names and addresses. Do not trust published accuracy. Benchmark both on 30 to 50 real forms from Baguio SDA School in week 2 (see Phase 7) and decide from your own numbers.
- Download every model weight once, store it in the repo's `models/` folder (or a release archive), and load it from disk. The host PC will have no internet.

## Phase 2: Set up the repo and environment

Set up one Git repo with a fixed folder layout and a one-command local run, so all three members test against the same backend from day one.

1. Create the GitHub repo with branches `main` (always deployable) and `dev` (integration). Work on feature branches and merge through pull requests.
2. Pin Python and dependencies in `pyproject.toml` or `requirements.txt` with exact versions. The offline host PC must install the same versions later.
3. Add a `docker-compose.yml` with PostgreSQL for local development. If the host PC cannot run Docker, Sison installs PostgreSQL natively. Keep both paths documented.
4. Keep secrets and machine settings in a `.env` file (database URL, storage path, JWT secret). Commit only `.env.example`.
5. Add a linter and formatter (Ruff) and run pytest on every pull request with GitHub Actions.

**Suggested project structure**

```
archived-backend/
  app/
    main.py            # FastAPI app, router registration
    config.py          # settings from .env
    db/                # engine, session, base model
    models/            # SQLAlchemy tables
    schemas/           # Pydantic request/response models
    api/               # routers: auth, students, enrollments, capture, ocr, batch, audit, export
    services/          # business logic, kept out of routers
    ocr/               # preprocessing, field templates, detection, recognition, scoring
    workers/           # background job runner
  migrations/          # Alembic
  models/              # OCR weights, loaded from disk
  storage/             # uploaded images (gitignored)
  tests/
  scripts/             # seed data, backup, restore
```

Keep routers thin: a router validates input and calls a service. This keeps the OCR and audit logic testable without HTTP.

## Phase 3: Design the database

Separate the learner (who they are) from the enrollment (what they enrolled in each year), and keep every source image and raw OCR result linked to the record it produced. That link is what lets Ma'am Teresa trace any value back to the paper.

| Table | Purpose | Key fields |
| --- | --- | --- |
| `users` | Staff accounts | id, id\_number, full\_name, password\_hash, role, is\_active, last\_login\_at |
| `learners` | One row per student, kept across years | id, lrn (12 digits, unique when present), last\_name, first\_name, middle\_name, extension, sex, birth\_date, place\_of\_birth, mother\_tongue, religion, ip\_group, house\_street, barangay, municipality, province, father\_name, mother\_name, guardian\_name, guardian\_relationship, contact\_number |
| `school_years` | e.g. 2026–2027 | id, label, start\_date, end\_date, is\_current |
| `sections` | Grade + section per year | id, school\_year\_id, grade\_level, name, adviser |
| `enrollments` | Learner in a section for a year | id, learner\_id, school\_year\_id, section\_id, status (draft, pending\_review, approved, rejected), enrolled\_at, remarks |
| `documents` | Every uploaded image | id, file\_path, sha256, mime\_type, source (phone, upload, batch), uploaded\_by, capture\_session\_id, created\_at |
| `form_templates` | Layout of each paper form | id, name, version, field\_boxes (JSON), is\_active |
| `ocr_results` | One row per field per document | id, document\_id, template\_id, field\_name, raw\_text, normalized\_text, confidence, bbox, model\_version |
| `extractions` | One OCR run on one document | id, document\_id, enrollment\_id, status (queued, processing, needs\_review, done, failed), started\_at, finished\_at, error |
| `capture_sessions` | QR pairing between PC and phone | id, token\_hash, user\_id, reference\_no, status, expires\_at |
| `jobs` | Background work queue | id, type, payload (JSON), status, attempts, run\_after, locked\_by, error |
| `audit_log` | Append-only change history | id, user\_id, action, entity, entity\_id, before (JSON), after (JSON), ip\_address, created\_at |

**Rules to apply while building**

- Use UUIDs or bigint IDs, never names, as keys. Two learners can share a name.
- Add a unique constraint on `(learner_id, school_year_id)` in `enrollments`, so one learner cannot be enrolled twice in a year.
- Add a check that `lrn` is exactly 12 digits when present. Old archived records may have no LRN, so allow it to be empty.
- Add indexes on `learners(last_name, first_name)`, `learners(lrn)` and `enrollments(school_year_id, section_id)`. Search Records (Figure 5) uses these.
- Use PostgreSQL trigram search (`pg_trgm`) on names, so a search for "Santoz" still finds "Santos". OCR and spelling mistakes make this essential.
- Never hard-delete learners or enrollments. Use a `deleted_at` column, and record the deletion in `audit_log`.
- Write a seed script that creates one admin, the current school year, sections, and 20 fake learners. Fermin needs this for UI work.

The learner fields follow DepEd's SF1 School Register, which lists LRN, sex, birth date, place of birth, mother tongue, IP group, religion, address, parents, guardian and contact number per learner. Matching it now makes the export in Phase 12 a straight copy. [source](https://www.teacherph.com/download-modified-school-form-1-school-register/)

## Phase 4: Authentication, roles and permissions

Use ID number + password login (Figure 1) with three roles, and check the role on every endpoint, not only in the UI.

| Role | Matches persona | Can do |
| --- | --- | --- |
| Admin | IT-in-charge or principal | Manage users, school years, sections, form templates; view audit log; run backups and restores |
| Registrar | Enrollment-in-charge faculty (Ma'am Teresa) | Capture, encode, batch import, edit drafts, submit for review, search, export |
| Reviewer | Admin reviewers | Approve or reject submitted records, search, view audit log for records they reviewed |

**Steps**

1. Hash passwords with Argon2 or bcrypt (`passlib` or `pwdlib`). Never store or log plain passwords.
2. Issue a short-lived access token (JWT, about 30 minutes) plus a refresh token in an HTTP-only cookie. "Remember Me" extends the refresh token, not the access token.
3. Write one FastAPI dependency, `require_role("registrar", "admin")`, and attach it to every router.
4. Lock an account for 15 minutes after 5 failed logins and log each failure. This matters even offline, since anyone on the school Wi-Fi can reach the login page.
5. "Forgot Password" cannot send email offline. Make it an admin reset: the admin sets a temporary password and the user must change it at next login.
6. Create the first admin account with a command-line script during installation, never through a public endpoint.
7. Log every login, logout and failed login to `audit_log`.

## Phase 5: Build the core API

Agree on the API contract with Fermin before writing logic, then build endpoints screen by screen. FastAPI generates the contract at `/docs` automatically; share it and freeze field names early.

| Screen (prototype figure) | Method and path | What it does |
| --- | --- | --- |
| Login (Fig. 1) | `POST /api/auth/login`, `POST /api/auth/refresh`, `POST /api/auth/logout` | Sign in and out |
| Dashboard (Fig. 2) | `GET /api/me`, `GET /api/dashboard/summary` | Current user, counts of drafts and items to review |
| New Enrollment (Fig. 7) | `POST /api/enrollments` | Manual entry; creates learner (or links existing) and a draft enrollment |
| Search Records (Fig. 5) | `GET /api/learners?q=&grade=&section=&school_year=&page=` | Fuzzy search by name or ID, paginated |
| Record Details (Fig. 6) | `GET /api/learners/{id}`, `PATCH /api/learners/{id}` | View and edit; edits go to the audit log |
| Encode Record (Fig. 3) | `POST /api/documents`, `GET /api/extractions/{id}`, `PUT /api/extractions/{id}/review` | Upload one image, poll OCR status, save corrected fields |
| Batch Import (Fig. 4) | `POST /api/batches`, `GET /api/batches/{id}`, `POST /api/batches/{id}/confirm` | Many files, queue status, confirm import |
| QR pairing (Fig. 8, 9) | See Phase 6 | Phone capture |
| Review | `POST /api/enrollments/{id}/approve`, `POST /api/enrollments/{id}/reject` | Reviewer decisions |
| Settings | `GET/POST /api/users`, `GET/POST /api/sections`, `GET/POST /api/school-years` | Admin setup |

**Rules for every endpoint**

- Validate all input with Pydantic: dates are real dates, contact numbers match `09XXXXXXXXX`, LRN is 12 digits, grade level is in K to 12.
- Return one error format everywhere, e.g. `{"error": "validation_error", "fields": {"contact_number": "Must be 11 digits starting with 09"}}`. The UI turns this into the red field messages in Figure 3.
- Paginate every list (default 25 rows).
- Before creating a learner, check for likely duplicates (same LRN, or same name + birth date) and return them so the UI can ask "Is this the same student?".
- Wrap each create or update and its audit-log row in one database transaction, so neither can be saved without the other.

## Phase 6: QR session pairing and phone capture

Serve the phone capture screen as a plain web page from the backend, so teachers scan a QR code and shoot photos with no app to install. Pair each phone to one desktop session with a short-lived token.

**The flow (Figures 8 and 9)**

1. The registrar clicks "Scan using your phone". The desktop calls `POST /api/capture-sessions`. The backend creates a session with a random 32-byte token (store only its hash), a reference number, and a 10-minute expiry.
2. The desktop shows a QR code for `http://<host-ip>:<port>/m/<token>`. Sison fixes the host PC's IP so this URL stays valid.
3. The phone opens the URL. The page calls `POST /api/capture-sessions/{token}/claim`. The backend marks the session claimed and gives the phone an upload-only cookie. A second phone cannot claim the same token.
4. The teacher takes the photo, adjusts the crop, and taps Submit. The phone sends `POST /api/capture-sessions/{token}/images` with the image and the four crop corners.
5. The backend saves the original file, records the crop corners, creates a `documents` row and queues an OCR job. It replies with the document ID.
6. The desktop learns of the upload by polling `GET /api/capture-sessions/{id}` every 2 seconds, or through Server-Sent Events. Either works offline; polling is simpler to build first.
7. The phone shows "Sent to dashboard for review" with the reference number.

**The camera trap to avoid**

Browsers allow live camera access (`getUserMedia`) only on HTTPS pages or `localhost`. A phone opening `http://192.168.x.x` will be blocked. You have two options:

- **Simplest:** use `<input type="file" accept="image/*" capture="environment">`. It opens the phone's own camera app and works over plain HTTP. Do the crop step in the page after the photo is taken.
- **Nicer, more setup:** Sison creates a local certificate authority (for example with mkcert), issues a certificate for the host PC, and installs the CA on each teacher's phone. This enables live preview and HTTPS everywhere.

Start with the file-input approach; add HTTPS later if time allows.

**Upload rules**

- Accept JPEG and PNG up to 15 MB. Convert iPhone HEIC with `pillow-heif` if needed.
- Name stored files by their SHA-256 hash. The same photo sent twice is detected and not processed twice, which answers "I do not want to encode the same record twice".
- Write the file to disk and flush it before replying "saved". This is the "immediate backup after capture" feature from your impact-effort matrix; Phase 11 copies it to a second drive.
- Limit each session to 50 uploads and expire it after 10 minutes idle.
- The phone cookie can only upload to its own session. It can never read student records.

## Phase 7: Build the OCR pipeline

Treat OCR as "read each known box on a known form", not "read the whole page". The school's enrollment forms have a fixed layout, so you can cut out each field and recognize it alone, which is far more accurate than full-page OCR.

**Step 0: Build your test set first (week 2)**

1. Get 30 to 50 filled enrollment forms from Baguio SDA School, with the school's permission. Photograph them with 2 or 3 different phones in normal classroom light.
2. Type the correct value of every field into a spreadsheet. This is your ground truth.
3. Photograph one blank form flat and sharp. This becomes the template image.
4. Every OCR change is measured against this set. Without it, you cannot defend your accuracy claims in the defense.

**The pipeline, run by the background worker for each document**

1. **Load and orient.** Read the image and apply its EXIF rotation.
2. **Flatten.** Apply the four crop corners from the phone with a perspective warp, producing a fixed-size page (for example 1700 × 2200 px).
3. **Align to the template.** Match the photo to the blank template (OpenCV ORB features + homography, or locate printed labels with PP-OCRv6). This corrects small shifts and tilts.
4. **Cut the fields.** Use the template's `field_boxes` JSON to crop each field: name, birth date, address, grade, guardian, contact number, and so on.
5. **Clean each crop.** Grayscale, contrast boost (CLAHE), light denoise, and remove printed ruling lines with morphology so they are not read as "1" or "l".
6. **Recognize.** Handwritten fields go to TrOCR; printed fields go to the PP-OCRv6 recognizer. Checkbox fields (such as sex) are decided by counting dark pixels in each box, not by OCR.
7. **Score confidence.** Take the per-character probabilities from the model and keep the lowest one as the field's confidence. One unsure character is enough to flag a field.
8. **Normalize.** Dates to `YYYY-MM-DD`; contact numbers to 11 digits starting with 09 (convert +63); LRN digits only; "Gr. 8" to 8; names trimmed and title-cased.
9. **Validate and flag.** Mark a field `needs_check` if confidence is below the threshold, the value fails its rule, or a required field is empty. These become the red fields in Figure 3.
10. **Save.** Write raw text, normalized text, confidence, box position and model version to `ocr_results`, then set the extraction to `needs_review`.

**Fit it in 16 GB of RAM**

- Load both models once when the worker starts, never per request.
- Run one OCR worker process and limit ONNX Runtime threads to the number of physical cores, so the API stays responsive while OCR runs.
- Use the INT8 or FP16 export of TrOCR and batch all field crops of one form into one call.
- Measure memory and seconds per form on the actual host PC (or a laptop with the same RAM) and record them. Your panel will ask.

**Measure and tune**

- Report two numbers on the test set: character error rate per field, and the share of fields read exactly right.
- Set the confidence threshold so most wrong fields get flagged, even if some correct ones are flagged too. A flagged correct field costs the registrar a glance; an unflagged wrong one becomes a bad record.
- If handwriting accuracy is too low, fine-tune TrOCR on labeled field crops. Do the training on a separate GPU machine during development, and use only consented or anonymized samples.

## Phase 8: Validation and human review

No OCR value becomes an official record until a person confirms it. The backend enforces this with record states, so the rule holds even if the UI has a bug.

**Record states**

1. `queued` → `processing`: the worker picks up the OCR job.
2. `processing` → `needs_review`: OCR finished; flagged fields are marked. (`failed` if the image is unreadable, with a reason such as "too blurry" shown to the user.)
3. `needs_review` → `pending_approval`: the registrar corrects fields and saves (Figure 3, Save Record). The backend refuses this step while any required field is empty or invalid.
4. `pending_approval` → `approved` or `rejected`: a reviewer decides. Rejection requires a reason and sends the record back to `needs_review`.
5. Only `approved` records appear in exports and count as enrolled.

**Server-side validation rules (one file, `services/validation.py`)**

- Required: last name, first name, birth date, sex, grade level, school year, guardian or parent name.
- Birth date is in the past, and age fits the grade level within a tolerance (for example a Grade 8 learner aged 9 is flagged, not blocked).
- LRN is 12 digits and not already used by another learner.
- Contact number is 11 digits starting with 09.
- Grade and section exist for the selected school year.
- Prerequisites (from your value proposition): flag missing documents such as a birth certificate or previous report card, if the school tracks them.

**Keep both values**

When the registrar corrects a field, store the OCR value and the corrected value. The difference tells you exactly where OCR fails, which becomes data for tuning in Phase 7 and evidence for your paper's evaluation chapter.

## Phase 9: Batch import and the job queue

Run all OCR in a background worker, never inside a web request. One phone photo and 300 archived scans then use the same path, and the API never freezes.

**Job queue in PostgreSQL**

1. Uploading creates a `jobs` row: `type = 'ocr'`, `status = 'queued'`, `payload = {document_id}`.
2. The worker loops: claim the oldest queued job with `SELECT … FOR UPDATE SKIP LOCKED`, mark it `running`, process it, mark it `done` or `failed`.
3. On failure, increase `attempts` and retry up to 3 times with a delay. After that, mark it `failed` with the error so the UI can show it.
4. On worker start, reset jobs stuck in `running` for over 10 minutes back to `queued`. This recovers from a power cut mid-job.
5. Run the worker as its own process (`python -m app.workers.run`) next to the API.

**Batch Import (Figure 4)**

1. `POST /api/batches` accepts many files (or one ZIP), saves each as a document, and creates one OCR job per file. It returns a batch ID immediately.
2. `GET /api/batches/{id}` returns each file's state (Queued, Processing, Done, Failed) for the Processing Queue list, and the extracted name and grade for the Converted Records Preview.
3. `POST /api/batches/{id}/confirm` moves every finished record into `needs_review`. Files with flagged fields still need a person to check them.
4. Skip duplicates by file hash and report them as "already imported".

**Archived paper records**

Old records may use older form layouts. Add a template per layout and let the user pick the form type when uploading a batch. For layouts with no template, fall back to full-page detection and recognition and flag every field for review.

## Phase 10: Audit trail

Record who changed what, when, and from what value, for every write. This directly answers the persona's fear "Will I be blamed if the encoding is wrong?": the log shows whether a value came from OCR, from her correction, or from someone else.

1. Write audit rows in the service layer, inside the same transaction as the change (Phase 5).
2. Log these actions: login, logout, failed login, create, update, approve, reject, soft delete, restore, export, backup, password reset, template change.
3. Store `before` and `after` as JSON of only the fields that changed, plus the source of the value: `ocr`, `manual`, or `import`.
4. Make the table append-only: the app's database user gets `INSERT` and `SELECT` on `audit_log`, never `UPDATE` or `DELETE`.
5. Add `GET /api/audit?entity=learner&entity_id=…` so the Record Details screen can show a history tab.
6. Never write passwords, tokens or full images into the log.

## Phase 11: Backup and recovery

Back up automatically every day to a second physical drive, and prove a restore works before the pilot. The whole value of ArchivED is that records stop getting lost; a dead hard drive must not undo that.

**What to back up**

- The database, with `pg_dump` in custom format (`-Fc`).
- The `storage/` folder of original images.
- The model files and `.env` (once, at install, and after changes).

**How**

1. Write `scripts/backup.py`: dump the database, copy new image files, and write both to a dated folder on an external USB drive or a second internal disk.
2. Schedule it daily after school hours (Windows Task Scheduler or a Linux systemd timer). Sison sets this up.
3. Keep 14 daily and 12 monthly copies; delete older ones automatically.
4. Encrypt backups with a password kept by the school head, since the drive can be carried out of the building.
5. Write `scripts/restore.py` and run a full restore onto a spare machine at least once. A backup that has never been restored is untested.
6. Show "Last backup: date, OK/FAILED" on the admin dashboard, and log every run to `audit_log`.

**Immediate protection between daily backups**

Copy each new uploaded image to the backup drive as part of its OCR job. The original photo, the hardest thing to recreate, is then safe within minutes of capture.

## Phase 12: DepEd exports and data privacy

Export approved records as an Excel file laid out like SF1, to speed up encoding into DepEd's Learner Information System (LIS). Do not present it as an official school form: LIS-generated school forms are the official versions and are not meant to be edited outside the system. [source](https://docs.google.com/presentation/d/1TcK2rtB9_SEFvEToBUgki7PneoQ8yYAg/htmlpresent) Word the "DepEd EMIS-ready exports" claim in your paper as "LIS-encoding-ready" to stay accurate.

**Exports**

1. `GET /api/exports/sf1?school_year=&section=` builds an `.xlsx` with openpyxl: one row per approved learner, columns in SF1 order (LRN, name, sex, birth date, place of birth, mother tongue, IP group, religion, address parts, father, mother, guardian, relationship, contact number, remarks).
2. Also offer a plain CSV of the same data and a per-learner PDF summary for the paper file.
3. Restrict exports to registrar and admin roles, and log each export with who, when, and how many records.
4. Ask the school's LIS coordinator to check one export against what they actually encode, and adjust the columns from their feedback.

**Data privacy (Republic Act 10173, the Data Privacy Act of 2012)**

The records hold minors' personal data, and religion and ethnic or IP group count as sensitive personal information under the law. Build these in:

- **Collect only what is needed.** If a field has no use, do not add a column for it.
- **Access control.** Role checks on every endpoint (Phase 4); the phone upload cookie cannot read data (Phase 6).
- **Encryption at rest.** Sison turns on full-disk encryption on the host PC (BitLocker or LUKS) and backups are encrypted (Phase 11).
- **Network.** The server listens only on the school LAN; no port forwarding to the internet.
- **Retention.** Agree with the school how long rejected drafts and unused images are kept, then delete them on schedule.
- **Correction requests.** Corrections go through the normal edit flow, so they appear in the audit log.
- **Breach plan.** A one-page procedure: who to tell, and how to restore from backup if the PC is stolen or fails.
- **Privacy notice.** Help the school add a short notice to the paper enrollment form explaining that the form will be scanned and stored digitally.

## Phase 13: Testing

Test at four levels, and keep the results: they become the evaluation chapter of your capstone paper.

| Level | What to test | Tool | Pass when |
| --- | --- | --- | --- |
| Unit | Validation rules, normalizers (dates, phone numbers, LRN), permission checks | pytest | Every rule has a passing and a failing case |
| API | Every endpoint, including wrong role, bad input, duplicates | pytest + httpx against a test database | All endpoints return the documented status codes and error format |
| OCR accuracy | The 30–50 form test set from Phase 7 | A script that prints error rates and flag recall | Numbers recorded per release; flagged fields catch nearly all wrong ones |
| Load and stability | 3–5 phones uploading at once during a simulated peak day; 300-file batch | Locust or a simple script | No lost uploads, API stays responsive, memory stays under the host PC's limit |

**Also test the failures**

- Pull the network cable mid-upload: the phone shows an error and the retry succeeds without a duplicate record.
- Kill the worker mid-job: the job is retried after restart.
- Fill the disk: uploads are refused with a clear message, not silently lost.
- Restore from backup onto a clean machine: all records and images come back.

**User acceptance test with the school**

Run a session with the registrar at Baguio SDA School using real (or realistic) forms. Time how long one form takes by hand versus with ArchivED, and record her corrections. This turns "reclaims instructional time" from a claim into a measured result.

## Phase 14: Deployment handoff to Sison

Package the backend so it installs on an offline PC from a USB drive, and hand Sison a written checklist. Agree on these items with him by mid-project, not at the end.

**What the backend hands over**

- [ ] An offline install bundle: Python wheels for every dependency (`pip download -r requirements.txt`), model weights, and the app code. Or a Docker image saved with `docker save`, if the host PC runs Docker.
- [ ] A production `.env.example` with every setting explained.
- [ ] `scripts/init.py` that runs migrations and creates the first admin.
- [ ] Two service definitions: the API (Uvicorn) and the OCR worker, both set to start on boot and restart on crash (Windows service via NSSM, or systemd units on Linux).
- [ ] Backup and restore scripts with the schedule (Phase 11).
- [ ] A health endpoint, `GET /api/health`, that reports database, disk space, worker status and last backup.

**What to agree with Sison**

| Item | Decide |
| --- | --- |
| Host PC operating system | Windows or Linux; affects service and disk-encryption setup |
| Static IP and port | Used in the QR URL; e.g. one fixed LAN address and port 8000 |
| HTTPS | Plain HTTP with file-input capture, or a local CA installed on phones (Phase 6) |
| Wi-Fi and Layer 3 routing | Phones on the teachers' Wi-Fi must reach the host PC's subnet |
| Firewall | Allow the app port from the school LAN only; block everything from outside |
| Static file serving | Who serves the frontend build: FastAPI itself, or a reverse proxy such as Caddy or Nginx |
| Updates | How a new version is installed: USB bundle, migration, restart, rollback plan |

## Suggested timeline

Start the OCR work early and in parallel with the CRUD screens, because it carries the most risk. The plan below assumes 16 working weeks; stretch or compress it to fit your actual term.

&#91;embedded content: Backend build plan · 14 workstreams, 4 milestones, 16 weeks\]

Freeze the API contract by the end of week 3 so Fermin can build screens against it, show OCR working end to end by week 9, and stop adding features after week 12 so the last month goes to testing, packaging and the pilot with Sison.

## Definition of done

The backend is ready for the pilot when every box below is ticked.

- [ ] Every screen in Figures 1 to 9 works against the real API, not mock data
- [ ] A phone on the school Wi-Fi can pair by QR, capture, and upload with no app installed
- [ ] OCR runs on the host PC's hardware, with seconds per form and memory use measured and recorded
- [ ] Accuracy on the test set is measured and written up, with the confidence threshold justified
- [ ] No record reaches `approved` without a person reviewing its flagged fields
- [ ] Every create, edit, approval and export appears in the audit log with user and time
- [ ] Role checks are enforced on the server for every endpoint
- [ ] Daily encrypted backup runs automatically, and a full restore has been done once
- [ ] SF1-layout Excel export checked by the school's LIS coordinator
- [ ] Offline install bundle installs on a clean PC with the internet unplugged
- [ ] Automated tests pass on every pull request
- [ ] Registrar user acceptance session done, with time per form compared to manual encoding

## Sources

- [PP-OCRv6 paper (arXiv, June 2026)](https://arxiv.org/html/2606.13108): model tiers, sizes and CPU speed
- [TrOCR base-handwritten CPU/NPU adaptation notes](https://ai.gitcode.com/gcw_C8PI9e90/trocr-base-handwritten-npu): parameter count
- [Modified School Form 1 (SF1) data elements, TeacherPH](https://www.teacherph.com/download-modified-school-form-1-school-register/): learner fields
- [DepEd guidelines on preparing and checking school forms](https://docs.google.com/presentation/d/1TcK2rtB9_SEFvEToBUgki7PneoQ8yYAg/htmlpresent): LIS-generated forms are not edited outside LIS
