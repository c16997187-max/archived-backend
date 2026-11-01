# ArchivED backend

Offline, intranet-only student enrollment and records system for small
K-12 schools (pilot: Baguio SDA School). Teachers photograph handwritten
enrollment forms with their phones; OCR on the school's host PC extracts
the fields; staff review and approve.

Full spec: `docs/backend-guide.md`. Read the relevant phase before
starting any task.

## Hard constraints
- 100% offline at runtime: no cloud APIs, CDNs, or model downloads.
  Models load from `models/` on disk.
- Host PC: 16 GB RAM, GPU optional. OCR must run on CPU.
- Student data belongs to minors. Never put real student data in
  fixtures, tests, logs, or commit messages. Use `scripts/seed.py` fakes.

## Stack
Python 3.12, FastAPI, SQLAlchemy 2 + Alembic, PostgreSQL, pytest,
Ruff. OCR: PaddleOCR PP-OCRv6 + TrOCR (ONNX). openpyxl for exports.

## Commands
- Start DB: `docker compose up -d db`
- Run API: `uvicorn app.main:app --reload`
- Run worker: `python -m app.workers.run`
- Tests: `pytest -q`
- Lint/format: `ruff check . && ruff format .`
- New migration: `alembic revision --autogenerate -m "msg"`

## Conventions
- Routers in `app/api/` stay thin; logic goes in `app/services/`.
- Every write goes through a service that also writes `audit_log`
  in the same transaction.
- Every endpoint uses `require_role(...)`.
- One error format: `{"error": code, "fields": {...}}`.
- Never hard-delete learners or enrollments; set `deleted_at`.
- Every new endpoint or service gets pytest tests in the same PR.
- Run `pytest -q` and `ruff check .` before saying a task is done.

## Team
Valera: backend (this repo). Fermin: UI, consumes `/docs` OpenAPI.
Sison: network and deployment of the host PC.
