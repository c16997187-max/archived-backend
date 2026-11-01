import itertools
import os
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import CONNECT_TIMEOUT_SECONDS, get_db
from app.main import create_app
from app.models import Enrollment, Learner, SchoolYear, Section, User
from app.services.passwords import hash_password


def _test_database_url() -> str:
    """TEST_DATABASE_URL, or DATABASE_URL with `_test` added to the database name,
    so the tests never touch the development data."""
    if url := os.environ.get("TEST_DATABASE_URL"):
        return url
    url = make_url(get_settings().database_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


def _create_database_if_missing(url: str) -> None:
    url_obj = make_url(url)
    admin = create_engine(
        url_obj.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
    )
    with admin.connect() as conn:
        exists = conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url_obj.database}
        )
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{url_obj.database}"'))
    admin.dispose()


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """Test database migrated with Alembic. Downgrading first proves the
    downgrade works and gives every run a clean schema."""
    url = _test_database_url()
    _create_database_if_missing(url)
    engine = create_engine(url, connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS})

    config = Config("alembic.ini")
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        command.downgrade(config, "base")
        command.upgrade(config, "head")

    yield engine
    engine.dispose()


@pytest.fixture
def db(db_engine: Engine) -> Iterator[Session]:
    """Session whose work is rolled back after each test. The session runs in a
    savepoint, so a test can hit an IntegrityError and keep using it after rollback."""
    with db_engine.connect() as conn:
        outer = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            session.close()
            outer.rollback()


# --- API fixtures (Phase 4) ---------------------------------------------------

# Fake credentials for test accounts only.
TEST_PASSWORD = "test-only-password"
TEST_JWT_SECRET = "test-only-jwt-secret-" + "x" * 32
# Hashed once: Argon2 is deliberately slow, and most tests need many users.
_TEST_PASSWORD_HASH = hash_password(TEST_PASSWORD)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, app_env="development", jwt_secret=TEST_JWT_SECRET)


@pytest.fixture
def app(db: Session, settings: Settings) -> FastAPI:
    """App whose requests share the test's rolled-back session. Service commits
    only release a savepoint, so nothing outlives the test."""
    app = create_app(settings)
    app.dependency_overrides[get_db] = lambda: db
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


@pytest.fixture
def make_user(db: Session) -> Callable[..., User]:
    counter = itertools.count(1)

    def _make(role: str = "registrar", password: str = TEST_PASSWORD, **fields: Any) -> User:
        n = next(counter)
        user = User(
            id_number=f"T-{n:04d}",
            full_name=f"Test {role.title()} {n}",
            role=role,
            password_hash=_TEST_PASSWORD_HASH
            if password == TEST_PASSWORD
            else hash_password(password),
            **fields,
        )
        db.add(user)
        db.flush()
        return user

    return _make


@pytest.fixture
def login(client: TestClient) -> Callable[..., dict[str, str]]:
    """Log `user` in through the API and return the Authorization header."""

    def _login(user: User, password: str = TEST_PASSWORD, remember_me: bool = False):
        response = client.post(
            "/api/auth/login",
            json={"id_number": user.id_number, "password": password, "remember_me": remember_me},
        )
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return _login


# --- Record fixtures (Phase 5). All values are fake. ------------------------------


@pytest.fixture
def make_year(db: Session) -> Callable[..., SchoolYear]:
    counter = itertools.count(2000)

    def _make(is_current: bool = False, start_year: int | None = None) -> SchoolYear:
        start = start_year or next(counter)
        year = SchoolYear(
            label=f"{start}-{start + 1}",
            start_date=date(start, 6, 1),
            end_date=date(start + 1, 3, 31),
            is_current=is_current,
        )
        db.add(year)
        db.flush()
        return year

    return _make


@pytest.fixture
def make_section(db: Session) -> Callable[..., Section]:
    counter = itertools.count(1)

    def _make(year: SchoolYear, grade_level: int = 8, name: str | None = None) -> Section:
        section = Section(
            school_year_id=year.id,
            grade_level=grade_level,
            name=name or f"Testsection {next(counter)}",
        )
        db.add(section)
        db.flush()
        return section

    return _make


@pytest.fixture
def make_learner(db: Session) -> Callable[..., Learner]:
    def _make(last_name: str = "Testlast", first_name: str = "Testfirst", **fields: Any) -> Learner:
        learner = Learner(last_name=last_name, first_name=first_name, **fields)
        db.add(learner)
        db.flush()
        return learner

    return _make


@pytest.fixture
def make_enrollment(db: Session) -> Callable[..., Enrollment]:
    def _make(
        learner: Learner, year: SchoolYear, section: Section | None = None, **fields: Any
    ) -> Enrollment:
        fields.setdefault("grade_level", section.grade_level if section else None)
        enrollment = Enrollment(
            learner_id=learner.id,
            school_year_id=year.id,
            section_id=section.id if section else None,
            **fields,
        )
        db.add(enrollment)
        db.flush()
        return enrollment

    return _make
