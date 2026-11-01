import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import CONNECT_TIMEOUT_SECONDS


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
