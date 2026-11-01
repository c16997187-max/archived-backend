from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import ApiError


def constraint_name(exc: IntegrityError) -> str | None:
    diag = getattr(exc.orig, "diag", None)
    return diag.constraint_name if diag else None


@contextmanager
def conflicts_as_api_errors(
    session: Session, errors: Mapping[str, Callable[[], ApiError]]
) -> Iterator[None]:
    """Flush the changes made in this block and turn a unique-constraint violation
    into a clean error. Checks made beforehand cannot catch two people saving the
    same record at once; only the database can.

    Runs in a savepoint so the session can still be queried when building the error.
    """
    try:
        with session.begin_nested():
            yield
            session.flush()
    except IntegrityError as exc:
        make_error = errors.get(constraint_name(exc))
        if make_error is None:
            raise
        raise make_error() from None
