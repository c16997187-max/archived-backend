"""Create an admin account from the command line, during installation.

Admins are never created through a public endpoint (Phase 4). Run on the host PC
after `alembic upgrade head`:

    python -m scripts.create_admin --id-number 2026-0001 --full-name "Juan Dela Cruz"

The password is asked for twice and never shown. For an install script, pipe it
in with --password-stdin. Passwords are never accepted as arguments, since those
end up in shell history.
"""

import argparse
import getpass
import sys

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.errors import ApiError
from app.models import User
from app.services.users import create_user


def create_admin(session: Session, id_number: str, full_name: str, password: str) -> User:
    return create_user(session, id_number, full_name, "admin", password)


def _read_password(from_stdin: bool) -> str | None:
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    password = getpass.getpass("Password: ")
    if getpass.getpass("Repeat password: ") != password:
        print("Passwords do not match.", file=sys.stderr)
        return None
    return password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an ArchivED admin account.")
    parser.add_argument("--id-number", required=True, help="School ID number used to log in")
    parser.add_argument("--full-name", required=True)
    parser.add_argument(
        "--password-stdin", action="store_true", help="Read the password from standard input"
    )
    args = parser.parse_args(argv)

    password = _read_password(args.password_stdin)
    if password is None:
        return 1

    with SessionLocal() as session:
        try:
            user = create_admin(session, args.id_number, args.full_name, password)
        except ApiError as exc:
            for field, message in exc.fields.items():
                print(f"{field}: {message}", file=sys.stderr)
            return 1

    print(f"Created admin '{user.id_number}' ({user.full_name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
