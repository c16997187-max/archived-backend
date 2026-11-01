"""Fill a development database with fake data for UI work.

Creates one admin, the current school year, one section per grade (K to 12),
and 20 fake learners with enrollments. Every name, LRN and phone number is
randomly generated; never put real student data here.

Usage (after `alembic upgrade head`):
    python -m scripts.seed

The admin logs in with ID number "admin" and the password in
SEED_ADMIN_PASSWORD. If that is unset, a random password is generated and
printed once.
"""

import random
import secrets
import sys
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import SessionLocal
from app.models import AuditLog, Enrollment, Learner, SchoolYear, Section, User
from app.models.school import KINDERGARTEN, MAX_GRADE
from app.services.passwords import hash_password

ADMIN_ID_NUMBER = "admin"
LEARNER_COUNT = 20
# Fixed seed so every developer (and Fermin's screenshots) sees the same fake data.
RANDOM_SEED = 2026

LAST_NAMES = [
    "Santos", "Reyes", "Cruz", "Bautista", "Ocampo", "Garcia", "Mendoza", "Torres",
    "Flores", "Villanueva", "Ramos", "Aquino", "Castillo", "Dela Cruz", "Navarro",
    "Pascual", "Domingo", "Salazar", "Mercado", "Agustin",
]  # fmt: skip
GIVEN_NAMES = {
    "M": ["Juan", "Miguel", "Jose", "Gabriel", "Rafael", "Paolo", "Carlo", "Marco", "Andres"],
    "F": ["Maria", "Angela", "Sofia", "Isabel", "Camille", "Bea", "Patricia", "Andrea", "Lea"],
}
BARANGAYS = ["Aurora Hill", "Irisan", "Pacdal", "Loakan Proper", "Camp 7", "Bakakeng Central"]
SECTION_NAMES = [
    "Sampaguita", "Ilang-Ilang", "Rosal", "Dahlia", "Camia", "Orchid", "Jasmine",
    "Narra", "Molave", "Acacia", "Mahogany", "Yakal", "Kamagong",
]  # fmt: skip
# How the 20 enrollments are spread across statuses, so every dashboard count is non-zero.
STATUSES = ["approved"] * 14 + ["pending_review"] * 3 + ["draft"] * 3


def current_school_year(today: date) -> tuple[str, date, date]:
    """Philippine school years start mid-year; before June we are still in last year's."""
    start_year = today.year if today.month >= 6 else today.year - 1
    label = f"{start_year}-{start_year + 1}"
    return label, date(start_year, 6, 1), date(start_year + 1, 3, 31)


def seed(session: Session, admin_password: str, today: date) -> dict[str, int]:
    """Insert the fake data in the caller's transaction and return row counts."""
    rng = random.Random(RANDOM_SEED)

    admin = User(
        id_number=ADMIN_ID_NUMBER,
        full_name="System Administrator",
        password_hash=hash_password(admin_password),
        role="admin",
    )
    session.add(admin)

    label, start, end = current_school_year(today)
    school_year = SchoolYear(label=label, start_date=start, end_date=end, is_current=True)
    session.add(school_year)

    sections = [
        Section(
            school_year=school_year,
            grade_level=grade,
            name=SECTION_NAMES[grade],
            adviser=f"{rng.choice(GIVEN_NAMES['F'])} {rng.choice(LAST_NAMES)}",
        )
        for grade in range(KINDERGARTEN, MAX_GRADE + 1)
    ]
    session.add_all(sections)

    used_lrns: set[str] = set()
    for i, status in enumerate(STATUSES[:LEARNER_COUNT]):
        sex = rng.choice(("M", "F"))
        last_name = rng.choice(LAST_NAMES)
        section = rng.choice(sections)
        # A learner is about grade + 6 years old at the start of the year (K is 5).
        age = section.grade_level + 5
        birth_date = date(start.year - age, rng.randint(1, 12), rng.randint(1, 28))

        lrn = None
        # Every fifth learner is an old archived record with no LRN.
        if i % 5 != 4:
            while (lrn := f"{rng.randrange(10**12):012d}") in used_lrns:
                pass
            used_lrns.add(lrn)

        learner = Learner(
            lrn=lrn,
            last_name=last_name,
            first_name=rng.choice(GIVEN_NAMES[sex]),
            middle_name=rng.choice(LAST_NAMES),
            sex=sex,
            birth_date=birth_date,
            place_of_birth="Baguio City",
            mother_tongue=rng.choice(["Ilocano", "Tagalog", "Kankanaey", "Ibaloi"]),
            religion="Seventh-day Adventist",
            house_street=f"{rng.randint(1, 300)} Fake Street",
            barangay=rng.choice(BARANGAYS),
            municipality="Baguio City",
            province="Benguet",
            father_name=f"{rng.choice(GIVEN_NAMES['M'])} {last_name}",
            mother_name=f"{rng.choice(GIVEN_NAMES['F'])} {rng.choice(LAST_NAMES)}",
            guardian_name=None,
            # 0900 is not an assigned mobile prefix, so these cannot reach a real phone.
            contact_number=f"09000{rng.randrange(10**6):06d}",
        )
        session.add(
            Enrollment(
                learner=learner,
                school_year=school_year,
                section=section,
                status=status,
                enrolled_at=datetime.now(UTC) if status == "approved" else None,
            )
        )

    counts = {"users": 1, "school_years": 1, "sections": len(sections), "learners": LEARNER_COUNT}
    session.flush()
    session.add(AuditLog(user_id=admin.id, action="seed", entity="database", after=counts))
    return counts


def main() -> int:
    settings = get_settings()
    if settings.app_env == "production":
        print("Refusing to seed fake data with APP_ENV=production.", file=sys.stderr)
        return 1

    password = settings.seed_admin_password
    generated = password is None
    if generated:
        password = secrets.token_urlsafe(12)

    with SessionLocal.begin() as session:
        if session.scalar(select(User.id).limit(1)) is not None:
            print("Database already has users; not seeding. Start from an empty database.")
            return 1
        counts = seed(session, password, date.today())

    print("Seeded: " + ", ".join(f"{n} {table}" for table, n in counts.items()))
    if generated:
        print(f"Admin login: ID number '{ADMIN_ID_NUMBER}', password {password}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
