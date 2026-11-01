from datetime import date

from sqlalchemy import func, select

from app.models import AuditLog, Enrollment, Learner, SchoolYear, Section, User
from app.services.passwords import verify_password
from scripts.seed import current_school_year, seed


def test_seed_creates_admin_year_sections_and_20_learners(db):
    seed(db, admin_password="test-only-password", today=date(2026, 11, 1))
    db.flush()

    admin = db.scalars(select(User)).one()
    assert admin.role == "admin"
    assert verify_password("test-only-password", admin.password_hash)

    year = db.scalars(select(SchoolYear)).one()
    assert (year.label, year.is_current) == ("2026-2027", True)
    assert db.scalar(select(func.count()).select_from(Section)) == 13
    assert db.scalar(select(func.count()).select_from(Learner)) == 20
    assert db.scalar(select(func.count()).select_from(Enrollment)) == 20
    assert db.scalars(select(AuditLog.action)).all() == ["seed"]


def test_seeded_enrollments_take_their_grade_from_the_section(db):
    seed(db, admin_password="test-only-password", today=date(2026, 11, 1))
    db.flush()

    for enrollment in db.scalars(select(Enrollment)):
        assert enrollment.grade_level == enrollment.section.grade_level


def test_seeded_learners_have_fake_contact_numbers(db):
    seed(db, admin_password="test-only-password", today=date(2026, 11, 1))
    db.flush()

    numbers = db.scalars(select(Learner.contact_number)).all()
    assert all(n.startswith("09000") and len(n) == 11 for n in numbers)


def test_school_year_rolls_over_in_june():
    assert current_school_year(date(2027, 5, 31))[0] == "2026-2027"
    assert current_school_year(date(2027, 6, 1))[0] == "2027-2028"
