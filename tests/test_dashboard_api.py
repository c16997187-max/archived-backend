"""GET /api/dashboard/summary. Roles are covered by tests/test_role_matrix.py."""

from datetime import UTC, datetime

import pytest


@pytest.fixture
def headers(make_user, login):
    return login(make_user("reviewer"))


def summary(client, headers):
    response = client.get("/api/dashboard/summary", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_counts_current_year_enrollments_by_status(
    client, headers, make_year, make_learner, make_enrollment
):
    year = make_year(is_current=True)
    for status, n in (("draft", 3), ("pending_review", 2), ("approved", 4), ("rejected", 1)):
        for _ in range(n):
            make_enrollment(make_learner(), year, status=status)

    assert summary(client, headers) == {
        "school_year": {"id": year.id, "label": year.label},
        "drafts": 3,
        "pending_review": 2,
        "approved": 4,
        "rejected": 1,
        "total_learners": 10,
    }


def test_other_years_are_not_counted(client, headers, make_year, make_learner, make_enrollment):
    make_year(is_current=True)
    make_enrollment(make_learner(), make_year(), status="draft")

    body = summary(client, headers)

    assert body["drafts"] == 0
    assert body["total_learners"] == 1


def test_soft_deleted_rows_are_not_counted(
    client, headers, make_year, make_learner, make_enrollment
):
    year = make_year(is_current=True)
    now = datetime.now(UTC)
    make_enrollment(make_learner(deleted_at=now), year, status="draft")
    make_enrollment(make_learner(), year, status="draft", deleted_at=now)
    make_enrollment(make_learner(), year, status="draft")

    body = summary(client, headers)

    assert body["drafts"] == 1
    assert body["total_learners"] == 2


def test_no_current_year_gives_zeros(client, headers, make_year, make_learner, make_enrollment):
    make_enrollment(make_learner(), make_year(), status="draft")

    assert summary(client, headers) == {
        "school_year": None,
        "drafts": 0,
        "pending_review": 0,
        "approved": 0,
        "rejected": 0,
        "total_learners": 1,
    }
