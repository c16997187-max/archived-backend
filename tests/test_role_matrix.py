"""Every role against every protected route, plus a guard that no route skips
require_role. When you add a route, add it to PROTECTED_ROUTES below."""

import pytest
from fastapi.routing import iter_route_contexts

from app.main import create_app
from tests.conftest import TEST_PASSWORD

# The only routes allowed without require_role (see app/api/auth.py, app/api/health.py).
PUBLIC_ROUTES = {
    ("GET", "/api/health"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/refresh"),
    ("POST", "/api/auth/logout"),
}

ROLES_UNDER_TEST = ["anonymous", "admin", "registrar", "reviewer"]

# (method, path) -> expected status per role.
PROTECTED_ROUTES = {
    ("GET", "/api/me"): {"anonymous": 401, "admin": 200, "registrar": 200, "reviewer": 200},
    ("POST", "/api/auth/change-password"): {
        "anonymous": 401,
        "admin": 200,
        "registrar": 200,
        "reviewer": 200,
    },
    ("POST", "/api/users/{user_id}/reset-password"): {
        "anonymous": 401,
        "admin": 204,
        "registrar": 403,
        "reviewer": 403,
    },
    ("POST", "/api/enrollments"): {
        "anonymous": 401,
        "admin": 201,
        "registrar": 201,
        "reviewer": 403,
    },
    ("GET", "/api/learners"): {"anonymous": 401, "admin": 200, "registrar": 200, "reviewer": 200},
    ("GET", "/api/learners/{learner_id}"): {
        "anonymous": 401,
        "admin": 200,
        "registrar": 200,
        "reviewer": 200,
    },
    ("PATCH", "/api/learners/{learner_id}"): {
        "anonymous": 401,
        "admin": 200,
        "registrar": 200,
        "reviewer": 403,
    },
    ("GET", "/api/dashboard/summary"): {
        "anonymous": 401,
        "admin": 200,
        "registrar": 200,
        "reviewer": 200,
    },
    ("GET", "/api/school-years"): {
        "anonymous": 401,
        "admin": 200,
        "registrar": 200,
        "reviewer": 200,
    },
    ("POST", "/api/school-years"): {
        "anonymous": 401,
        "admin": 201,
        "registrar": 403,
        "reviewer": 403,
    },
    ("GET", "/api/sections"): {"anonymous": 401, "admin": 200, "registrar": 200, "reviewer": 200},
    ("POST", "/api/sections"): {
        "anonymous": 401,
        "admin": 201,
        "registrar": 403,
        "reviewer": 403,
    },
}

# (method, path) -> body, built from the school year made for each test.
REQUEST_BODIES = {
    ("POST", "/api/auth/change-password"): lambda year: {
        "current_password": TEST_PASSWORD,
        "new_password": "new-test-only-password",
    },
    ("POST", "/api/users/{user_id}/reset-password"): lambda year: {
        "temporary_password": "temp-test-only-pw"
    },
    ("POST", "/api/enrollments"): lambda year: {
        "school_year_id": year.id,
        "learner": {"last_name": "Newlast", "first_name": "Newfirst"},
    },
    ("PATCH", "/api/learners/{learner_id}"): lambda year: {"middle_name": "Newmiddle"},
    ("POST", "/api/school-years"): lambda year: {
        "label": "2090-2091",
        "start_date": "2090-06-01",
        "end_date": "2091-03-31",
    },
    ("POST", "/api/sections"): lambda year: {
        "school_year_id": year.id,
        "grade_level": 8,
        "name": "Newsection",
    },
}


def _route_roles(route) -> tuple[str, ...] | None:
    """The roles of the require_role dependency on `route`, if it has one."""
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if hasattr(dep.call, "allowed_roles"):
            return dep.call.allowed_roles
        stack.extend(dep.dependencies)
    return None


def _api_routes():
    # FastAPI keeps included routers nested; iter_route_contexts flattens them
    # the same way the OpenAPI schema does. Docs pages have no dependant.
    for route in iter_route_contexts(create_app().routes):
        if hasattr(route, "dependant"):
            for method in route.methods:
                yield (method, route.path), route


def test_route_walk_finds_the_api_routes():
    # Guards the guard: an empty walk would make the next test pass vacuously.
    assert ("GET", "/api/me") in {key for key, _ in _api_routes()}


def test_every_non_public_route_uses_require_role():
    missing = [
        key for key, route in _api_routes() if key not in PUBLIC_ROUTES and not _route_roles(route)
    ]

    assert missing == []


def test_role_matrix_covers_every_protected_route():
    protected = {key for key, _ in _api_routes()} - PUBLIC_ROUTES

    assert protected == set(PROTECTED_ROUTES)


@pytest.mark.parametrize("role", ROLES_UNDER_TEST)
@pytest.mark.parametrize(("method", "path"), sorted(PROTECTED_ROUTES))
def test_role_access(client, make_user, make_year, make_learner, login, method, path, role):
    target = make_user("registrar")
    year = make_year()
    learner = make_learner()
    headers = {} if role == "anonymous" else login(make_user(role))
    body = REQUEST_BODIES.get((method, path))

    response = client.request(
        method,
        path.format(user_id=target.id, learner_id=learner.id),
        json=body(year) if body else None,
        headers=headers,
    )

    expected = PROTECTED_ROUTES[(method, path)][role]
    assert response.status_code == expected, response.text
    if expected == 401:
        assert response.json()["error"] == "not_authenticated"
    elif expected == 403:
        assert response.json()["error"] == "forbidden"


@pytest.mark.parametrize("role", ["admin", "registrar", "reviewer"])
def test_forced_password_change_blocks_every_role(client, make_user, login, role):
    user = make_user(role, must_change_password=True)
    headers = login(user)

    response = client.post(
        f"/api/users/{user.id}/reset-password",
        json={"temporary_password": "temp-test-only-pw"},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"] == "password_change_required"
    assert client.get("/api/me", headers=headers).status_code == 200


def test_require_role_rejects_unknown_roles():
    from app.api.deps import require_role

    with pytest.raises(ValueError):
        require_role("principal")
    with pytest.raises(ValueError):
        require_role()
