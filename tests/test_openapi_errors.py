"""The documented errors must be the ones the API really returns. These walk the
whole schema, so a future route that forgets `responses=` fails here."""

import logging
from datetime import UTC, datetime, timedelta

import pytest

from app.log_filters import StripQueryString
from app.main import create_app
from app.schemas.errors import ErrorResponse
from tests.conftest import TEST_PASSWORD

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PUBLIC_OPERATIONS = {
    ("get", "/api/health"),
    ("post", "/api/auth/login"),
    ("post", "/api/auth/refresh"),
    ("post", "/api/auth/logout"),
}
ERROR_SCHEMAS = {"#/components/schemas/ErrorResponse", "#/components/schemas/DuplicateLearnerError"}


@pytest.fixture(scope="module")
def schema():
    return create_app().openapi()


def operations(schema):
    for path, item in schema["paths"].items():
        for method, operation in item.items():
            if method in HTTP_METHODS:
                yield method, path, operation


def test_the_walk_finds_operations(schema):
    assert ("post", "/api/enrollments") in {(m, p) for m, p, _ in operations(schema)}


def test_every_documented_error_uses_our_format(schema):
    wrong = []
    for method, path, operation in operations(schema):
        for status, response in operation["responses"].items():
            if status.startswith("4"):
                ref = response["content"]["application/json"]["schema"].get("$ref")
                if ref not in ERROR_SCHEMAS:
                    wrong.append((method, path, status, ref))

    assert wrong == []


def test_fastapis_default_validation_error_is_gone(schema):
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    assert "ValidationError" not in schema["components"]["schemas"]


def test_error_schema_is_error_and_fields(schema):
    error = schema["components"]["schemas"]["ErrorResponse"]

    assert set(error["properties"]) == {"error", "fields"}
    assert error["properties"]["fields"]["additionalProperties"] == {"type": "string"}


def test_protected_routes_document_401_and_403(schema):
    missing = [
        (method, path)
        for method, path, operation in operations(schema)
        if (method, path) not in PUBLIC_OPERATIONS
        and not {"401", "403"} <= set(operation["responses"])
    ]

    assert missing == []


def test_routes_with_input_document_422(schema):
    missing = [
        (method, path)
        for method, path, operation in operations(schema)
        if (operation.get("parameters") or operation.get("requestBody"))
        and "422" not in operation["responses"]
    ]

    assert missing == []


def test_login_documents_its_errors(schema):
    responses = schema["paths"]["/api/auth/login"]["post"]["responses"]

    assert {"401", "403", "422", "423"} <= set(responses)


def test_enrollment_409_documents_the_matches(schema):
    response = schema["paths"]["/api/enrollments"]["post"]["responses"]["409"]

    assert response["content"]["application/json"]["schema"]["$ref"].endswith(
        "DuplicateLearnerError"
    )
    duplicate = schema["components"]["schemas"]["DuplicateLearnerError"]
    assert {"error", "fields", "matches"} == set(duplicate["properties"])


def test_public_routes_with_no_role_check_do_not_claim_403(schema):
    assert "403" not in schema["paths"]["/api/auth/refresh"]["post"]["responses"]
    assert "401" not in schema["paths"]["/api/health"]["get"]["responses"]


# --- real responses match the documented format ----------------------------------------


def assert_error_format(response, status, error):
    assert response.status_code == status
    body = ErrorResponse.model_validate(response.json())
    assert body.error == error
    assert set(response.json()) == {"error", "fields"}


def test_real_401_matches(client):
    assert_error_format(client.get("/api/learners"), 401, "not_authenticated")


def test_real_403_matches(client, make_user, login):
    headers = login(make_user("reviewer"))

    assert_error_format(client.post("/api/sections", json={}, headers=headers), 403, "forbidden")


def test_real_422_matches(client, make_user, login):
    headers = login(make_user("registrar"))

    response = client.get("/api/learners", params={"grade": 13}, headers=headers)

    assert_error_format(response, 422, "validation_error")
    assert response.json()["fields"] == {"grade": "Must be K or a grade from 0 to 12"}


def test_real_423_matches(client, make_user):
    user = make_user(locked_until=datetime.now(UTC) + timedelta(minutes=5))

    response = client.post(
        "/api/auth/login", json={"id_number": user.id_number, "password": TEST_PASSWORD}
    )

    assert_error_format(response, 423, "account_locked")
    assert "locked_until" in response.json()["fields"]


# --- access log --------------------------------------------------------------------------


def access_record(path):
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", path, "1.1", 200),
        None,
    )


def test_access_log_drops_the_query_string():
    record = access_record("/api/learners?q=Fakename&grade=8")

    assert StripQueryString().filter(record) is True
    assert record.getMessage() == '127.0.0.1:5000 - "GET /api/learners HTTP/1.1" 200'


def test_access_log_leaves_other_paths_alone():
    record = access_record("/api/health")

    assert StripQueryString().filter(record) is True
    assert record.getMessage() == '127.0.0.1:5000 - "GET /api/health HTTP/1.1" 200'


def test_create_app_installs_the_access_log_filter_once():
    create_app()
    create_app()

    filters = [
        f for f in logging.getLogger("uvicorn.access").filters if isinstance(f, StripQueryString)
    ]
    assert len(filters) == 1
