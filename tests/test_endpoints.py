"""Endpoint tests for a running admission service container.

Expected flow:
1) Start the API container (for example: `make run`).
2) Run `pytest -v` from a Python environment with dependencies installed.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import requests


# ==========
# I. Congigs and setups
# ==========


BASE_URL = os.getenv("ADMISSION_BASE_URL", "http://127.0.0.1:3000")
LOGIN_URL = f"{BASE_URL}/login"
PREDICT_URL = f"{BASE_URL}/v1/models/admission_lr/predict"

# Keep this configurable for enviromnents where the service secret difers.
JWT_SECRET = os.getenv(
    "TEST_JWT_SECRET",
    "my_super_secret_key_that_is_at_least_32_characters_long",
)
JWT_ALGORITHM = "HS256"

VALID_CREDENTIALS = {
    "credentials": {
        "username": "admin",
        "password": "password",
    }
}

INVALID_CREDENTIALS = {
    "credentials": {
        "username": "admin",
        "password": "wrong",
    }
}

VALID_INPUT = {
    "data": {
        "gre_score": 330,
        "toefl_score": 115,
        "university_rating": 5,
        "sop": 4.5,
        "lor": 4.5,
        "cgpa": 9.5,
        "research": 1,
    }
}

INVALID_INPUT_MISSING_FIELD = {
    "data": {
        # Missing required field: gre_score
        "toefl_score": 115,
        "university_rating": 5,
        "sop": 4.5,
        "lor": 4.5,
        "cgpa": 9.5,
        "research": 1,
    }
}


@pytest.fixture(scope="session", autouse=True)
def wait_for_service_ready():
    """Wait briefly for the containerized API to become reachable."""
    last_error = None
    for _ in range(20):
        try:
            response = requests.post(LOGIN_URL, json=INVALID_CREDENTIALS, timeout=2)
            if response.status_code in (200, 401, 400, 422):
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)

    pytest.fail(
        "Service is not reachable. Start the container first (example: `make run`). "
        f"Last error: {last_error}"
    )


@pytest.fixture(scope="session")
def token() -> str:
    """Retrieve a valid JWT token from the login endpoint."""
    response = requests.post(LOGIN_URL, json=VALID_CREDENTIALS, timeout=5)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "token" in body
    return body["token"]


@pytest.fixture(scope="session")
def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ==========
# 1. Login API Tests
# ==========


def test_login_success():
    """Verify API returns a valid JWT token for correct credentials."""
    response = requests.post(LOGIN_URL, json=VALID_CREDENTIALS, timeout=5)
    assert response.status_code == 200
    assert "token" in response.json()


def test_login_invalid_credentials():
    """Verify API returns 401 for incorrect credentials."""
    response = requests.post(LOGIN_URL, json=INVALID_CREDENTIALS, timeout=5)
    assert response.status_code == 401


# ==========
# 2. JWT Authentication Tests
# ==========


def test_predict_auth_missing_token():
    """Verify 401 if JWT token is missing."""
    response = requests.post(PREDICT_URL, json=VALID_INPUT, timeout=5)
    assert response.status_code == 401


def test_predict_auth_invalid_token():
    """Verify 401 if JWT token is malformed or invalid."""
    headers = {"Authorization": "Bearer not.a.valid.jwt"}
    response = requests.post(PREDICT_URL, headers=headers, json=VALID_INPUT, timeout=5)
    assert response.status_code == 401


def test_predict_auth_expired_token():
    """Verify 401 if JWT token has expired."""
    exp = datetime.now(timezone.utc) - timedelta(hours=1)
    token = jwt.encode({"sub": "admin", "exp": exp}, JWT_SECRET, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(PREDICT_URL, headers=headers, json=VALID_INPUT, timeout=5)
    assert response.status_code == 401



def test_predict_auth_valid_token(auth_headers: dict):
    """Verify authenmtication succeeds with a valid JWT token."""
    response = requests.post(PREDICT_URL, headers=auth_headers, json=VALID_INPUT, timeout=5)
    assert response.status_code == 200
    assert "prediction" in response.json()


# ==========
# 3. Prediction API Tests
# ==========

def test_predict_success(auth_headers: dict):
    """Verify valid prediction for correct input data."""
    response = requests.post(PREDICT_URL, headers=auth_headers, json=VALID_INPUT, timeout=5)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_invalid_input_data(auth_headers: dict):
    """Verify API returns an error for invalid input data."""
    response = requests.post(
        PREDICT_URL,
        headers=auth_headers,
        json=INVALID_INPUT_MISSING_FIELD,
        timeout=5,
    )
    assert response.status_code in (400, 422)
