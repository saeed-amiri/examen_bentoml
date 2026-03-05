import os
import time
import datetime as dt
import jwt
import pytest
import requests

# ========================
# Configuration
# ========================
BASE_URL = os.getenv("ADMISSION_BASE_URL", "http://127.0.0.1:3000")

LOGIN_URL = f"{BASE_URL}/login"
PREDICT_URL = f"{BASE_URL}/v1/models/admission_lr/predict"

# Use a longer key to avoid InsecureKeyLengthWarning (min 32 chars for HS256)
JWT_SECRET = os.getenv("TEST_JWT_SECRET", "my_super_secret_key_that_is_at_least_32_characters_long")
JWT_ALGORITHM = "HS256"

# ========================
# Test data - FIXED STRUCTURES
# ========================

# Wrap in "credentials" key because service argument name is 'credentials'
VALID_CREDENTIALS = {
    "credentials": {
        "username": "admin",
        "password": "password",
    }
}

INVALID_CREDENTIALS = {
    "credentials": {
        "username": "wrong",
        "password": "wrong",
    }
}

# Wrap in "data" key because service argument name is 'data'
VALID_INPUT = {
    "data": {
        "gre_score": 320,
        "toefl_score": 110,
        "university_rating": 4,
        "sop": 4.0,
        "lor": 4.0,
        "cgpa": 9.0,
        "research": 1,
    }
}

INVALID_INPUT_MISSING_FIELD = {
    "data": {
        "toefl_score": 110,
        "university_rating": 4,
        "sop": 4.0,
        "lor": 4.0,
        "cgpa": 9.0,
        "research": 1,
    }
}

# ========================
# Fixtures
# ========================
@pytest.fixture(scope="session")
def token() -> str:
    """Retrieve a valid JWT token from the login endpoint."""
    for _ in range(5):
        try:
            response = requests.post(LOGIN_URL, json=VALID_CREDENTIALS, timeout=3)
            if response.status_code == 200:
                return response.json()["token"]
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    pytest.fail("Unable to retrieve JWT token from /login endpoint")

@pytest.fixture(scope="session")
def headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

# ========================
# Tests
# ========================

def test_login_success():
    response = requests.post(LOGIN_URL, json=VALID_CREDENTIALS)
    assert response.status_code == 200
    assert "token" in response.json()

def test_login_failure():
    response = requests.post(LOGIN_URL, json=INVALID_CREDENTIALS)
    assert response.status_code == 401

def test_auth_valid_token(headers: dict):
    response = requests.post(PREDICT_URL, json=VALID_INPUT, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_payload(headers: dict):
    # Missing 'gre_score' inside 'data'
    response = requests.post(PREDICT_URL, json=INVALID_INPUT_MISSING_FIELD, headers=headers)
    assert response.status_code in (400, 422)

def test_auth_missing_token():
    response = requests.post(PREDICT_URL, json=VALID_INPUT)
    assert response.status_code == 401