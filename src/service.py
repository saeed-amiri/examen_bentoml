import logging
import warnings
from datetime import datetime, timedelta, timezone

import jwt
import numpy as np
import pandas as pd
import bentoml
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ==========
# Warning configuration
# ==========
warnings.filterwarnings(
    "ignore",
    message="`bentoml.io` is deprecated",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message="`bentoml.Service` is deprecated",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=Warning,
)

# ==========
# Logging
# ==========
logging.basicConfig(level=logging.INFO)

# ==========
# JWT configuration
# ==========
JWT_SECRET_KEY = "my_super_secret_key_that_is_at_least_32_characters_long"
JWT_ALGORITHM = "HS256"

USERS = {
    "admin": "password",
    "user123": "password123",
}

# ==========
# Training columns
# ==========
TRAIN_COLS = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
]

# ==========
# JWT Middleware
# ==========
class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware verifying JWT token for protected routes."""

    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/admission_lr/predict":

            logging.info("JWT middleware triggered on %s", request.url.path)

            token = request.headers.get("Authorization")

            if not token:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing authentication token"},
                )

            try:
                token = token.split()[1]
                payload = jwt.decode(
                    token,
                    JWT_SECRET_KEY,
                    algorithms=[JWT_ALGORITHM],
                )

            except jwt.ExpiredSignatureError:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Token expired"},
                )

            except jwt.InvalidTokenError:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid token"},
                )

            request.state.user = payload.get("sub")

        return await call_next(request)


# ==========
# JWT helper
# ==========
def create_jwt_token(user_id: str) -> str:
    """Create a JWT authentication token."""
    expiration = datetime.now(timezone.utc) + timedelta(hours=1)

    payload = {
        "sub": user_id,
        "exp": expiration,
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


# ==========
# Pydantic models
# ==========
class AdmissionInput(BaseModel):
    """Input schema for prediction."""

    gre_score: int = Field(..., description="GRE score (0–340)")
    toefl_score: int = Field(..., description="TOEFL score (0–120)")
    university_rating: int = Field(..., description="University rating (1–5)")
    sop: float = Field(..., description="Statement of Purpose strength (0–5)")
    lor: float = Field(..., description="Letter of Recommendation strength (0–5)")
    cgpa: float = Field(..., description="Cumulative GPA (0–10)")
    research: int = Field(..., description="Research experience (0=no, 1=yes)")


class LoginInput(BaseModel):
    """Input schema for authentication."""

    username: str = Field(..., description="Username for authentication")
    password: str = Field(..., description="Password for authentication")


# ==========
# BentoML Service
# ==========
@bentoml.service
class AdmissionService:
    def __init__(self):
        # Load model
        model_ref = bentoml.models.get("admission_lr:latest")
        self.sklearn_model = bentoml.sklearn.load_model(model_ref)

    # ==========
    # Login endpoint
    # ==========
    @bentoml.api(route="/login")
    async def login(self, credentials: LoginInput, ctx: bentoml.Context) -> dict:
        """
        Authenticate user and return a JWT token.
        
        Note: By typing 'credentials' as LoginInput, BentoML automatically 
        validates the request body against the Pydantic model.
        """
        if USERS.get(credentials.username) != credentials.password:
            ctx.response.status_code = 401
            return {"detail": "Invalid credentials"}

        token = create_jwt_token(credentials.username)
        return {"token": token}

    # ==========
    # Prediction endpoint
    # ==========
    @bentoml.api(route="/v1/models/admission_lr/predict")
    async def predict(self, data: AdmissionInput, ctx: bentoml.Context) -> dict:
        """
        Predict admission probability.
        
        Note: By typing 'data' as AdmissionInput, BentoML automatically 
        validates the request body.
        """
        # Accessing fields directly from the Pydantic object
        df = pd.DataFrame(
            [
                {
                    "gre_score": data.gre_score,
                    "toefl_score": data.toefl_score,
                    "university_rating": data.university_rating,
                    "sop": data.sop,
                    "lor": data.lor,
                    "cgpa": data.cgpa,
                    "research": data.research,
                }
            ],
            columns=TRAIN_COLS,
        )

        pred = self.sklearn_model.predict(df)
        return {"prediction": np.asarray(pred).reshape(-1).astype(float).tolist()}


# ==========
# Configure Service Middleware & Alias
# ==========

# Add the middleware to the Service class
AdmissionService.add_asgi_middleware(JWTAuthMiddleware)

# Alias for backward compatibility with bentoml.yml / Makefile
admission_service = AdmissionService