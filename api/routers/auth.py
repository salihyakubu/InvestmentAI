"""Authentication endpoints: issue JWT access tokens.

The rest of the API requires a JWT (``get_current_user``); this is the endpoint
that issues one in exchange for valid user credentials.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from api.dependencies import get_db, get_settings
from api.middleware.auth import JWTAuth, hash_password, verify_password
from config.settings import Settings
from core.models.users import User

router = APIRouter(prefix="/auth")

# Precomputed hash used to keep timing uniform for unknown users (mitigates
# user-enumeration via response timing).
_DUMMY_HASH = hash_password("invalid-placeholder-password")


class LoginRequest(BaseModel):
    """Credentials for obtaining an access token."""

    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    """An issued bearer token."""

    access_token: str
    token_type: str = "bearer"


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Obtain a JWT access token",
)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Verify credentials and return a signed JWT access token."""
    # noload(api_keys): login only needs the user's own columns, and skipping the
    # eager relationship keeps the query to a single table.
    result = await db.execute(
        select(User).where(User.email == payload.email).options(noload(User.api_keys))
    )
    user = result.scalar_one_or_none()

    # Always run a hash verification (against a dummy when the user is unknown)
    # so the response time does not reveal whether the email exists.
    hashed = user.hashed_password if user is not None else _DUMMY_HASH
    password_ok = verify_password(payload.password, hashed)

    if user is None or not user.is_active or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jwt_auth = JWTAuth(
        secret=settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
        expire_minutes=settings.jwt_expire_minutes,
    )
    token = jwt_auth.create_access_token(user_id=str(user.id), role=user.role)
    return TokenResponse(access_token=token)
