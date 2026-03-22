"""Authentication middleware: JWT and API key validation."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Header scheme for API key authentication
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class JWTAuth:
    """JSON Web Token authentication helper."""

    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        expire_minutes: int = 1440,
    ) -> None:
        self._secret = secret
        self._algorithm = algorithm
        self._expire_minutes = expire_minutes

    def create_access_token(
        self,
        user_id: str,
        role: str = "viewer",
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a signed JWT access token.

        Args:
            user_id: Subject claim (typically the user UUID as a string).
            role: User role embedded in the token payload.
            extra_claims: Optional additional claims to include.

        Returns:
            Encoded JWT string.
        """
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "sub": str(user_id),
            "role": role,
            "iat": now,
            "exp": now + timedelta(minutes=self._expire_minutes),
        }
        if extra_claims:
            payload.update(extra_claims)

        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Decode and validate a JWT token.

        Returns:
            The decoded payload dict, or ``None`` if the token is invalid.
        """
        try:
            payload = jwt.decode(
                token, self._secret, algorithms=[self._algorithm]
            )
            if payload.get("sub") is None:
                return None
            return payload
        except JWTError:
            logger.debug("JWT verification failed", exc_info=True)
            return None


class APIKeyAuth:
    """Validate API keys supplied via the ``X-API-Key`` header.

    Keys are compared by SHA-256 hash against the database.
    """

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Return the SHA-256 hex digest of *raw_key*."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    @staticmethod
    async def validate(
        request: Request,
        api_key: str | None = Security(_api_key_header),
    ) -> dict[str, Any]:
        """FastAPI dependency that validates the API key header.

        Returns a dict with ``user_id`` and ``permissions`` on success.
        Raises 401 if the key is missing or not found.
        """
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key header",
            )

        key_hash = APIKeyAuth.hash_key(api_key)

        # Look up the key in the database via the app-level session factory
        from sqlalchemy import select
        from core.models.users import APIKey

        factory = request.app.state.db_session_factory
        async with factory() as session:
            stmt = select(APIKey).where(
                APIKey.key_hash == key_hash,
            )
            result = await session.execute(stmt)
            db_key = result.scalar_one_or_none()

        if db_key is None or db_key.is_expired:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )

        return {
            "user_id": db_key.user_id,
            "permissions": db_key.permissions,
        }
