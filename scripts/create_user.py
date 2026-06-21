"""Create an API login user (there is no self-service registration).

    PYTHONPATH=. python scripts/create_user.py <email> <password> [role]

Uses settings.DATABASE_URL. Roles: viewer (default) / trader / admin.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

from sqlalchemy import select

from api.middleware.auth import hash_password
from core.models.base import get_async_session_factory
from core.models.users import User


async def create_user(email: str, password: str, role: str = "viewer") -> int:
    factory = get_async_session_factory()
    async with factory() as session:
        existing = (
            await session.execute(select(User).where(User.email == email))
        ).scalar_one_or_none()
        if existing is not None:
            print(f"user already exists: {email}")
            return 1
        session.add(
            User(
                email=email,
                hashed_password=hash_password(password),
                role=role,
                is_active=True,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()
    print(f"created user {email} (role={role})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: create_user.py <email> <password> [role]")
        sys.exit(2)
    sys.exit(asyncio.run(create_user(*sys.argv[1:])))
