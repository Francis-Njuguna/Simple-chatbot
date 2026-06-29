"""Authentication service."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import User
from backend.app.models.schemas import LoginRequest, RegisterRequest, TokenResponse
from backend.app.utils.exceptions import AuthenticationError
from backend.app.utils.security import create_access_token, hash_password, verify_password


class AuthService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def register(self, request: RegisterRequest) -> TokenResponse:
        result = await self.db.execute(select(User).where(User.email == request.email))
        if result.scalar_one_or_none():
            raise AuthenticationError("Email already registered")

        user = User(
            email=request.email,
            password_hash=hash_password(request.password),
            full_name=request.full_name,
        )
        self.db.add(user)
        await self.db.flush()

        token = create_access_token({"sub": str(user.id), "email": user.email})
        return TokenResponse(access_token=token)

    async def login(self, request: LoginRequest) -> TokenResponse:
        result = await self.db.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user or not verify_password(request.password, user.password_hash):
            raise AuthenticationError("Invalid email or password")

        token = create_access_token({"sub": str(user.id), "email": user.email})
        return TokenResponse(access_token=token)

    async def get_user(self, user_id: uuid.UUID) -> User | None:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
