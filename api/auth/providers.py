#!/usr/bin/env python3
# api/auth/providers.py

"""
Authentication providers for deep-recall.

This module provides different authentication mechanisms including:
- JWT-based authentication
- API key-based authentication
- OAuth2 integration (extensible)
"""

import os
import time
import secrets
from typing import Dict, Optional, List, Tuple, Any, Union
import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel
from loguru import logger

from api.auth.models import User, TokenResponse


# Security scheme definitions
jwt_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

# Environment variables and defaults
JWT_SECRET = os.getenv(
    "JWT_SECRET", "development_secret_key"
)  # Use env vars in production
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 30))


class TokenData(BaseModel):
    """Data extracted from a JWT token"""

    user_id: str
    username: str
    scopes: List[str] = []
    exp: int


# JWT Authentication Provider
class JWTAuth:
    """JWT-based authentication provider"""

    @staticmethod
    def create_access_token(
        user: User, expires_delta: Optional[timedelta] = None
    ) -> Tuple[str, int]:
        """
        Create a new JWT access token

        Args:
            user: User to create token for
            expires_delta: Optional expiration time override

        Returns:
            Tuple of (token string, expiration in seconds)
        """
        expire_delta = expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        expire_time = datetime.utcnow() + expire_delta
        expire_timestamp = int(expire_time.timestamp())

        # Get permissions from user roles (in production, would fetch from role DB)
        permissions = []
        if "admin" in user.roles:
            permissions = ["read:all", "write:all", "delete:all"]
        elif "basic_user" in user.roles:
            permissions = ["read:own", "write:own"]

        # Create token payload
        payload = {
            "sub": user.id,
            "username": user.username,
            "scopes": permissions,
            "exp": expire_timestamp,
            "iat": int(datetime.utcnow().timestamp()),
        }

        # Encode the token
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        return token, expire_delta.seconds

    @staticmethod
    def create_refresh_token(user: User) -> str:
        """
        Create a long-lived refresh token for refreshing access tokens

        Args:
            user: User to create token for

        Returns:
            Token string
        """
        expire_time = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        payload = {
            "sub": user.id,
            "type": "refresh",
            "exp": int(expire_time.timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
        }

        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Optional[TokenData]:
        """
        Verify and decode a JWT token

        Args:
            token: JWT token string

        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

            if "sub" not in payload or "exp" not in payload:
                logger.warning("JWT missing required fields")
                return None

            # Create token data from payload
            token_data = TokenData(
                user_id=payload["sub"],
                username=payload.get("username", ""),
                scopes=payload.get("scopes", []),
                exp=payload["exp"],
            )

            return token_data
        except jwt.ExpiredSignatureError:
            logger.warning(f"Expired token attempt")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    @staticmethod
    def refresh_access_token(
        refresh_token: str, user_db: Dict[str, User]
    ) -> Optional[Tuple[str, int]]:
        """
        Generate a new access token using a refresh token

        Args:
            refresh_token: Valid refresh token
            user_db: User database to verify user still exists and is enabled

        Returns:
            Tuple of (new access token, expiration in seconds) if valid
        """
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

            # Verify it's a refresh token
            if payload.get("type") != "refresh":
                logger.warning("Attempted to use non-refresh token for refresh")
                return None

            user_id = payload.get("sub")
            if not user_id or user_id not in user_db:
                logger.warning(f"Refresh token with unknown user ID: {user_id}")
                return None

            user = user_db[user_id]

            # Check if user is still active
            if user.disabled:
                logger.warning(f"Refresh attempt for disabled user: {user_id}")
                return None

            # Generate new access token
            return JWTAuth.create_access_token(user)

        except jwt.ExpiredSignatureError:
            logger.warning("Expired refresh token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None


# API Key Authentication Provider
class APIKeyAuth:
    """API key-based authentication"""

    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure random API key"""
        return secrets.token_urlsafe(32)

    @staticmethod
    def verify_api_key(
        api_key: str, api_keys_db: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """
        Verify an API key and return the associated user ID if valid

        Args:
            api_key: The API key to verify
            api_keys_db: Database of API keys mapping to user IDs

        Returns:
            User ID if valid, None otherwise
        """
        if api_key in api_keys_db:
            key_data = api_keys_db[api_key]

            # Check if key is expired
            if "expires_at" in key_data and datetime.utcnow() > key_data["expires_at"]:
                logger.warning("Expired API key used")
                return None

            # Check if key is disabled
            if key_data.get("disabled", False):
                logger.warning("Disabled API key used")
                return None

            return key_data.get("user_id")

        return None


# Get current user from authentication token or API key
async def get_current_user(
    token: Optional[HTTPAuthorizationCredentials] = Depends(jwt_scheme),
    api_key: Optional[str] = Depends(api_key_scheme),
    user_db: Dict[str, User] = None,  # Would be injected in production
    api_keys_db: Dict[str, Dict[str, Any]] = None,  # Would be injected in production
) -> User:
    """
    Get the current authenticated user from either JWT token or API key

    This is used as a FastAPI dependency to protect routes

    Args:
        token: HTTP Bearer token (JWT)
        api_key: API key from header
        user_db: User database (injected in production)
        api_keys_db: API keys database (injected in production)

    Returns:
        User object if authentication is valid

    Raises:
        HTTPException: If authentication is invalid or missing
    """
    # Mock databases for demonstration
    if user_db is None:
        # This would be replaced with a real database in production
        from api.auth.models import User, UserCreate

        mock_user = UserCreate(
            username="demo",
            email="demo@example.com",
            password="Demo1234",
            full_name="Demo User",
        )
        user_db = {"user123": User.create(mock_user)}
        user_db["user123"].id = "user123"

    if api_keys_db is None:
        # This would be replaced with a real database in production
        api_keys_db = {
            "demo_api_key_12345": {
                "user_id": "user123",
                "name": "Demo Key",
                "created_at": datetime.utcnow(),
                "disabled": False,
            }
        }

    # Check JWT token first
    if token and token.credentials:
        token_data = JWTAuth.verify_token(token.credentials)
        if token_data and token_data.user_id in user_db:
            user = user_db[token_data.user_id]
            if not user.disabled:
                return user

    # Then check API key
    if api_key:
        user_id = APIKeyAuth.verify_api_key(api_key, api_keys_db)
        if user_id and user_id in user_db:
            user = user_db[user_id]
            if not user.disabled:
                return user

    # If we get here, authentication failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Check for specific permission
def require_permission(required_permission: str):
    """
    Dependency factory for requiring a specific permission

    Usage:
        @router.get("/protected", dependencies=[Depends(require_permission("read:data"))])

    Args:
        required_permission: The permission scope required

    Returns:
        Dependency function that checks for the permission
    """

    async def check_permission(user: User = Depends(get_current_user)) -> User:
        if user.has_permission(required_permission):
            return user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions: {required_permission} required",
        )

    return check_permission
