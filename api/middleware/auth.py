#!/usr/bin/env python3
# api/middleware/auth.py

import os
import time
from typing import Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel

# JWT settings (should be in environment variables or config)
JWT_SECRET = os.getenv(
    "JWT_SECRET", "development_secret_key"
)  # In production use env vars
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()


class TokenData(BaseModel):
    """Data structure for decoded token information"""

    user_id: str
    exp: int
    scopes: list[str] = []


def create_access_token(user_id: str, scopes: list[str] = None) -> str:
    """
    Generate a new JWT access token

    Args:
        user_id: Unique identifier for the user
        scopes: List of permission scopes granted to this token

    Returns:
        str: Encoded JWT token
    """
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + TOKEN_EXPIRE_MINUTES * 60,
        "scopes": scopes or [],
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token

    Args:
        token: JWT token string

    Returns:
        TokenData: Decoded token data or None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(**payload)
    except jwt.ExpiredSignatureError:
        logger.warning(f"Expired token attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


async def authenticate_request(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> TokenData:
    """
    Authenticate a request using JWT token

    Args:
        credentials: HTTP Authorization header credentials

    Returns:
        TokenData: Authenticated user data

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authorization credentials provided",
        )

    token_data = decode_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    return token_data


def check_scope(required_scope: str):
    """
    Create a dependency that checks if the token has the required scope

    Args:
        required_scope: Scope needed for the operation

    Returns:
        Callable: A dependency function that checks the scope and returns TokenData
    """
    async def scope_checker(token_data: TokenData = Depends(authenticate_request)) -> TokenData:
        if required_scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {required_scope} scope required",
            )
        return token_data
    
    return scope_checker
