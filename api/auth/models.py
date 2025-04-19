#!/usr/bin/env python3
# api/auth/models.py

"""
Data models for authentication and authorization.
"""

from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import re
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Role(BaseModel):
    """Role definition for authorization"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Name of the role")
    description: Optional[str] = Field(None, description="Description of the role")
    permissions: List[str] = Field(
        default_factory=list,
        description="List of permission scopes granted by this role",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        schema_extra = {
            "example": {
                "name": "admin",
                "description": "Administrator role with full access",
                "permissions": ["read:all", "write:all", "delete:all"],
            }
        }


class UserCreate(BaseModel):
    """Schema for user registration"""

    username: str = Field(
        ..., min_length=3, max_length=50, description="Username for login"
    )
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password")
    full_name: Optional[str] = Field(None, description="User's full name")

    @validator("username")
    def username_alphanumeric(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username must be alphanumeric with only underscores and hyphens"
            )
        return v

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v

    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john.doe@example.com",
                "password": "SecurePass123",
                "full_name": "John Doe",
            }
        }


class User(BaseModel):
    """User model for database storage and internal use"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    hashed_password: str
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = Field(
        default_factory=list, description="List of role IDs assigned to this user"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict, description="User preferences"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    @classmethod
    def create(cls, user_create: UserCreate):
        """Create a new user from registration data"""
        return cls(
            username=user_create.username,
            email=user_create.email,
            hashed_password=pwd_context.hash(user_create.password),
            full_name=user_create.full_name,
            roles=["basic_user"],  # Default role
        )

    def verify_password(self, plain_password: str) -> bool:
        """Verify a password against the hashed password"""
        return pwd_context.verify(plain_password, self.hashed_password)

    def has_permission(self, required_permission: str) -> bool:
        """Check if user has a specific permission through their roles"""
        # This would typically check against a roles database
        # For now, we'll use a simplified approach
        if "admin" in self.roles:
            return True  # Admin has all permissions

        # In a real implementation, we would look up all roles and their permissions
        basic_permissions = ["read:own", "write:own"]
        if required_permission in basic_permissions and "basic_user" in self.roles:
            return True

        return False


class UserResponse(BaseModel):
    """User information for API responses"""

    id: str
    username: str
    email: EmailStr
    full_name: Optional[str]
    roles: List[str]
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "johndoe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "roles": ["basic_user"],
                "created_at": "2023-01-01T00:00:00Z",
                "last_login": "2023-01-02T12:34:56Z",
            }
        }


class TokenResponse(BaseModel):
    """Token response for login"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "johndoe",
            }
        }
