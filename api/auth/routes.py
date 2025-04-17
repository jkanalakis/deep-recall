#!/usr/bin/env python3
# api/auth/routes.py

"""
API routes for authentication and user management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Body, Query
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, EmailStr

from api.auth.models import User, UserCreate, UserResponse, TokenResponse
from api.auth.providers import (
    get_current_user, 
    require_permission, 
    JWTAuth,
    TokenData
)
from api.auth.repository import user_repository


# Create router
router = APIRouter()


class LoginRequest(BaseModel):
    """Login request schema"""
    username: str
    password: str


class ApiKeyRequest(BaseModel):
    """API key request schema"""
    name: str = Field(..., description="Name for the API key")
    expires_days: Optional[int] = Field(
        None, 
        description="Number of days until the API key expires (null for no expiration)",
        ge=1
    )


class ApiKeyResponse(BaseModel):
    """API key response schema"""
    key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None


class UserUpdateRequest(BaseModel):
    """User update request schema"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    disabled: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None


class RoleAssignmentRequest(BaseModel):
    """Role assignment request schema"""
    role_id: str


# Authentication endpoints
@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate a user with username and password
    
    Args:
        form_data: Form with username and password
        
    Returns:
        Access token if authentication successful
        
    Raises:
        HTTPException: If authentication fails
    """
    # Find user by username
    user = user_repository.get_user_by_username(form_data.username)
    
    # Check if user exists and password is correct
    if not user or not user.verify_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Check if user is disabled
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Update last login time
    user.last_login = datetime.utcnow()
    
    # Generate access token
    token, expires_in = JWTAuth.create_access_token(user)
    
    # Return token response
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=user.id,
        username=user.username
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str = Body(..., embed=True)):
    """
    Refresh an access token using a refresh token
    
    Args:
        refresh_token: Valid refresh token
        
    Returns:
        New access token
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    # Get user database (in production, this would be injected)
    user_db = {user.id: user for user in user_repository.users.values()}
    
    # Try to refresh the token
    result = JWTAuth.refresh_access_token(refresh_token, user_db)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    token, expires_in = result
    user_id = JWTAuth.verify_token(token).user_id
    user = user_repository.get_user_by_id(user_id)
    
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=user.id,
        username=user.username
    )


# User registration and management
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """
    Register a new user
    
    Args:
        user_data: User registration data
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: If username or email already exists
    """
    try:
        # Create the user
        user = user_repository.create_user(user_data)
        
        # Convert to response model
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            roles=[role_name for role_id in user.roles 
                  for role_name in [user_repository.roles.get(role_id, Role(id="", name="unknown")).name]],
            created_at=user.created_at,
            last_login=user.last_login
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get information about the currently authenticated user
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    # Convert role IDs to names
    role_names = []
    for role_id in current_user.roles:
        role = user_repository.roles.get(role_id)
        if role:
            role_names.append(role.name)
        else:
            role_names.append("unknown")
    
    # Return user response
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        roles=role_names,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update the current user's information
    
    Args:
        update_data: Fields to update
        current_user: Current authenticated user
        
    Returns:
        Updated user information
    """
    # Convert to dict and remove None values
    updates = update_data.dict(exclude_unset=True, exclude_none=True)
    
    # Apply updates
    updated_user = user_repository.update_user(current_user.id, updates)
    
    # Convert role IDs to names for response
    role_names = []
    for role_id in updated_user.roles:
        role = user_repository.roles.get(role_id)
        if role:
            role_names.append(role.name)
        else:
            role_names.append("unknown")
    
    # Return updated user
    return UserResponse(
        id=updated_user.id,
        username=updated_user.username,
        email=updated_user.email,
        full_name=updated_user.full_name,
        roles=role_names,
        created_at=updated_user.created_at,
        last_login=updated_user.last_login
    )


# API Key management
@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    key_request: ApiKeyRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new API key for the current user
    
    Args:
        key_request: API key request data
        current_user: Current authenticated user
        
    Returns:
        Created API key
    """
    expires_at = None
    if key_request.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=key_request.expires_days)
    
    api_key = user_repository.create_api_key(
        user_id=current_user.id,
        name=key_request.name,
        expires_at=expires_at
    )
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )
    
    return ApiKeyResponse(
        key=api_key,
        name=key_request.name,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )


@router.get("/api-keys", response_model=List[Dict[str, Any]])
async def list_api_keys(current_user: User = Depends(get_current_user)):
    """
    List all API keys for the current user
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of API keys
    """
    return user_repository.get_user_api_keys(current_user.id)


@router.delete("/api-keys/{key_prefix}")
async def disable_api_key(
    key_prefix: str = Path(..., min_length=8, description="Prefix of the API key to disable"),
    current_user: User = Depends(get_current_user)
):
    """
    Disable an API key
    
    Args:
        key_prefix: Prefix of the API key to disable
        current_user: Current authenticated user
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If key not found or doesn't belong to user
    """
    # Find keys matching the prefix and belonging to the user
    matching_keys = []
    for key, data in user_repository.api_keys.items():
        if key.startswith(key_prefix) and data.get("user_id") == current_user.id:
            matching_keys.append(key)
    
    if not matching_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No API keys found with prefix '{key_prefix}'"
        )
    
    # Disable all matching keys
    for key in matching_keys:
        user_repository.disable_api_key(key)
    
    return {"detail": f"Disabled {len(matching_keys)} API key(s)"}


# User management (admin only)
@router.get("/users", response_model=List[UserResponse], dependencies=[Depends(require_permission("user:manage"))])
async def list_users():
    """
    List all users (admin only)
    
    Returns:
        List of all users
    """
    responses = []
    for user in user_repository.users.values():
        # Convert role IDs to names
        role_names = []
        for role_id in user.roles:
            role = user_repository.roles.get(role_id)
            if role:
                role_names.append(role.name)
            else:
                role_names.append("unknown")
        
        responses.append(UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            roles=role_names,
            created_at=user.created_at,
            last_login=user.last_login
        ))
    
    return responses


@router.get("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(require_permission("user:manage"))])
async def get_user(user_id: str):
    """
    Get a user by ID (admin only)
    
    Args:
        user_id: ID of user to get
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user not found
    """
    user = user_repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    # Convert role IDs to names
    role_names = []
    for role_id in user.roles:
        role = user_repository.roles.get(role_id)
        if role:
            role_names.append(role.name)
        else:
            role_names.append("unknown")
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        roles=role_names,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.put("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(require_permission("user:manage"))])
async def update_user(user_id: str, update_data: UserUpdateRequest):
    """
    Update a user (admin only)
    
    Args:
        user_id: ID of user to update
        update_data: Fields to update
        
    Returns:
        Updated user information
        
    Raises:
        HTTPException: If user not found
    """
    # Convert to dict and remove None values
    updates = update_data.dict(exclude_unset=True, exclude_none=True)
    
    # Apply updates
    updated_user = user_repository.update_user(user_id, updates)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    # Convert role IDs to names for response
    role_names = []
    for role_id in updated_user.roles:
        role = user_repository.roles.get(role_id)
        if role:
            role_names.append(role.name)
        else:
            role_names.append("unknown")
    
    return UserResponse(
        id=updated_user.id,
        username=updated_user.username,
        email=updated_user.email,
        full_name=updated_user.full_name,
        roles=role_names,
        created_at=updated_user.created_at,
        last_login=updated_user.last_login
    )


@router.delete("/users/{user_id}", dependencies=[Depends(require_permission("user:manage"))])
async def delete_user(user_id: str):
    """
    Delete a user (admin only)
    
    Args:
        user_id: ID of user to delete
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If user not found
    """
    success = user_repository.delete_user(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    return {"detail": f"User {user_id} deleted successfully"}


@router.post(
    "/users/{user_id}/roles", 
    dependencies=[Depends(require_permission("user:manage"))],
    status_code=status.HTTP_201_CREATED
)
async def add_role_to_user(user_id: str, role_request: RoleAssignmentRequest):
    """
    Add a role to a user (admin only)
    
    Args:
        user_id: ID of user to update
        role_request: Role assignment request
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If user or role not found
    """
    success = user_repository.add_role_to_user(user_id, role_request.role_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} or role with ID {role_request.role_id} not found"
        )
    
    return {"detail": f"Role {role_request.role_id} added to user {user_id}"}


@router.delete(
    "/users/{user_id}/roles/{role_id}", 
    dependencies=[Depends(require_permission("user:manage"))]
)
async def remove_role_from_user(user_id: str, role_id: str):
    """
    Remove a role from a user (admin only)
    
    Args:
        user_id: ID of user to update
        role_id: ID of role to remove
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If user not found or had no such role
    """
    success = user_repository.remove_role_from_user(user_id, role_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found or had no role with ID {role_id}"
        )
    
    return {"detail": f"Role {role_id} removed from user {user_id}"} 