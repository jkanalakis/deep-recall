#!/usr/bin/env python3
# api/auth/repository.py

"""
User repository for managing user data.

This module provides a simple in-memory user repository for development.
In production, this would be replaced with a database-backed implementation.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import uuid
from loguru import logger

from api.auth.models import User, UserCreate, Role


class UserRepository:
    """
    User data repository
    
    This is a simple in-memory implementation for development.
    In production, this would interface with a database.
    """
    
    def __init__(self):
        """Initialize the repository with default users and roles"""
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Create default roles
        self._create_default_roles()
    
    def _create_default_roles(self):
        """Create the default roles in the system"""
        # Admin role
        admin_role = Role(
            id="role_admin",
            name="admin",
            description="Administrator with full system access",
            permissions=[
                "read:all", 
                "write:all", 
                "delete:all",
                "user:manage"
            ]
        )
        
        # Basic user role
        basic_role = Role(
            id="role_basic_user",
            name="basic_user",
            description="Standard user with limited access",
            permissions=[
                "read:own",
                "write:own",
                "memory:read",
                "memory:write",
                "inference:generate"
            ]
        )
        
        # Store roles
        self.roles[admin_role.id] = admin_role
        self.roles[basic_role.id] = basic_role
        
        logger.info(f"Created default roles: {', '.join(r.name for r in self.roles.values())}")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user
        
        Args:
            user_data: User creation data
            
        Returns:
            Created user
            
        Raises:
            ValueError: If username or email already exists
        """
        # Check if username exists
        if self.get_user_by_username(user_data.username):
            raise ValueError(f"Username '{user_data.username}' already exists")
            
        # Check if email exists
        if self.get_user_by_email(user_data.email):
            raise ValueError(f"Email '{user_data.email}' already exists")
            
        # Create user with basic_user role by default
        user = User.create(user_data)
        user.roles = ["role_basic_user"]  # Assign default role by ID
        
        # Store user
        self.users[user.id] = user
        logger.info(f"Created user: {user.username} ({user.id})")
        
        return user
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[User]:
        """
        Update a user's information
        
        Args:
            user_id: ID of user to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated user or None if not found
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return None
            
        # Apply updates
        for key, value in updates.items():
            if hasattr(user, key) and key != "id" and key != "hashed_password":
                setattr(user, key, value)
                
        # Set updated timestamp
        user.updated_at = datetime.utcnow()
        
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user
        
        Args:
            user_id: ID of user to delete
            
        Returns:
            True if user was deleted, False if not found
        """
        if user_id in self.users:
            del self.users[user_id]
            
            # Also delete any associated API keys
            keys_to_delete = []
            for key, data in self.api_keys.items():
                if data.get("user_id") == user_id:
                    keys_to_delete.append(key)
                    
            for key in keys_to_delete:
                del self.api_keys[key]
                
            logger.info(f"Deleted user: {user_id}")
            return True
            
        return False
    
    def create_api_key(self, user_id: str, name: str, expires_at: Optional[datetime] = None) -> Optional[str]:
        """
        Create a new API key for a user
        
        Args:
            user_id: ID of user to create key for
            name: Name of the API key
            expires_at: Optional expiration date
            
        Returns:
            Generated API key or None if user not found
        """
        if user_id not in self.users:
            return None
            
        # Generate a unique API key
        from api.auth.providers import APIKeyAuth
        api_key = APIKeyAuth.generate_api_key()
        
        # Store the key
        self.api_keys[api_key] = {
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "disabled": False
        }
        
        logger.info(f"Created API key '{name}' for user {user_id}")
        return api_key
    
    def get_user_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all API keys for a user
        
        Args:
            user_id: ID of user to get keys for
            
        Returns:
            List of API key data (without the actual key for security)
        """
        results = []
        for key, data in self.api_keys.items():
            if data.get("user_id") == user_id:
                # Create a copy without exposing the full key
                masked_key = f"{key[:8]}..."
                key_data = data.copy()
                key_data["key"] = masked_key
                results.append(key_data)
                
        return results
    
    def disable_api_key(self, api_key: str) -> bool:
        """
        Disable an API key
        
        Args:
            api_key: The API key to disable
            
        Returns:
            True if key was disabled, False if not found
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]["disabled"] = True
            logger.info(f"Disabled API key: {api_key[:8]}...")
            return True
            
        return False
    
    def get_role_permissions(self, role_id: str) -> List[str]:
        """
        Get permissions for a role
        
        Args:
            role_id: ID of role to get permissions for
            
        Returns:
            List of permission strings
        """
        role = self.roles.get(role_id)
        if role:
            return role.permissions
        return []
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Get all permissions for a user based on their roles
        
        Args:
            user_id: ID of user to get permissions for
            
        Returns:
            List of all permission strings the user has
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return []
            
        # Get permissions from all roles
        all_permissions = set()
        for role_id in user.roles:
            role_permissions = self.get_role_permissions(role_id)
            all_permissions.update(role_permissions)
            
        return list(all_permissions)
            
    def add_role_to_user(self, user_id: str, role_id: str) -> bool:
        """
        Add a role to a user
        
        Args:
            user_id: ID of user to add role to
            role_id: ID of role to add
            
        Returns:
            True if role was added, False if user or role not found
        """
        user = self.get_user_by_id(user_id)
        if not user or role_id not in self.roles:
            return False
            
        if role_id not in user.roles:
            user.roles.append(role_id)
            user.updated_at = datetime.utcnow()
            logger.info(f"Added role {role_id} to user {user_id}")
            return True
            
        return False  # Role already assigned
    
    def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """
        Remove a role from a user
        
        Args:
            user_id: ID of user to remove role from
            role_id: ID of role to remove
            
        Returns:
            True if role was removed, False if user not found or had no such role
        """
        user = self.get_user_by_id(user_id)
        if not user or role_id not in user.roles:
            return False
            
        user.roles.remove(role_id)
        user.updated_at = datetime.utcnow()
        logger.info(f"Removed role {role_id} from user {user_id}")
        return True


# Create a singleton instance for global access
user_repository = UserRepository() 