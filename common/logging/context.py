"""
Context utilities for managing request context and correlation IDs
"""

import asyncio
import contextvars
import uuid
from typing import Optional

# Context variables for request context
request_id_var = contextvars.ContextVar("request_id", default=None)
user_id_var = contextvars.ContextVar("user_id", default=None)
correlation_id_var = contextvars.ContextVar("correlation_id", default=None)


def get_request_id() -> str:
    """Get the current request ID or generate a new one"""
    request_id = request_id_var.get()
    if request_id is None:
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
    return request_id


def set_request_id(request_id: str) -> None:
    """Set the current request ID"""
    request_id_var.set(request_id)


def get_user_id() -> Optional[str]:
    """Get the current user ID"""
    return user_id_var.get()


def set_user_id(user_id: str) -> None:
    """Set the current user ID"""
    user_id_var.set(user_id)


def get_correlation_id() -> str:
    """Get the current correlation ID or generate a new one"""
    correlation_id = correlation_id_var.get()
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the current correlation ID"""
    correlation_id_var.set(correlation_id)


class RequestContextMiddleware:
    """
    Middleware for capturing request context variables
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Extract headers from scope
        headers = dict(scope.get("headers", []))

        # Look for request ID in headers
        request_id = None
        if b"x-request-id" in headers:
            request_id = headers[b"x-request-id"].decode("utf-8")

        # Look for correlation ID in headers
        correlation_id = None
        if b"x-correlation-id" in headers:
            correlation_id = headers[b"x-correlation-id"].decode("utf-8")

        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())

        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set context vars
        request_id_var.set(request_id)
        correlation_id_var.set(correlation_id)

        # Process the request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add headers to the response
                headers = message.get("headers", [])
                headers.append((b"x-request-id", request_id.encode("utf-8")))
                headers.append((b"x-correlation-id", correlation_id.encode("utf-8")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)
