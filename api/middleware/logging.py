#!/usr/bin/env python3
# api/middleware/logging.py

import json
import os
import time
import uuid

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Configure Loguru
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.remove()  # Remove default handler
logger.add(
    "logs/api_gateway.log",
    rotation="100 MB",
    retention="7 days",
    compression="zip",
    level=log_level,
)
logger.add(lambda msg: print(msg, end=""), level=log_level)  # Also log to console


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        """Process each request and log details"""
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract client info
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Start timer
        start_time = time.time()

        # Log the incoming request
        logger.info(
            f"Request {request_id} started: {request.method} {request.url.path} "
            f"from {client_host} [UA: {user_agent[:30]}...]"
        )

        # Process the request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response data
            logger.info(
                f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s"
            )

            # Add request ID header to response
            response.headers["X-Request-ID"] = request_id

            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed after {process_time:.4f}s: {str(e)}"
            )
            raise

    @staticmethod
    def sanitize_headers(headers):
        """Remove sensitive data from headers for logging"""
        result = dict(headers)
        sensitive_keys = ["authorization", "cookie", "x-api-key"]

        for key in sensitive_keys:
            if key in result:
                result[key] = "[REDACTED]"

        return result
