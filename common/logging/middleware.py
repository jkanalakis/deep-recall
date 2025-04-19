"""
Logging and tracing middleware for FastAPI
"""

import time
import uuid

from fastapi import Request
from loguru import logger
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import \
    TraceContextTextMapPropagator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses with OpenTelemetry trace correlation
    """

    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name
        self.propagator = TraceContextTextMapPropagator()

    async def dispatch(self, request: Request, call_next):
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Extract OpenTelemetry context from headers
        carrier = {}
        for key, value in request.headers.items():
            carrier[key] = value

        # Start timing
        start_time = time.time()

        # Get client IP
        client_host = request.client.host if request.client else "unknown"

        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path} from {client_host} "
            f"(request_id={request_id})"
        )

        # Add request ID to response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        # Calculate duration
        duration = time.time() - start_time
        duration_ms = round(duration * 1000, 2)

        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"status_code={response.status_code} duration={duration_ms}ms "
            f"(request_id={request_id})"
        )

        return response
