#!/usr/bin/env python3
# api/middleware/prometheus.py

import os
import time

import prometheus_client
from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Create metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP Request Latency", ["method", "endpoint"]
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress", "HTTP Requests In Progress", ["method", "endpoint"]
)

ERROR_COUNT = Counter(
    "http_request_errors_total",
    "Total HTTP Request Errors",
    ["method", "endpoint", "exception"],
)

# Only start the prometheus server if explicitly configured
if os.getenv("ENABLE_PROMETHEUS_SERVER", "false").lower() == "true":
    prometheus_client.start_http_server(9090)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics on API requests"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        """Process each request and record metrics"""
        method = request.method
        endpoint = self._get_endpoint_name(request)

        # Track requests in progress
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

        # Start the timer
        start_time = time.time()

        try:
            # Process the request
            response = await call_next(request)

            # Record metrics
            status_code = response.status_code
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status_code=status_code
            ).inc()

            # Record request latency
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

            return response
        except Exception as e:
            # Record exceptions
            ERROR_COUNT.labels(
                method=method, endpoint=endpoint, exception=type(e).__name__
            ).inc()
            raise
        finally:
            # Always decrement in-progress requests
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

    def _get_endpoint_name(self, request: Request) -> str:
        """
        Extract a clean endpoint pattern for metrics
        For example: /api/users/{id} instead of /api/users/123
        """
        for route in request.app.routes:
            if route.path_regex.match(request.url.path):
                return route.path

        # If no route matched (shouldn't happen), use the raw path
        return request.url.path
