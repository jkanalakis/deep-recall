"""
OpenTelemetry tracing configuration for Deep Recall
Provides distributed tracing across all services.
"""

import os
from typing import Optional, Dict
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def setup_tracing(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    debug: bool = False,
    resource_attributes: Optional[Dict[str, str]] = None,
):
    """
    Configure OpenTelemetry tracing for the specified service

    Args:
        service_name: Name of the service (inference, memory, orchestrator)
        otlp_endpoint: Optional OTLP endpoint for exporting traces
        debug: Whether to enable debug mode with console output
        resource_attributes: Additional resource attributes for spans

    Returns:
        Configured tracer provider
    """
    # Get OTLP endpoint from environment if not provided
    otlp_endpoint = otlp_endpoint or os.environ.get("OTLP_ENDPOINT")

    # Create resource with service information
    attributes = {
        "service.name": service_name,
        "service.namespace": "deep-recall",
        "deployment.environment": os.environ.get("DEPLOYMENT_ENV", "development"),
    }

    # Add custom resource attributes if provided
    if resource_attributes:
        attributes.update(resource_attributes)

    resource = Resource.create(attributes)

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Add exporters
    if otlp_endpoint:
        # OTLP exporter for production
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if debug:
        # Console exporter for debugging
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider


def get_tracer(module_name):
    """
    Get a tracer for the specified module

    Args:
        module_name: Name of the module

    Returns:
        Tracer instance
    """
    return trace.get_tracer(module_name)


def instrument_fastapi(app):
    """
    Instrument a FastAPI application

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)


def instrument_httpx_client(client):
    """
    Instrument an HTTPX client

    Args:
        client: HTTPX client instance
    """
    HTTPXInstrumentor.instrument_client(client)


def instrument_sqlalchemy(engine):
    """
    Instrument a SQLAlchemy engine

    Args:
        engine: SQLAlchemy engine instance
    """
    SQLAlchemyInstrumentor.instrument(engine=engine)


def instrument_redis(client):
    """
    Instrument a Redis client

    Args:
        client: Redis client instance
    """
    RedisInstrumentor.instrument(client=client)
