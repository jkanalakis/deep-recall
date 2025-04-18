# Deep Recall Logging and Tracing

This document describes the logging and distributed tracing configuration for the Deep Recall system.

## Overview

Deep Recall uses a comprehensive logging and tracing approach:

- **Centralized Logging**: Using Loguru with structured logging and correlation to trace IDs
- **Distributed Tracing**: Using OpenTelemetry for end-to-end request tracing across services
- **Observability Pipeline**: OpenTelemetry Collector for processing and routing telemetry data
- **Visualization**: Jaeger and Zipkin for trace visualization, Grafana for dashboards

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Inference  │     │   Memory    │     │Orchestrator │
│   Service   │     │   Service   │     │   Service   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │    Log & Trace    │    Log & Trace    │
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────┐
│            OpenTelemetry Collector              │
└──────┬──────────────────┬──────────────┬────────┘
       │                  │              │
       ▼                  ▼              ▼
┌──────────┐       ┌──────────┐    ┌──────────┐
│          │       │          │    │          │
│  Jaeger  │       │  Zipkin  │    │Prometheus│
│          │       │          │    │          │
└──────────┘       └──────────┘    └──────────┘
```

## Features

### Centralized Logging (Loguru)

- **Structured Logging**: Configurable JSON or human-readable format
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Rotation**: Size or time-based rotation policies
- **Context Correlation**: Automatic correlation with trace/span IDs
- **Request Context**: Inclusion of request/correlation IDs in logs

### Distributed Tracing (OpenTelemetry)

- **End-to-End Tracing**: Trace requests across service boundaries
- **Context Propagation**: W3C TraceContext propagation format
- **Automatic Instrumentation**: FastAPI, HTTPX, SQLAlchemy, Redis
- **Custom Spans**: Manual instrumentation for critical operations
- **Span Attributes**: Enriched spans with useful attributes

## Usage

### Logging

```python
from common.logging import setup_logger

# Initialize logger
logger = setup_logger(
    service_name="my-service",
    log_level="DEBUG",
    log_file="/var/log/deep-recall/my-service.log",
    json_logs=True
)

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Log with additional context
logger.info("Request processed", extra={"user_id": "user123", "request_id": "req456"})
```

### Tracing

```python
from common.logging import setup_tracing, get_tracer

# Initialize tracing
setup_tracing(
    service_name="my-service",
    otlp_endpoint="http://otel-collector:4317"
)

# Get a tracer
tracer = get_tracer(__name__)

# Create spans
with tracer.start_as_current_span("operation_name") as span:
    # Add attributes
    span.set_attribute("attribute_key", "attribute_value")
    
    # Do work
    result = process_something()
    
    # Add result information
    span.set_attribute("result.status", "success")
    span.set_attribute("result.count", len(result))
```

### Instrument FastAPI

```python
from fastapi import FastAPI
from common.logging.tracing import instrument_fastapi

app = FastAPI()
instrument_fastapi(app)
```

## Configuration

### Docker Environment Variables

```
# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/app.log
JSON_LOGS=true

# Tracing
OTLP_ENDPOINT=http://otel-collector:4317
DEPLOYMENT_ENV=production
DEBUG_TRACING=false
```

### Kubernetes Setup

1. Deploy the OpenTelemetry Collector:
   ```bash
   kubectl apply -f deployments/kubernetes/base/otel-collector.yaml
   ```

2. Deploy Jaeger:
   ```bash
   kubectl apply -f deployments/kubernetes/base/jaeger.yaml
   ```

3. Set environment variables in deployments to enable tracing

## Viewing Traces

- **Jaeger UI**: Available at http://localhost:16686 (when running locally)
- **Zipkin UI**: Available at http://localhost:9411 (when running locally)

## Best Practices

1. **Use Structured Logging**: Always use structured logging with relevant context
2. **Create Meaningful Spans**: Create spans for important operations
3. **Add Span Attributes**: Enrich spans with relevant attributes
4. **Propagate Context**: Ensure context is propagated across service boundaries
5. **Balance Detail**: Too much logging/tracing can impact performance

## Troubleshooting

If traces are not appearing in Jaeger/Zipkin:
1. Check that the OTLP_ENDPOINT is correctly configured
2. Verify that the OpenTelemetry Collector is running
3. Check for errors in the OpenTelemetry Collector logs
4. Ensure context propagation headers are properly forwarded 