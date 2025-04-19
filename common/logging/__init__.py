"""
Common logging module for Deep Recall
Provides centralized logging configuration with Loguru and OpenTelemetry.
"""

from common.logging.logger import setup_logger
from common.logging.tracing import get_tracer, setup_tracing

__all__ = ["setup_logger", "setup_tracing", "get_tracer"]
