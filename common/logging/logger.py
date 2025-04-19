"""
Logger configuration using Loguru for Deep Recall
Provides standardized logging across all services with configurable levels,
rotation policies, and output formats.
"""

import os
import sys
import json
from loguru import logger
from pathlib import Path


def setup_logger(
    service_name,
    log_level=None,
    log_file=None,
    log_format=None,
    json_logs=False,
    rotation="50 MB",
    retention="10 days",
    enqueue=True,
):
    """
    Configure Loguru logger for the specified service

    Args:
        service_name: Name of the service (inference, memory, orchestrator)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, if None logs to stderr only
        log_format: Custom log format string
        json_logs: Whether to output logs in JSON format
        rotation: Log rotation policy (size or time-based)
        retention: Log retention policy
        enqueue: Whether to enqueue logs (recommended for production)

    Returns:
        Configured logger instance
    """
    # Remove default loguru handler
    logger.remove()

    # Get log level from environment or parameter
    log_level = log_level or os.environ.get("LOG_LEVEL", "INFO").upper()

    # Define log format if not specified
    if not log_format:
        if json_logs:
            log_format = (
                lambda record: json.dumps(
                    {
                        "timestamp": record["time"].isoformat(),
                        "level": record["level"].name,
                        "service": service_name,
                        "message": record["message"],
                        "function": record["function"],
                        "file": f"{record['file'].name}:{record['line']}",
                        "trace_id": record["extra"].get("trace_id", ""),
                        "span_id": record["extra"].get("span_id", ""),
                        **record["extra"],
                    }
                )
                + "\n"
            )
        else:
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                f"<cyan>{service_name}</cyan> | "
                "<yellow>trace_id={extra[trace_id]}</yellow> | "
                "<yellow>span_id={extra[span_id]}</yellow> | "
                "<blue>{name}</blue>:<blue>{function}</blue>:<blue>{line}</blue> | "
                "<level>{message}</level>"
            )

    # Add stderr handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        enqueue=enqueue,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if log file specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            enqueue=enqueue,
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Initialized logger for {service_name} at level {log_level}")

    # Add correlation context for OpenTelemetry
    class CorrelationContextFilter:
        """Add trace and span info to logs"""

        def __call__(self, record):
            from opentelemetry import trace

            current_span = trace.get_current_span()
            if current_span:
                ctx = current_span.get_span_context()
                if ctx.is_valid:
                    record["extra"]["trace_id"] = f"{ctx.trace_id:032x}"
                    record["extra"]["span_id"] = f"{ctx.span_id:016x}"
                else:
                    record["extra"]["trace_id"] = ""
                    record["extra"]["span_id"] = ""
            else:
                record["extra"]["trace_id"] = ""
                record["extra"]["span_id"] = ""
            return True

    # Add filter for correlation IDs
    logger.configure(patcher=lambda record: CorrelationContextFilter()(record))

    return logger
