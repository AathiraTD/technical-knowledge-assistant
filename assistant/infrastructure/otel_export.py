"""Lightweight OpenTelemetry export. No SDK, no dependencies.

Sends traces, metrics, and logs to an OTLP endpoint in JSON format.
If endpoint is not reachable, silently continues — observability failures must not break answers.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger("assistant")


def _get_endpoint() -> str:
    """Get OTLP endpoint from environment (read at runtime, not import time)."""
    return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()


def _get_service_name() -> str:
    """Get service name from environment."""
    return os.getenv("OTEL_SERVICE_NAME", "technical-knowledge-assistant")


def _get_service_version() -> str:
    """Get service version from environment."""
    return os.getenv("OTEL_SERVICE_VERSION", "1.0.0")


def _get_deployment_env() -> str:
    """Get deployment environment from environment."""
    return os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "development")


def _make_resource() -> dict:
    """Build OTel Resource (service metadata)."""
    return {
        "attributes": [
            {"key": "service.name", "value": {"stringValue": _get_service_name()}},
            {"key": "service.version", "value": {"stringValue": _get_service_version()}},
            {"key": "deployment.environment", "value": {"stringValue": _get_deployment_env()}},
        ]
    }


def export_trace(trace_id: str, span_id: str, parent_span_id: str, name: str,
                 started_at: str, duration_ms: int, status: str,
                 attributes: dict[str, Any]) -> None:
    """Export one span to OTLP in JSON format.

    Args:
        trace_id: OTel trace ID
        span_id: OTel span ID
        parent_span_id: parent span ID (empty if root)
        name: span name
        started_at: ISO 8601 timestamp
        duration_ms: span duration in milliseconds
        status: "ok" or "error"
        attributes: span attributes dict
    """
    if not _get_endpoint():
        return

    # Convert started_at ISO string to Unix nano (best effort)
    try:
        dt = time.strptime(started_at[:19], "%Y-%m-%dT%H:%M:%S")
        start_unix_nano = int(time.mktime(dt) * 1_000_000_000)
    except (ValueError, OSError):
        start_unix_nano = int(time.time() * 1_000_000_000)

    end_unix_nano = start_unix_nano + (duration_ms * 1_000_000)

    # Convert attributes to OTel format
    otel_attrs = []
    for key, value in attributes.items():
        if isinstance(value, str):
            otel_attrs.append({"key": key, "value": {"stringValue": str(value)}})
        elif isinstance(value, (int, float)):
            otel_attrs.append({"key": key, "value": {"doubleValue": float(value)}})
        elif isinstance(value, bool):
            otel_attrs.append({"key": key, "value": {"boolValue": value}})

    payload = {
        "resourceSpans": [
            {
                "resource": _make_resource(),
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "parentSpanId": parent_span_id,
                                "name": name,
                                "startTimeUnixNano": start_unix_nano,
                                "endTimeUnixNano": end_unix_nano,
                                "attributes": otel_attrs,
                                "status": {"code": 2 if status == "error" else 0},  # 0=ok, 2=error
                            }
                        ]
                    }
                ]
            }
        ]
    }

    _send_otlp("traces", payload)


def export_metrics(metrics_data: dict[str, Any]) -> None:
    """Export metrics to OTLP in JSON format.

    Args:
        metrics_data: dict with metric definitions (name, type, values)
    """
    if not _get_endpoint():
        return

    payload = {
        "resourceMetrics": [
            {
                "resource": _make_resource(),
                "scopeMetrics": [
                    {
                        "metrics": metrics_data.get("metrics", [])
                    }
                ]
            }
        ]
    }

    _send_otlp("metrics", payload)


def export_log(level: str, message: str, timestamp: str,
               trace_id: str = "", span_id: str = "",
               attributes: dict[str, Any] | None = None) -> None:
    """Export a log entry to OTLP in JSON format.

    Args:
        level: log level (INFO, WARNING, ERROR, etc.)
        message: log message
        timestamp: ISO 8601 timestamp
        trace_id: OTel trace ID (optional)
        span_id: OTel span ID (optional)
        attributes: additional log attributes
    """
    if not _get_endpoint():
        return

    # Convert timestamp to Unix nano
    try:
        dt = time.strptime(timestamp[:19], "%Y-%m-%dT%H:%M:%S")
        time_unix_nano = int(time.mktime(dt) * 1_000_000_000)
    except (ValueError, OSError):
        time_unix_nano = int(time.time() * 1_000_000_000)

    # Convert log level to OTel severity number
    severity_map = {
        "TRACE": 1, "DEBUG": 5, "INFO": 9, "WARNING": 13,
        "ERROR": 17, "FATAL": 21
    }
    severity = severity_map.get(level.upper(), 9)

    # Convert attributes to OTel format
    otel_attrs = []
    if attributes:
        for key, value in attributes.items():
            if isinstance(value, str):
                otel_attrs.append({"key": key, "value": {"stringValue": str(value)}})
            elif isinstance(value, (int, float)):
                otel_attrs.append({"key": key, "value": {"doubleValue": float(value)}})
            elif isinstance(value, bool):
                otel_attrs.append({"key": key, "value": {"boolValue": value}})

    payload = {
        "resourceLogs": [
            {
                "resource": _make_resource(),
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": time_unix_nano,
                                "severityNumber": severity,
                                "severityText": level,
                                "body": {"stringValue": message},
                                "attributes": otel_attrs,
                                "traceId": trace_id,
                                "spanId": span_id,
                            }
                        ]
                    }
                ]
            }
        ]
    }

    _send_otlp("logs", payload)


def _send_otlp(path: str, payload: dict) -> None:
    """Send JSON payload to OTLP endpoint.

    Args:
        path: "traces", "metrics", or "logs"
        payload: dict to send as JSON
    """
    endpoint = _get_endpoint()
    if not endpoint:
        return

    try:
        url = f"{endpoint.rstrip('/')}/v1/{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()  # Consume response
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        # Silently continue; observability must not fail answers
        logger.debug(f"OTLP export failed: {e}")
    except Exception as e:  # pragma: no cover - defensive
        # Catch any other exception, still continue
        logger.debug(f"Unexpected OTLP error: {e}")
