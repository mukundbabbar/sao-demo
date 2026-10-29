#!/usr/bin/env python3
"""Start the banking /query API with OTel auto-instrumentation for Splunk APM.

Same pattern as galileo-golden-demo: opentelemetry-instrument creates the FastAPI
HTTP span, api.py stamps Splunk AO correlation attributes onto it, and the OTLP
exporter sends that span to Splunk APM.

Usage:
    python start_api.py
    PORT=8000 python start_api.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_PLACEHOLDERS = {"", "..."}


def _secret(secrets: dict, key: str) -> str:
    value = str(secrets.get(key, "") or "").strip()
    return "" if value in _PLACEHOLDERS else value


def _load_secrets() -> dict:
    try:
        import toml
    except ImportError:
        return {}
    path = Path(".streamlit/secrets.toml")
    if not path.exists():
        return {}
    return toml.load(path)


def configure_otel_from_secrets() -> None:
    """Set OTEL_* for the auto-instrumentor. Existing env wins."""
    secrets = _load_secrets()
    realm = _secret(secrets, "splunk_ao_realm")
    token = _secret(secrets, "splunk_ao_o11y_token")
    service = _secret(secrets, "otel_service_name") or "sao-demo-api"
    endpoint = _secret(secrets, "otel_exporter_otlp_endpoint")
    if not endpoint and realm:
        endpoint = f"https://ingest.{realm}.signalfx.com"

    os.environ.setdefault("OTEL_SERVICE_NAME", service)
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "otlp")
    os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
    os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    if endpoint:
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
    if token:
        os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", f"X-SF-TOKEN={token}")


def main() -> None:
    configure_otel_from_secrets()
    instrumentor = shutil.which("opentelemetry-instrument")
    if not instrumentor:
        sys.exit(
            "opentelemetry-instrument not found. Install deps with "
            "`pip install -r requirements.txt` (needs Python 3.11+)."
        )
    port = os.environ.get("PORT", "8000")
    os.execvp(
        instrumentor,
        [
            "opentelemetry-instrument",
            "uvicorn",
            "api:app",
            "--host",
            "0.0.0.0",
            "--port",
            port,
        ],
    )


if __name__ == "__main__":
    main()
