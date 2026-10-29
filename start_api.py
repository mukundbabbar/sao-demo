#!/usr/bin/env python3
"""Start the banking /query API with OTel auto-instrumentation.

opentelemetry-instrument creates the FastAPI HTTP span using the Python
agent's own OTLP defaults. api.py stamps Splunk AO correlation attributes
onto that span.

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
    """Set OTEL_SERVICE_NAME if present. Leave exporter settings to the agent."""
    secrets = _load_secrets()
    service = _secret(secrets, "otel_service_name") or "sao-demo-api"
    os.environ.setdefault("OTEL_SERVICE_NAME", service)


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
