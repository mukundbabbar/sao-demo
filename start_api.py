#!/usr/bin/env python3
"""Start the banking /query API with Splunk zero-code OpenTelemetry.

HTTP traces go to APM (caller correlation). SAO sessions stay on the SDK:
AI Assistant → Run Agent. SQL instrumentors are disabled so select/show
does not land in the agent session.

https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/splunk-ai-agent-monitoring/set-up-ai-agent-monitoring/zero-code-instrumentation

Usage:
    python start_api.py
    PORT=8000 python start_api.py
"""
from __future__ import annotations

import os
import shutil
import sys

# Zero-code GenAI + FastAPI, but not database clients (those were the
# select/show leaves inside the SAO session).
_DISABLED_INSTRUMENTATIONS = (
    "sqlalchemy",
    "psycopg",
    "psycopg2",
    "sqlite3",
)


def _merge_disabled_instrumentations() -> None:
    existing = [
        name.strip()
        for name in os.environ.get("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", "").split(",")
        if name.strip()
    ]
    for name in _DISABLED_INSTRUMENTATIONS:
        if name not in existing:
            existing.append(name)
    os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = ",".join(existing)


def main() -> None:
    os.environ.setdefault("OTEL_SERVICE_NAME", "sao-demo-api")
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")
    _merge_disabled_instrumentations()

    port = os.environ.get("PORT", "8000")
    uvicorn_args = [
        "uvicorn",
        "api:app",
        "--host",
        "0.0.0.0",
        "--port",
        port,
    ]

    instrument = shutil.which("opentelemetry-instrument")
    if instrument:
        os.execvp(instrument, ["opentelemetry-instrument", *uvicorn_args])

    uvicorn = shutil.which("uvicorn")
    if not uvicorn:
        sys.exit(
            "uvicorn not found. Install deps with "
            "`pip install -r requirements.txt` (needs Python 3.11+)."
        )
    print(
        "[otel] opentelemetry-instrument not on PATH; starting uvicorn without "
        "zero-code HTTP tracing. pip install opentelemetry-distro"
    )
    os.execvp(uvicorn, uvicorn_args)


if __name__ == "__main__":
    main()
