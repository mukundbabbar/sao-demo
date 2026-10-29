"""Splunk AO OpenTelemetry provider and remote-client instrumentors.

SAO's logger.start_trace does not set an OpenTelemetry current span, so
get_current_span() is not recording and Bedrock/Postgres calls never
appear as CLIENT spans. Docs: attach SplunkAOSpanProcessor, then
OpenInference LLM instrumentors (not opentelemetry-instrument uvicorn).

https://agent-observability-docs.splunk.com/how-to-guides/third-party-integrations/otel
"""
from __future__ import annotations

import os

_initialized = False


def _try_instrument(label: str, fn) -> None:
    try:
        fn()
        print(f"[otel] {label} instrumented")
    except Exception as exc:
        print(f"[otel] {label} skipped: {exc}")


def init_sao_otel() -> None:
    """Install SAO's TracerProvider and instrument LLM + vector-DB clients once."""
    global _initialized
    if _initialized:
        return

    from opentelemetry import trace as trace_api
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource

    from helpers.agent_control_helpers import register_correlation_processor

    try:
        from splunk_ao import otel
    except Exception as exc:
        print(f"[otel] splunk_ao.otel unavailable: {exc}")
        return

    project = os.environ.get("SPLUNK_AO_PROJECT") or "sao-demo"
    agent_stream = os.environ.get("SPLUNK_AO_AGENT_STREAM") or "default"
    service_name = os.environ.get("OTEL_SERVICE_NAME") or project

    try:
        splunk_ao_span_processor = otel.SplunkAOSpanProcessor(
            project=project,
            agentstream=agent_stream,
        )
    except TypeError:
        splunk_ao_span_processor = otel.SplunkAOSpanProcessor()
    except Exception as exc:
        print(f"[otel] SplunkAOSpanProcessor failed: {exc}")
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "splunk_ao.project.name": project,
            "splunk_ao.agentstream.name": agent_stream,
        }
    )
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    try:
        otel.add_splunk_ao_span_processor(tracer_provider, splunk_ao_span_processor)
    except Exception as exc:
        print(f"[otel] add_splunk_ao_span_processor failed, using add_span_processor: {exc}")
        tracer_provider.add_span_processor(splunk_ao_span_processor)

    trace_api.set_tracer_provider(tracer_provider)
    register_correlation_processor()

    def _bedrock():
        from openinference.instrumentation.bedrock import BedrockInstrumentor

        BedrockInstrumentor().instrument(tracer_provider=tracer_provider)

    def _openai():
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    def _sqlalchemy():
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(tracer_provider=tracer_provider)

    def _psycopg():
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        PsycopgInstrumentor().instrument(tracer_provider=tracer_provider)

    def _httpx():
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)

    _try_instrument("Bedrock (LLM + embeddings)", _bedrock)
    _try_instrument("OpenAI", _openai)
    _try_instrument("SQLAlchemy (pgvector + SQL tools)", _sqlalchemy)
    _try_instrument("psycopg", _psycopg)
    _try_instrument("httpx (Ollama)", _httpx)

    _initialized = True
    print(
        f"[otel] SAO TracerProvider ready project={project!r} "
        f"agent_stream={agent_stream!r}"
    )
