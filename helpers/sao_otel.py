"""SAO OpenTelemetry provider for span attributes only.

Do not instrument FastAPI, SQL, or HTTP clients. Those create POST /query,
select/show, and extra POST leaves and parent the agent tree under a Partial
HTTP trace. Galileo structure is session → Run Agent → LangChain spans from
the SAO SDK callback.

https://agent-observability-docs.splunk.com/how-to-guides/third-party-integrations/otel
"""
from __future__ import annotations

import os

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

_initialized = False


def isolate_http_parent():
    """Detach from FastAPI POST /query so Run Agent is its own root."""
    return otel_context.attach(otel_trace.set_span_in_context(otel_trace.INVALID_SPAN))


def restore_otel_context(token) -> None:
    if token is None:
        return
    try:
        otel_context.detach(token)
    except Exception:
        pass


def init_sao_otel() -> None:
    """Install SAO's TracerProvider so attributes can be stamped on agent spans."""
    global _initialized
    if _initialized:
        return

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

    otel_trace.set_tracer_provider(tracer_provider)
    register_correlation_processor()
    _initialized = True
    print(
        f"[otel] SAO TracerProvider ready (attributes only) "
        f"project={project!r} agent_stream={agent_stream!r}"
    )
