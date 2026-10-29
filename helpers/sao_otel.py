"""HTTP/APM vs SAO session split (Galileo pattern).

Zero-code OpenTelemetry (`opentelemetry-instrument`) owns the HTTP trace:
caller `traceparent` → POST /query in Splunk APM, with splunk_ao.* attributes.

The SAO SDK owns the agent session: AI Assistant → Run Agent → LangChain.
Do not attach SplunkAOSpanProcessor to the global TracerProvider — that is
what dumped POST /query, select/show, and fastapi.endpoint into the SAO
session.

https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/splunk-ai-agent-monitoring/set-up-ai-agent-monitoring/zero-code-instrumentation
"""
from __future__ import annotations

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

_initialized = False


def isolate_http_parent():
    """Detach from POST /query so SAO Run Agent is not an HTTP child."""
    return otel_context.attach(otel_trace.set_span_in_context(otel_trace.INVALID_SPAN))


def restore_otel_context(token) -> None:
    if token is None:
        return
    try:
        otel_context.detach(token)
    except Exception:
        pass


def init_sao_otel() -> None:
    """Stamp IDs on whatever TracerProvider zero-code already installed."""
    global _initialized
    if _initialized:
        return

    from helpers.agent_control_helpers import register_correlation_processor

    register_correlation_processor()
    _initialized = True
    provider = type(otel_trace.get_tracer_provider()).__name__
    print(f"[otel] correlation processor on existing TracerProvider ({provider})")
