"""HTTP/APM vs SAO session split (Galileo pattern).

Zero-code OpenTelemetry (`opentelemetry-instrument`) owns the HTTP trace:
caller `traceparent` → POST /query in Splunk APM, with splunk_ao.* attributes.

Do not attach SplunkAOSpanProcessor to the global TracerProvider — that is
what dumped POST /query, select/show, and fastapi.endpoint into the SAO
session.

https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/splunk-ai-agent-monitoring/set-up-ai-agent-monitoring/zero-code-instrumentation
"""
from __future__ import annotations

from contextlib import contextmanager

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


@contextmanager
def run_agent_span(name: str = "Run Agent"):
    """Complete root span named Run Agent (Galileo trace name).

    Isolate POST /query first so this is not a Partial HTTP child. Agent
    Control @control spans then nest under Run Agent instead of becoming
    the root.
    """
    http_token = isolate_http_parent()
    tracer = otel_trace.get_tracer("sao-demo")
    span = tracer.start_span(name)
    span_token = otel_context.attach(otel_trace.set_span_in_context(span))
    try:
        yield span
    finally:
        if span.is_recording():
            span.end()
        restore_otel_context(span_token)
        restore_otel_context(http_token)


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
