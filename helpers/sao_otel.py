"""HTTP/APM vs SAO session split (Galileo pattern).

Zero-code OpenTelemetry (`opentelemetry-instrument`) owns the HTTP trace:
caller `traceparent` → POST /query in Splunk APM, with splunk_ao.* attributes.

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


def _as_recording_span(obj):
    """Unwrap a SAO parent object to a recording OTel span, if present."""
    if obj is None:
        return None
    seen: set[int] = set()
    stack = [obj]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        is_recording = getattr(current, "is_recording", None)
        get_ctx = getattr(current, "get_span_context", None)
        if callable(is_recording) and callable(get_ctx):
            try:
                if is_recording():
                    return current
            except Exception:
                return current
        for attr in ("span", "_span", "otel_span", "_otel_span", "current_span"):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)
    return None


def activate_sao_parent(sao_logger):
    """Make logger.start_trace('Run Agent') the current OTel span.

    @control emits an OTel span. If nothing is current, that control becomes
    the root. Galileo's start_trace was the parent; SAO needs it activated.
    """
    parent = None
    if sao_logger is not None:
        try:
            parent = sao_logger.current_parent()
        except Exception as exc:
            print(f"[otel] current_parent failed: {exc}")
    span = _as_recording_span(parent)
    if span is None:
        current = otel_trace.get_current_span()
        if current is not None and current.is_recording():
            span = current
    if span is None:
        print("[otel] no Run Agent parent to activate")
        return None
    print(f"[otel] activate parent span={getattr(span, 'name', None)!r}")
    return otel_context.attach(otel_trace.set_span_in_context(span))


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
