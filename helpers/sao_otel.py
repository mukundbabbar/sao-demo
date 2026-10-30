"""OpenTelemetry glue for the SAO demo.

How the pieces fit (verified against splunk-ao 0.6.1 source):

* Splunk AO spans (Run Agent, LangChain nodes, Agent Control spans) are built
  by the SDK and handed to the SDK's own sink. They never pass through the
  global TracerProvider, so global span processors cannot stamp them.
  ``install_span_attributes`` wraps that sink instead and adds plain OTel
  attributes to every exported span.
* Zero-code OpenTelemetry (``opentelemetry-instrument``) owns the HTTP trace
  (POST /query -> Splunk APM). The agent run is detached from that HTTP span so
  the Splunk AO trace is complete rather than "Partial".
* A Splunk AO "session" is the ``gen_ai.conversation.id`` attribute on spans.
  ``logger.start_session()`` additionally creates a separate session record,
  which is what showed up as an empty extra session. We only set the id.

https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/splunk-ai-agent-monitoring/set-up-ai-agent-monitoring/zero-code-instrumentation
"""
from __future__ import annotations

import hashlib
import uuid

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan

_initialized = False
# id(logger) -> attribute dict stamped on every span that logger exports
_ATTR_HOLDERS: dict[int, dict] = {}


def isolate_http_parent():
    """Detach from POST /query so the SAO run is not an HTTP child (Partial trace)."""
    return otel_context.attach(otel_trace.set_span_in_context(otel_trace.INVALID_SPAN))


def restore_otel_context(token) -> None:
    if token is None:
        return
    try:
        otel_context.detach(token)
    except Exception:
        pass


def conversation_id_for(tab_id: str) -> str:
    """Stable gen_ai.conversation.id for one browser tab / Streamlit session.

    Must be a version-4 UUID: the SDK's LoggedControlSpan (and other span
    models) reject any other UUID version for session_id, which silently
    dropped every control span. Derive it deterministically from the tab id
    (SHA-256, used only as a stable mapping, not for security) and stamp the
    version/variant bits so it is a valid v4 UUID.
    """
    digest = hashlib.sha256(f"sao-demo/{tab_id}".encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


def _with_attributes(span: ReadableSpan, extra: dict) -> ReadableSpan:
    merged = dict(extra)
    merged.update(dict(span.attributes or {}))
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=merged,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def install_span_attributes(sao_logger) -> dict:
    """Return this logger's attribute dict, wrapping its span sink on first use.

    Everything put in the returned dict becomes an OTel attribute on every span
    the logger exports (Run Agent, chat, tool, control, ...).
    """
    key = id(sao_logger)
    holder = _ATTR_HOLDERS.get(key)
    if holder is not None:
        return holder

    holder = {}
    _ATTR_HOLDERS[key] = holder

    sink = getattr(sao_logger, "_sink", None)
    if sink is None or not hasattr(sink, "emit"):
        print("[otel] logger has no span sink; attributes will not be added to SAO spans")
        return holder

    original_emit = sink.emit

    def emit(span):
        extra = {k: v for k, v in holder.items() if v not in (None, "")}
        if extra:
            try:
                span = _with_attributes(span, extra)
            except Exception as exc:  # never lose a span because of stamping
                print(f"[otel] could not add attributes to span {getattr(span, 'name', None)!r}: {exc}")
        return original_emit(span)

    sink.emit = emit
    return holder


def init_sao_otel() -> None:
    """Placeholder kept for callers; nothing global to register any more."""
    global _initialized
    if _initialized:
        return
    _initialized = True
    provider = type(otel_trace.get_tracer_provider()).__name__
    print(f"[otel] global TracerProvider: {provider} (zero-code HTTP tracing only)")
