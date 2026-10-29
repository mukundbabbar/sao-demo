"""
Agent Control helper functions for runtime guardrails.
"""
import functools
import inspect
import json
import os
import threading
from contextvars import ContextVar, Token
from typing import Any, Callable, Optional

import agent_control
from agent_control import ControlSteerError, ControlViolationError, control
from agent_control.settings import configure_settings

_initialized = False
_correlation_attrs: ContextVar[dict] = ContextVar("sao_otel_corr", default={})
_correlation_attrs_fallback: dict = {}
_processor_provider_ids: set[int] = set()
MAX_STEER_RETRIES = 3
STEER_EXHAUSTED_MESSAGE = (
    "Agent could not complete the request after "
    f"{MAX_STEER_RETRIES} steering retries."
)


def extract_steering_instructions(error: Exception) -> str:
    """Return steering guidance from a ControlSteerError."""
    if isinstance(error, ControlSteerError):
        ctx = error.steering_context
        if isinstance(ctx, dict):
            return str(ctx.get("message") or ctx)
        if ctx:
            return str(ctx)
        if error.message:
            return str(error.message)
    return str(error)


def build_steer_correction_prompt(*, steering_instructions: str, previous_output: str = "") -> str:
    """Prompt text instructing the LLM to produce a corrected response."""
    parts = [
        "Your previous output was flagged by Agent Control and must be corrected.",
        "",
        "Agent Control instructions — follow these exactly and produce a completely "
        "new response that complies with them:",
        steering_instructions,
    ]
    if previous_output:
        parts.extend(["", "Your previous response:", previous_output])
    parts.extend([
        "",
        "Generate a revised response now. Do not repeat the previous response.",
    ])
    return "\n".join(parts)


def format_steer_tool_result(error: Exception) -> str:
    """JSON tool payload returned when a tool call is steered."""
    steering_instructions = extract_steering_instructions(error)
    return json.dumps(
        {
            "error": (
                "Tool call requires correction before execution: "
                f"{steering_instructions}"
            ),
            "steering_instructions": steering_instructions,
            "steered_by_agent_control": True,
        }
    )

# Hardcoded fallback when no domain tool list is available.
STANDARD_AGENT_CONTROL_STEPS = [
    {"type": "llm", "name": "Bank Assistant"},
    {"type": "tool", "name": "get_customer_info"},
    {"type": "tool", "name": "delete_customer_record"},
    {"type": "tool", "name": "retrieval_step"},
]

def format_blocked_message(
    error: Exception,
    step_name: str = "tool_step",
    *,
    steered: bool = False,
) -> str:
    """Return a user-friendly message for Agent Control blocks."""
    if steered:
        return (
            "This action was adjusted by Agent Control. "
            f"{error}"
        )
    return (
        "I'm sorry, this action was blocked by Agent Control. "
        "Please rephrase your request or try a different approach."
    )


def _serialize_messages(messages) -> list:
    serialized = []
    for msg in messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            serialized.append({"role": msg.type, "content": msg.content})
        elif isinstance(msg, dict):
            serialized.append(msg)
    return serialized


def _extract_trace_input(messages) -> str:
    """Return the latest user message for trace input (matches reference guardrails app)."""
    for msg in reversed(list(messages)):
        if hasattr(msg, "type") and msg.type == "human" and hasattr(msg, "content"):
            return str(msg.content)
        if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            return str(msg.get("content", ""))
    serialized = _serialize_messages(messages)
    return json.dumps(serialized) if serialized else ""


try:
    from opentelemetry.sdk.trace import SpanProcessor as _OtelSpanProcessor
except Exception:  # pragma: no cover - API-only environments
    class _OtelSpanProcessor:  # type: ignore[no-redef]
        pass


class _CorrelationSpanProcessor(_OtelSpanProcessor):
    """Stamp IDs on every OTel span SAO actually exports (invoke_agent, PII, …)."""

    def on_start(self, span, parent_context=None):
        try:
            attrs = _correlation_attrs.get() or _correlation_attrs_fallback
            if not attrs or span is None:
                return
            set_attr = getattr(span, "set_attribute", None)
            if set_attr is None:
                return
            for key, value in attrs.items():
                if value is None or value == "":
                    continue
                try:
                    set_attr(str(key), str(value))
                except Exception:
                    continue
            print(
                f"[otel] processor on_start span={getattr(span, 'name', None)!r} "
                f"attrs={attrs}"
            )
        except Exception as exc:
            print(f"[otel] processor on_start failed: {exc}")

    def on_end(self, span):
        return None

    def _on_ending(self, span):
        # OTel SDK 1.38+ calls _on_ending instead of on_end on custom processors.
        return self.on_end(span)

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=0):
        return True


def register_correlation_processor() -> None:
    """Attach the correlation processor to every TracerProvider we can find.

    SAO may replace the global provider after logger/AC init, so this is
    idempotent per provider id and should be called again after start_trace.
    """
    from opentelemetry import trace as otel_trace

    providers = []
    provider = otel_trace.get_tracer_provider()
    if provider is not None:
        providers.append(provider)
        for attr in ("_tracer_provider", "_provider", "tracer_provider"):
            inner = getattr(provider, attr, None)
            if inner is not None and inner not in providers:
                providers.append(inner)

    attached = False
    for candidate in providers:
        pid = id(candidate)
        if pid in _processor_provider_ids:
            attached = True
            continue
        adder = getattr(candidate, "add_span_processor", None)
        if adder is None:
            print(f"[otel] no add_span_processor on {type(candidate).__name__}")
            continue
        adder(_CorrelationSpanProcessor())
        _processor_provider_ids.add(pid)
        attached = True
        print(f"[otel] registered correlation processor on {type(candidate).__name__}")
    if not attached:
        print("[otel] correlation processor not attached (no TracerProvider yet)")


def set_correlation_attrs(attrs: Optional[dict]) -> Token:
    global _correlation_attrs_fallback
    register_correlation_processor()
    data = dict(attrs or {})
    _correlation_attrs_fallback = data
    return _correlation_attrs.set(data)


def reset_correlation_attrs(token: Optional[Token]) -> None:
    global _correlation_attrs_fallback
    _correlation_attrs_fallback = {}
    if token is None:
        return
    try:
        _correlation_attrs.reset(token)
    except Exception:
        pass


def stamp_otel_attributes(
    attrs: Optional[dict],
    *,
    where: str = "",
    span=None,
) -> None:
    """Set correlation IDs on the live OpenTelemetry span (not SAO metadata)."""
    from opentelemetry import trace as otel_trace

    if span is None:
        span = otel_trace.get_current_span()
    if span is None or not span.is_recording():
        print(f"[otel] skip stamp {where}: no recording span")
        return
    ctx = span.get_span_context()
    name = getattr(span, "name", None) or where
    stamped = {}
    for key, value in (attrs or {}).items():
        if value is None or value == "":
            continue
        span.set_attribute(str(key), str(value))
        stamped[str(key)] = str(value)
    print(
        f"[otel] stamped {where} span={name!r} "
        f"trace_id={format(ctx.trace_id, '032x') if ctx and ctx.trace_id else None} "
        f"attrs={stamped}"
    )


def stamp_logger_parent(sao_logger, attrs: Optional[dict], *, where: str = "sao-parent") -> None:
    """Stamp OTel attributes on the SAO parent span object, if it wraps one."""
    if not sao_logger:
        print(f"[otel] skip stamp {where}: no logger")
        return
    try:
        parent = sao_logger.current_parent()
    except Exception as exc:
        print(f"[otel] skip stamp {where}: current_parent error {exc}")
        return
    if parent is None:
        print(f"[otel] skip stamp {where}: no SAO parent")
        return
    print(
        f"[otel] {where} type={type(parent).__name__} "
        f"name={getattr(parent, 'name', None)!r}"
    )
    seen: set[int] = set()
    stack = [parent]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if callable(getattr(obj, "set_attribute", None)):
            stamp_otel_attributes(attrs, where=where, span=obj)
        for attr in ("span", "_span", "otel_span", "_otel_span", "current_span"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def string_metadata(raw: Optional[dict] = None) -> dict:
    """Flat string metadata for start_session / start_trace (SDK requires str keys/values)."""
    if not raw:
        return {}
    metadata = {}
    for key, value in raw.items():
        if value is None or value == "":
            continue
        metadata[str(key)[:256]] = str(value)[:256]
    return metadata


def _logger_ids(sao_logger) -> dict:
    if sao_logger is None:
        return {
            "project_id": None,
            "agent_stream_id": None,
            "session_id": None,
            "current_parent": None,
        }
    parent = None
    try:
        parent = sao_logger.current_parent()
        parent = getattr(parent, "name", None) or parent
    except Exception as exc:
        parent = f"<error {exc}>"
    return {
        "project_id": getattr(sao_logger, "project_id", None),
        "agent_stream_id": getattr(sao_logger, "agent_stream_id", None)
        or getattr(sao_logger, "log_stream_id", None),
        "session_id": getattr(sao_logger, "session_id", None),
        "current_parent": parent,
    }


def debug_sao(prefix: str, sao_logger=None, extra: Optional[dict] = None) -> None:
    ids = _logger_ids(sao_logger)
    print(
        f"[sao] {prefix} "
        f"project_id={ids['project_id']!r} "
        f"agent_stream_id={ids['agent_stream_id']!r} "
        f"session_id={ids['session_id']!r} "
        f"current_parent={ids['current_parent']!r} "
        f"metadata={extra}"
    )


def logger_id_metadata(sao_logger, extra: Optional[dict] = None) -> dict:
    """Project / agent-stream / session UUIDs from the live logger, not display names."""
    merged = dict(extra or {})
    ids = _logger_ids(sao_logger)
    if ids["project_id"]:
        merged["splunk_ao.project_id"] = ids["project_id"]
        merged["project_id"] = ids["project_id"]
    if ids["agent_stream_id"]:
        merged["splunk_ao.agent_stream_id"] = ids["agent_stream_id"]
        merged["agent_stream_id"] = ids["agent_stream_id"]
    if ids["session_id"]:
        merged["splunk_ao_user_session_id"] = ids["session_id"]
        merged["sao_session_id"] = ids["session_id"]
        merged["splunk_ao.session.id"] = ids["session_id"]
    project_name = os.environ.get("SPLUNK_AO_PROJECT")
    agent_stream_name = os.environ.get("SPLUNK_AO_AGENT_STREAM")
    if project_name:
        merged["splunk_ao.project.name"] = project_name
    if agent_stream_name:
        merged["splunk_ao.agentstream.name"] = agent_stream_name
    project_id = ids["project_id"] or merged.get("splunk_ao.project_id")
    stream_id = ids["agent_stream_id"] or merged.get("splunk_ao.agent_stream_id")
    if project_id and stream_id:
        realm = os.environ.get("SPLUNK_AO_REALM", "")
        console = (
            os.environ.get("SPLUNK_AO_CONSOLE_URL")
            or (f"https://app.{realm}.observability.splunkcloud.com" if realm else "")
        ).rstrip("/")
        if console:
            merged["splunk_ao.url"] = (
                f"{console}/project/{project_id}/agent-streams/{stream_id}"
            )
    return string_metadata(merged)


def ensure_trace_started(
    sao_logger,
    messages=None,
    *,
    trace_input: Optional[str] = None,
    trace_name: str = "Run Agent",
    metadata: Optional[dict] = None,
) -> None:
    """Start a Splunk AO trace when none is active (required before add_llm_span)."""
    if not sao_logger:
        print("[sao] start_trace skipped: no logger")
        return
    parent = None
    try:
        parent = sao_logger.current_parent()
    except Exception:
        parent = None
    if parent is not None:
        debug_sao("start_trace skipped (parent already active)", sao_logger, metadata)
        return
    if trace_input is None and messages is not None:
        trace_input = _extract_trace_input(messages)
    meta = logger_id_metadata(sao_logger, metadata)
    debug_sao(f"start_trace name={trace_name!r}", sao_logger, meta)
    kwargs = {"input": trace_input or "", "name": trace_name}
    if meta:
        kwargs["metadata"] = meta
    sao_logger.start_trace(**kwargs)
    register_correlation_processor()
    stamp_otel_attributes(meta, where="after start_trace")
    stamp_logger_parent(sao_logger, meta, where="after start_trace")


def finalize_trace(sao_logger, output: str) -> None:
    """Conclude and flush the active trace after a query completes."""
    if not sao_logger or sao_logger.current_parent() is None:
        return
    sao_logger.conclude(output=output)
    sao_logger.flush()


def notify_control_block(
    error: Exception,
    *,
    step_name: str,
    guardrail_result: str = "blocked",
) -> None:
    """Console notice for Agent Control blocks.

    Splunk AO LLM/tool spans come from LangChain callbacks (SplunkAOCallback).
    Control evaluation spans are emitted by enable_agent_control() on the logger.
    """
    label = "STEERED" if guardrail_result == "steered" else "BLOCKED"
    print(f"  🚫 {label} by Agent Control ({step_name}): {error}")


def infer_control_step_name(func_name: str) -> str:
    """Map tool function names to Agent Control step names."""
    if func_name.startswith("search_") or func_name.startswith("retrieve_"):
        return "retrieval_step"
    return func_name


def _handle_control_error(
    error: Exception,
    *,
    step_name: str,
    steered: bool = False,
) -> str:
    """Return a friendly JSON tool response for Agent Control blocks."""
    notify_control_block(
        error,
        step_name=step_name,
        guardrail_result="steered" if steered else "blocked",
    )
    if steered:
        return format_steer_tool_result(error)

    return json.dumps(
        {
            "error": format_blocked_message(error, step_name, steered=False),
            "blocked_by_agent_control": True,
        }
    )


def build_agent_control_steps(llm_step_name: str, tool_names: list[str]) -> list[dict]:
    """
    Build Agent Control step registrations for a domain.

    Tool step names must match @control(step_name=...) on tool handlers.
    Search/retrieval tools use the shared retrieval_step name.
    """
    steps: list[dict] = [{"type": "llm", "name": llm_step_name}]
    registered = {llm_step_name}

    for name in tool_names:
        step_name = infer_control_step_name(name)
        if step_name in registered:
            continue
        steps.append({"type": "tool", "name": step_name})
        registered.add(step_name)

    return steps


def _mark_as_agent_control_tool(func: Callable, step_name: str, tool_name: Optional[str] = None) -> str:
    """
    Mark a function as a tool before @control() infers step type.

    Agent Control uses func.tool_name (or func.name) to emit span_type=tool steps
    with input kwargs (e.g. input.sql). Without this, nested SQL handlers are
    treated as generic calls and tool-scoped controls never fire.
    """
    effective_tool_name = tool_name or step_name
    func.tool_name = effective_tool_name
    func.name = effective_tool_name
    return effective_tool_name


def make_controlled_tool(
    func: Callable,
    step_name: str,
    *,
    tool_name: Optional[str] = None,
) -> Callable:
    """Apply @control and return friendly JSON errors for tool guardrail blocks."""
    effective_tool_name = _mark_as_agent_control_tool(func, step_name, tool_name)
    controlled_fn = control(step_name=step_name)(func)
    controlled_fn.tool_name = effective_tool_name
    controlled_fn.name = effective_tool_name

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await controlled_fn(*args, **kwargs)
            except ControlViolationError as e:
                return _handle_control_error(e, step_name=step_name)
            except ControlSteerError as e:
                return _handle_control_error(e, step_name=step_name, steered=True)

        async_wrapper._agent_control_step = step_name  # type: ignore[attr-defined]
        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return controlled_fn(*args, **kwargs)
        except ControlViolationError as e:
            return _handle_control_error(e, step_name=step_name)
        except ControlSteerError as e:
            return _handle_control_error(e, step_name=step_name, steered=True)

    sync_wrapper._agent_control_step = step_name  # type: ignore[attr-defined]
    return sync_wrapper


def uses_internal_sql_control(func_name: str) -> bool:
    """Tools that run SQL via an internal _execute_* handler already under @control."""
    return (
        (func_name.startswith("get_") and func_name.endswith("_info"))
        or (func_name.startswith("delete_") and func_name.endswith("_record"))
    )


def domain_controlled_tool(
    step_name: str = "tool_step",
    *,
    resolve_logger: Optional[Callable[..., Any]] = None,
    tool_name: Optional[str] = None,
):
    """
    Decorator for domain logic.py tools.

    Applies @control, catches Agent Control errors, and returns a friendly JSON response.
    Tool spans and control spans are recorded via LangChain callbacks + enable_agent_control().
    """

    def decorator(func: Callable) -> Callable:
        return make_controlled_tool(func, step_name, tool_name=tool_name)

    return decorator


def wrap_controlled_tool(
    func: Callable,
    sao_logger,
    step_name: Optional[str] = None,
) -> Callable:
    """Catch ControlViolationError and return a friendly tool response."""
    step = step_name or infer_control_step_name(func.__name__)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ControlViolationError as e:
                notify_control_block(e, step_name=step)
                return json.dumps(
                    {
                        "error": format_blocked_message(e, step),
                        "blocked_by_agent_control": True,
                    }
                )
            except ControlSteerError as e:
                notify_control_block(e, step_name=step, guardrail_result="steered")
                return format_steer_tool_result(e)

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ControlViolationError as e:
            notify_control_block(e, step_name=step)
            return json.dumps(
                {
                    "error": format_blocked_message(e, step),
                    "blocked_by_agent_control": True,
                }
            )
        except ControlSteerError as e:
            notify_control_block(e, step_name=step, guardrail_result="steered")
            return format_steer_tool_result(e)

    return sync_wrapper


def wrap_controlled_langchain_tool(tool, sao_logger, step_name: str = "retrieval_step"):
    """Wrap a LangChain BaseTool coroutine/func with Agent Control error handling."""
    from langchain_core.tools import BaseTool

    if not isinstance(tool, BaseTool):
        return tool

    original = tool.coroutine or tool.func
    if original is None:
        return tool

    is_async = inspect.iscoroutinefunction(original)

    def _blocked_response(error, steered=False):
        notify_control_block(
            error,
            step_name=step_name,
            guardrail_result="steered" if steered else "blocked",
        )
        if steered:
            return build_steer_correction_prompt(
                steering_instructions=extract_steering_instructions(error)
            )
        return format_blocked_message(error, step_name, steered=steered)

    if is_async:

        @functools.wraps(original)
        async def async_wrapper(*args, **kwargs):
            try:
                return await original(*args, **kwargs)
            except ControlViolationError as e:
                return _blocked_response(e)
            except ControlSteerError as e:
                return _blocked_response(e, steered=True)

        return type(tool)(
            name=tool.name,
            description=tool.description,
            coroutine=async_wrapper,
            args_schema=tool.args_schema,
        )

    @functools.wraps(original)
    def sync_wrapper(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ControlViolationError as e:
            return _blocked_response(e)
        except ControlSteerError as e:
            return _blocked_response(e, steered=True)

    return type(tool)(
        name=tool.name,
        description=tool.description,
        func=sync_wrapper,
        args_schema=tool.args_schema,
    )


_sao_session_ids: dict[str, str] = {}
_sao_session_lock = threading.Lock()


def start_demo_session(
    sao_logger,
    *,
    name: str,
    external_id: str,
    metadata: Optional[dict] = None,
) -> Optional[str]:
    """Start or reuse a named SAO session (SDK reuses by external_id)."""
    if sao_logger is None:
        return None
    meta = string_metadata(metadata)
    kwargs = {"name": name, "external_id": external_id}
    if meta:
        kwargs["metadata"] = meta
    session_id = sao_logger.start_session(**kwargs)
    return session_id or getattr(sao_logger, "session_id", None)


def ensure_logger_session(
    sao_logger,
    *,
    name: str,
    external_id: str,
    metadata: Optional[dict] = None,
) -> Optional[str]:
    """Attach this browser visit's session on the current thread, then start traces.

    Galileo kept the session on the logger instance. SAO set_session is ambient
    for the current thread/async context, and /query runs process_query in a
    worker thread — so this must run on that same thread as start_trace.
    https://agent-observability-docs.splunk.com/sdk-api/python/reference/logger/logger
    """
    if sao_logger is None or not external_id:
        print(f"[sao] ensure_logger_session skipped logger={sao_logger} external_id={external_id!r}")
        return None
    with _sao_session_lock:
        cached = _sao_session_ids.get(external_id)
    if cached:
        sao_logger.set_session(cached)
        debug_sao(f"set_session cached name={name!r} external_id={external_id!r}", sao_logger, metadata)
        return cached
    session_meta = logger_id_metadata(sao_logger, metadata)
    debug_sao(f"start_session name={name!r} external_id={external_id!r}", sao_logger, session_meta)
    session_id = start_demo_session(
        sao_logger,
        name=name,
        external_id=external_id,
        metadata=session_meta,
    )
    if session_id:
        with _sao_session_lock:
            _sao_session_ids[external_id] = session_id
        sao_logger.set_session(session_id)
    debug_sao(f"start_session result sao_id={session_id!r}", sao_logger, session_meta)
    return session_id


def attach_control_bridge(sao_logger) -> None:
    """Register the Agent Control bridge on this thread after start_trace.

    Control spans are leaves on the active Splunk AO parent. If the bridge is
    missing, @control emits a standalone OTel span named after the control
    (e.g. PII_Input_output_Luna-clone) and SAO auto-creates a session named
    "session". Galileo's enable_agent_control() attached those as children of
    Run Agent instead.
    https://agent-observability-docs.splunk.com/sdk-api/python/reference/handlers/agent_control/bridge
    """
    if sao_logger is None:
        return
    try:
        sao_logger.enable_agent_control()
    except Exception as exc:
        print(f"⚠️ Agent Control bridge not attached: {exc}")


def _resolve_agent_stream_id(sao_logger) -> Optional[str]:
    """Return logger.agent_stream_id (UUID). Do not start a session here.

    Docs: target_id must be logger.agent_stream_id, not the stream name.
    Starting a session here would name it and steal later traces.
    https://agent-observability-docs.splunk.com/how-to-guides/agent-control/initialize-and-configure-agent-control
    """
    if sao_logger is None:
        return os.environ.get("SPLUNK_AO_AGENT_STREAM_ID")
    return (
        getattr(sao_logger, "agent_stream_id", None)
        or getattr(sao_logger, "log_stream_id", None)
        or os.environ.get("SPLUNK_AO_AGENT_STREAM_ID")
    )


def init_agent_control(
    sao_logger,
    project_name: str,
    agent_stream: str,
    agent_description: str = "Multi-domain demo agent",
    *,
    steps: Optional[list] = None,
    force: bool = False,
) -> bool:
    """Initialize or refresh Agent Control for the current Splunk AO logger/session."""
    global _initialized

    if sao_logger is None:
        print("⚠️ Agent Control not initialized (no Splunk AO logger)")
        return False

    server_url = os.environ.get("AGENT_CONTROL_URL")
    agent_name = os.environ.get("AGENT_CONTROL_AGENT_NAME")
    api_key = (
        os.environ.get("AGENT_CONTROL_API_KEY")
        or os.environ.get("SPLUNK_AO_API_KEY")
        or os.environ.get("SPLUNK_AO_O11Y_API_TOKEN")
        or os.environ.get("SPLUNK_AO_O11Y_TOKEN")
    )
    api_key_header = os.environ.get("AGENT_CONTROL_API_KEY_HEADER", "Splunk-AO-API-Key")
    # SAO Agent Control only auto-resolves log_stream targets (stream id, not name).
    target_type = os.environ.get("AGENT_CONTROL_TARGET_TYPE", "log_stream")
    if target_type == "agent_stream":
        target_type = "log_stream"

    if not all([server_url, agent_name, api_key]):
        print(
            "⚠️ Agent Control not configured "
            "(set AGENT_CONTROL_URL, AGENT_CONTROL_AGENT_NAME, and SPLUNK_AO_API_KEY)"
        )
        return False

    session_key = (agent_name, project_name, agent_stream, server_url)
    if not force and getattr(init_agent_control, "_last_session_key", None) == session_key:
        return True

    # Control spans require sao_logger.enable_agent_control() (done in app.py).

    target_id = _resolve_agent_stream_id(sao_logger)
    if not target_id:
        print(
            f"⚠️ Agent Control: failed to resolve agent stream '{agent_stream}' "
            "(call start_session first so logger.agent_stream_id is set)"
        )
        return False
    os.environ["SPLUNK_AO_AGENT_STREAM_ID"] = target_id
    project_id = getattr(sao_logger, "project_id", None)
    if project_id:
        os.environ["SPLUNK_AO_PROJECT_ID"] = project_id

    control_steps = steps or STANDARD_AGENT_CONTROL_STEPS

    configure_settings(
        url=server_url,
        api_key=api_key,
        api_key_header=api_key_header,
    )

    agent_control.init(
        agent_name=agent_name,
        agent_description=agent_description,
        server_url=server_url,
        api_key=api_key,
        api_key_header=api_key_header,
        observability_enabled=True,
        observability_sink_name="registered",
        target_type=target_type,
        target_id=target_id,
        steps=control_steps,
    )
    _initialized = True
    init_agent_control._last_session_key = session_key
    register_correlation_processor()
    step_names = ", ".join(s["name"] for s in control_steps)
    print(
        f"✅ Agent Control initialized for agent '{agent_name}' "
        f"(project={project_name}, agent_stream={agent_stream}, steps={step_names})"
    )
    return True
